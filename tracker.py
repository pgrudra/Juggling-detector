import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
import random

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video file")
else:
    print("Video opened successfully.")

TRAIL_LENGTH = 5
FADE_FACTOR = 1.0 / TRAIL_LENGTH
MAX_LOST = 7
track_id = 0

def create_kalman(x, y):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.x[:2] = np.array([[x], [y]])
    kf.P *= 1000.
    kf.R *= 5.
    kf.Q *= 0.1
    return kf

def generate_color(seed):
    random.seed(seed)
    return tuple(random.randint(100, 255) for _ in range(3))

active_tracks = []

def is_ball_in_hand(y_coord, frame_height):
    return y_coord > 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height = frame.shape[0]
    overlay = frame.copy()
    detections = []
    results = model(frame, verbose=False)[0]

    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.4:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            detections.append((cx, cy))

    predicted = []
    for track in active_tracks:
        track["kf"].predict()
        px, py = track["kf"].x[:2].reshape(-1)
        predicted.append((px, py))

    if predicted and detections:
        cost = np.linalg.norm(np.array(predicted)[:, None] - np.array(detections), axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        row_ind, col_ind = [], []

    assigned_dets = set()
    assigned_preds = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r][c] < 50:
            track = active_tracks[r]
            det = detections[c]
            track["kf"].update(np.array(det))
            track["lost"] = 0
            cx, cy = int(track["kf"].x[0]), int(track["kf"].x[1])
            if not is_ball_in_hand(cy, frame_height):
                track["trail"].append((cx, cy))
            assigned_preds.add(r)
            assigned_dets.add(c)

    for i, det in enumerate(detections):
        if i not in assigned_dets:
            kf = create_kalman(*det)
            trail = deque(maxlen=TRAIL_LENGTH)
            cx, cy = det
            if not is_ball_in_hand(cy, frame_height):
                trail.append((cx, cy))
            color = generate_color(track_id)
            active_tracks.append({"id": track_id, "kf": kf, "lost": 0, "trail": trail, "color": color})
            track_id += 1

    for i, track in enumerate(active_tracks):
        if i not in assigned_preds:
            track["lost"] += 1

    active_tracks = [t for t in active_tracks if t["lost"] <= MAX_LOST]

    for t in active_tracks:
        color = t["color"]
        if "trail" in t and len(t["trail"]) > 1:
            for i, (x, y) in enumerate(t["trail"]):
                alpha = 1.0 - (i * FADE_FACTOR)
                circle_color = tuple(int(c * alpha) for c in color)
                cv2.circle(overlay, (x, y), 20, circle_color, -1)

        x, y = t["kf"].x[:2]
        cv2.circle(overlay, (int(x), int(y)), 40, color, -1)
        cv2.putText(overlay, f'ID: {t["id"]}', (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    blended = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    cv2.imshow("Ball Tracker", blended)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
