import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

########################################################
# USER CONFIG
########################################################
YOLO_MODEL_PATH = "PTs/best_tuning.pt"  # Path to your YOLO model
CLASS_NAMES = ["ball", "fmo_ball","golf_club", "golf_club_handle"]

# Single local video for behind-the-golfer view
VIDEO_PATH = "Test_videos/TV2_bh.mp4"

# YOLO thresholds
CONF_THRESHOLD = 0.28
IOU_THRESHOLD = 0.3

# If you want to track ONLY the ball, set True
TRACK_BALL_ONLY = False

# Force GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Deep SORT config for stable IDs
DEEPSORT_CONFIG = {
    "max_age": 35,
    "n_init": 3,
    "max_iou_distance": 0.4,
    "max_cosine_distance": 0.5,
}

# Optional class-based color map
CLASS_COLORS = {
    "golf_club":        (0, 255, 255),  # Yellow
    "golf_club_handle": (255, 0, 255),  # Magenta
    "ball":             (0, 255, 0),    # Green
    "fmo_ball":         (0, 0, 255)     # Red
}


########################################################
# OPTIONAL PREPROCESS (Gamma + CLAHE)
########################################################
def preprocess_frame(frame):
    """
    Adjust brightness/contrast or do denoising/sharpening if desired.
    Example: gamma correction + CLAHE
    """
    gamma = 1.2
    lut = np.empty((1,256), np.uint8)
    for i in range(256):
        lut[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    frame_gamma = cv2.LUT(frame, lut)

    lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return frame_eq


########################################################
# 2D Kalman Filter: [x, y, vx, vy]
########################################################
class Kalman2D:
    def __init__(self, dt=1/30, process_noise=5.0, measurement_noise=10.0):
        self.dt = dt
        # state = [x, y, vx, vy]
        self.state = np.zeros((4,1), dtype=np.float32)

        # Transition matrix
        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ]
        ], dtype=np.float32)

        # Process noise
        self.Q = np.eye(4, dtype=np.float32) * process_noise

        # Measurement matrix: we measure [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Measurement noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        # Covariance
        self.P = np.eye(4, dtype=np.float32) * 500

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, xy):
        mx, my = xy
        z = np.array([mx, my], dtype=np.float32).reshape(2,1)

        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.state


########################################################
# TrackKalman2D: per-track instance
########################################################
class TrackKalman2D:
    def __init__(self):
        self.kf = Kalman2D(dt=1/30, process_noise=5.0, measurement_noise=10.0)
        self.path = []

    def predict(self):
        self.kf.predict()

    def update(self, xy):
        self.kf.update(xy)
        x, y, vx, vy = self.kf.state.flatten()
        self.path.append((x, y))

    def get_latest_position(self):
        x, y, vx, vy = self.kf.state.flatten()
        return (x, y)


########################################################
# PROCESS A FRAME
########################################################
def process_frame(frame, model, deepsort, kalman_dict):
    # Preprocess
    frame_pp = preprocess_frame(frame)

    # Resize for YOLO
    frame_resized = cv2.resize(frame_pp, (640,640))

    # YOLO inference on GPU if available
    results = model.predict(frame_resized, device=DEVICE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

    # Parse detections for Deep SORT
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if conf < CONF_THRESHOLD:
            continue
        if 0 <= cls_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[cls_id]
        else:
            continue

        # If we only want to track the ball:
        if TRACK_BALL_ONLY and class_name not in ["ball", "fmo_ball"]:
            continue

        detections.append(([x1, y1, w, h], conf, class_name))

    # Run Deep SORT
    tracks = deepsort.update_tracks(detections, frame=frame_resized)

    # Annotate + Kalman
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx = (l + r) / 2
        cy = (t + b) / 2
        label = track.get_det_class()

        # If we only want to display ball, skip if not ball
        if TRACK_BALL_ONLY and label not in ["ball", "fmo_ball"]:
            continue

        # Kalman dict
        if track_id not in kalman_dict:
            kalman_dict[track_id] = TrackKalman2D()

        kalman = kalman_dict[track_id]
        kalman.predict()
        kalman.update((cx, cy))
        kf_x, kf_y = kalman.get_latest_position()

        # Draw bounding box
        color = CLASS_COLORS.get(label, (255,255,255))
        cv2.rectangle(frame_resized, (int(l), int(t)), (int(r), int(b)), color, 2)
        cv2.putText(frame_resized, f"{label} ID {track_id}", (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Kalman predicted center
        cv2.circle(frame_resized, (int(kf_x), int(kf_y)), 5, (0,0,255), -1)

        # Path
        path_pts = kalman.path
        for i in range(len(path_pts)-1):
            p1 = (int(path_pts[i][0]), int(path_pts[i][1]))
            p2 = (int(path_pts[i+1][0]), int(path_pts[i+1][1]))
            cv2.line(frame_resized, p1, p2, (0,0,255), 2)

    return frame_resized  # to display


########################################################
# MAIN
########################################################
def main():
    print(f"[INFO] Using device={DEVICE}")
    model = YOLO(YOLO_MODEL_PATH)
    model.to(DEVICE)

    # Deep SORT
    deepsort = DeepSort(
        max_iou_distance=DEEPSORT_CONFIG["max_iou_distance"],
        max_age=DEEPSORT_CONFIG["max_age"],
        n_init=DEEPSORT_CONFIG["n_init"],
        max_cosine_distance=DEEPSORT_CONFIG["max_cosine_distance"]
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {VIDEO_PATH}")
        return

    # Per-track Kalman state
    kalman_dict = {}

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or no frames.")
            break

        annotated = process_frame(frame, model, deepsort, kalman_dict)
        cv2.imshow("Golf Ball Tracking", annotated)

        # Wait enough time so Windows can handle the GUI events
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done single-cam golf ball tracking on Windows.")


if __name__ == "__main__":
    main()
