import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
import time

############################################
# CONFIGURATION
############################################

VIDEO_PATH = "Test_videos/TV2_bh.mp4"    # Path to input video
YOLO_MODEL_PATH = "PTs/best_tuning.pt"   # Path to YOLO weights

# Classes:
#  0 -> golf ball
#  2 -> golf club
BALL_CLASS_ID = 0
CLUB_CLASS_ID = 2

# Confidence & IOU thresholds for YOLO
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3

# For tracking & physics
TARGET_FPS = 60
GRAVITY = 9.81  # m/s^2
ROI_MARGIN = 100  # ROI margin in pixels

# Pixel-to-meter scale (calibration).
PIXELS_PER_METER = 50.0

# Launch height above ground (if any)
LAUNCH_HEIGHT = 0.0

# WIND (only manual user inputs)
wind_speed_m_s = 2.0   # m/s
wind_dir_deg   = 0.0   # 0 => wind blowing +X direction

############################################
# 1. YOLO MODEL LOADING
############################################
model = YOLO(YOLO_MODEL_PATH)
model.fuse()  # Optional speed-up
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

############################################
# 2. ADVANCED PREPROCESSING
############################################
def advanced_preprocess(frame):
    """
    Apply several preprocessing steps:
      - Gamma correction, CLAHE, Gaussian blur, unsharp mask.
      - Convert to HSV and apply a morphological top-hat to highlight small bright objects.
    """
    # Basic enhancements
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    frame_gamma = cv2.LUT(frame, table)
    
    lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    frame_denoised = cv2.GaussianBlur(frame_clahe, (3,3), 0)
    blur = cv2.GaussianBlur(frame_denoised, (0,0), 3)
    frame_sharp = cv2.addWeighted(frame_denoised, 1.5, blur, -0.5, 0)

    # Morphological top-hat in HSV to highlight small bright objects
    hsv = cv2.cvtColor(frame_sharp, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    tophat = cv2.morphologyEx(hsv, cv2.MORPH_TOPHAT, kernel)
    hsv_combined = cv2.addWeighted(hsv, 1.0, tophat, 0.5, 0)
    processed = cv2.cvtColor(hsv_combined, cv2.COLOR_HSV2BGR)
    return processed

############################################
# 3. YOLO DETECTION
############################################
def detect_objects(frame):
    """
    Run YOLO on the full frame.
    Returns (ball_bbox, club_bbox) in [x1, y1, x2, y2] or None if not found.
    """
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    ball_bbox = None
    club_bbox = None
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        if conf < CONF_THRESHOLD:
            continue
        if cls_id == BALL_CLASS_ID:
            ball_bbox = [x1, y1, x2, y2]
        elif cls_id == CLUB_CLASS_ID:
            club_bbox = [x1, y1, x2, y2]
    return ball_bbox, club_bbox

############################################
# 4. ROI DETECTION
############################################
def detect_object_in_roi(frame, roi_center, margin, desired_class):
    """
    Crop an ROI around roi_center and run YOLO on that ROI.
    Returns the bounding box in full-frame coordinates or None.
    """
    h, w, _ = frame.shape
    cx, cy = roi_center
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(w, cx + margin)
    y2 = min(h, cy + margin)
    
    roi = frame[y1:y2, x1:x2]
    # If ROI is empty, return None immediately
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return None

    results = model(roi, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        if cls_id == desired_class:
            bx1, by1, bx2, by2 = map(int, det.xyxy[0].cpu().numpy())
            return [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
    return None

############################################
# 5. KALMAN FILTERS FOR TRACKING
############################################
class Kalman2D:
    """
    A simple 2D Kalman Filter (state: [x, y, vx, vy]) using a constant velocity model.
    """
    def __init__(self, dt=1/TARGET_FPS, process_noise=5.0, measurement_noise=10.0):
        self.dt = dt
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.P = np.eye(4, dtype=np.float32) * 500

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(2, 1)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.state

class TrackKalman2D:
    """
    Wraps Kalman2D to store the object's path.
    """
    def __init__(self):
        self.kf = Kalman2D()
        self.path = []  # list of (x, y) for each frame

    def predict(self):
        self.kf.predict()
        pred_x = self.kf.state[0, 0]
        pred_y = self.kf.state[1, 0]
        return (int(pred_x), int(pred_y))

    def update(self, xy):
        self.kf.update(xy)
        x, y, vx, vy = self.kf.state.flatten()
        self.path.append((int(x), int(y)))

    def get_latest_position(self):
        x, y, vx, vy = self.kf.state.flatten()
        return (int(x), int(y))

############################################
# 6. WIND AND PROJECTILE UTILS
############################################
def degrees_to_radians(deg):
    return deg * math.pi / 180.0

def get_wind_components(wind_speed_m_s, wind_dir_deg):
    """
    Convert wind speed and direction to (w_x, w_y) components.
    Direction 0 means wind blowing in the +X direction (tailwind).
    """
    theta = degrees_to_radians(wind_dir_deg)
    w_x = wind_speed_m_s * math.cos(theta)
    w_y = wind_speed_m_s * math.sin(theta)
    return (w_x, w_y)

def compute_launch_velocity(ball_positions, launch_index, fps, pixels_per_meter):
    """
    Compute the ball's launch speed and angle (v0, launch_angle_deg) using the positions
    at indices launch_index-1 and launch_index.
    """
    if launch_index < 1 or launch_index >= len(ball_positions):
        return None

    (x0_px, y0_px) = ball_positions[launch_index - 1]
    (x1_px, y1_px) = ball_positions[launch_index]

    dt = 1.0 / fps

    # Convert from pixels to meters
    x0_m = x0_px / pixels_per_meter
    y0_m = y0_px / pixels_per_meter
    x1_m = x1_px / pixels_per_meter
    y1_m = y1_px / pixels_per_meter

    vx = (x1_m - x0_m) / dt
    vy = (y0_m - y1_m) / dt  # y0 - y1: positive vy is upward

    v0 = math.sqrt(vx**2 + vy**2)
    angle_deg = math.degrees(math.atan2(vy, vx)) if vx != 0 else (90.0 if vy>0 else -90.0)
    return (v0, angle_deg)

def compute_carry_distance_with_wind(v0, launch_angle_deg, wind_speed_m_s, wind_dir_deg, launch_height=0.0):
    """
    Compute the carry distance given launch speed, angle, and wind.
    The effective horizontal velocity is adjusted by wind.
    """
    angle_rad = math.radians(launch_angle_deg)
    vx_ball = v0 * math.cos(angle_rad)
    vy_ball = v0 * math.sin(angle_rad)

    w_x, w_y = get_wind_components(wind_speed_m_s, wind_dir_deg)
    vx_eff = vx_ball + w_x
    vy_eff = vy_ball  # ignoring vertical wind for simplicity

    a = -0.5 * GRAVITY
    b = vy_eff
    c = launch_height
    disc = b**2 - 4*a*c
    if disc < 0:
        return 0.0
    t1 = (-b + math.sqrt(disc)) / (2*a)
    t2 = (-b - math.sqrt(disc)) / (2*a)
    T = max(t1, t2) if max(t1, t2) > 0 else 0.0

    carry = vx_eff * T
    return carry

############################################
# 7. LAUNCH DETECTION
############################################
def detect_launch_index(ball_positions, fps, pixels_per_meter, speed_threshold=3.0):
    """
    Returns the first index where the speed computed from consecutive positions exceeds speed_threshold (m/s).
    """
    for i in range(1, len(ball_positions)):
        x0, y0 = ball_positions[i-1]
        x1, y1 = ball_positions[i]
        dt = 1.0 / fps
        dx_m = (x1 - x0) / pixels_per_meter
        dy_m = (y0 - y1) / pixels_per_meter
        speed = math.sqrt(dx_m**2 + dy_m**2) / dt
        if speed > speed_threshold:
            return i
    return None

############################################
# 8. FRAME PROCESSING
############################################
def process_frame(frame, frame_idx, ball_tracker, club_tracker, in_flight):
    """
    Process one frame:
      1. Advanced preprocess.
      2. Full-frame detection.
      3. If detection fails, use a predictive ROI based on Kalman prediction.
      4. Update Kalman trackers.
      5. Annotate the frame with bounding boxes and tracked paths.
    Returns the annotated frame.
    """
    # Step 1: Advanced preprocessing
    frame_pp = advanced_preprocess(frame)
    frame_resized = cv2.resize(frame_pp, (640, 640))

    # Step 2: Full-frame detection
    ball_bbox, club_bbox = detect_objects(frame_resized)

    # Step 3: If ball detection fails, use predictive ROI based on Kalman prediction
    if ball_bbox is None:
        pred_ball = ball_tracker.predict()  # predicted position (x, y)
        ball_bbox = detect_object_in_roi(frame_resized, pred_ball, ROI_MARGIN, BALL_CLASS_ID)

    # For the club, to speed things up, we skip ROI detection and rely on Kalman prediction
    if club_bbox is None:
        club_bbox = None  # Simply do not attempt extra detection for the club

    # Step 4: Compute centers
    ball_center = None
    if ball_bbox is not None:
        x1, y1, x2, y2 = ball_bbox
        ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    club_center = None
    if club_bbox is not None:
        x1, y1, x2, y2 = club_bbox
        club_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Step 5: Update trackers
    if ball_center is not None:
        ball_tracker.update(ball_center)
    else:
        ball_tracker.predict()

    if club_center is not None:
        club_tracker.update(club_center)
    else:
        club_tracker.predict()

    # Step 6: Annotation
    annotated = frame_resized.copy()
    if ball_bbox is not None:
        bx1, by1, bx2, by2 = map(int, ball_bbox)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(annotated, "ball", (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if club_bbox is not None:
        cx1, cy1, cx2, cy2 = map(int, club_bbox)
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.putText(annotated, "club", (cx1, cy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw ball tracking path
    for i in range(1, len(ball_tracker.path)):
        pt1 = ball_tracker.path[i - 1]
        pt2 = ball_tracker.path[i]
        cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)
    # Draw club tracking path
    for i in range(1, len(club_tracker.path)):
        pt1 = club_tracker.path[i - 1]
        pt2 = club_tracker.path[i]
        cv2.line(annotated, pt1, pt2, (255, 0, 255), 2)

    return annotated

############################################
# 9. MAIN PIPELINE
############################################
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_PATH}")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = TARGET_FPS

    out_width, out_height = 640, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("processed_golf_video.mp4", fourcc, input_fps, (out_width, out_height))

    ball_tracker = TrackKalman2D()
    club_tracker = TrackKalman2D()

    frame_idx = 0
    start_time = time.time()

    print("[INFO] Processing video. Press 'q' to exit early...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame, frame_idx, ball_tracker, club_tracker, in_flight=False)
        out.write(annotated_frame)
        cv2.imshow("Processed Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"[INFO] Processed {frame_idx} frames in {total_time:.2f} seconds.")

    # After processing, detect the launch index from the tracked ball positions
    launch_index = detect_launch_index(ball_tracker.path, input_fps, PIXELS_PER_METER, speed_threshold=3.0)
    if launch_index is None:
        print("[WARN] No 'launch' event found based on speed threshold.")
        return

    # Compute initial velocity at launch
    v0_angle = compute_launch_velocity(ball_tracker.path, launch_index, input_fps, PIXELS_PER_METER)
    if v0_angle is None:
        print("[WARN] Could not compute initial velocity at launch index.")
        return

    v0, launch_angle_deg = v0_angle

    # Compute carry distance with wind
    carry_dist = compute_carry_distance_with_wind(
        v0, launch_angle_deg,
        wind_speed_m_s, wind_dir_deg,
        launch_height=LAUNCH_HEIGHT
    )

    print("==============================================")
    print(f"[RESULT] Launch detected at frame {launch_index}.")
    print(f"  -> Launch speed:   {v0:.2f} m/s")
    print(f"  -> Launch angle:  {launch_angle_deg:.2f} deg")
    print(f"  -> Carry distance: {carry_dist:.2f} m (with wind)")
    print("==============================================")

if __name__ == "__main__":
    main()
