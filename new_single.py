import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
import time
from numpy.polynomial import Polynomial

###############################################################################
# CONFIGURATION
###############################################################################
VIDEO_PATH = "Test_videos/TV1_bh.mp4"
YOLO_MODEL_PATH = "PTs/best_tuning.pt"

# Classes
BALL_CLASS_ID = 0
CLUB_CLASS_ID = 2

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.2

TARGET_FPS = 60
GRAVITY = 9.81
LAUNCH_SPEED_THRESHOLD = 3.0  # m/s, for detecting launch
PIXELS_PER_METER = 50.0

# ROI settings
ROI_MARGIN_INITIAL = 150
ROI_MARGIN_GROWTH = 30
ROI_MARGIN_MAX = 500

# If computed speed > this, skip launch detection (outlier)
MAX_PHYSICAL_SPEED = 100.0

# For coordinate handling
FLIP_BALL_X_COORDS = False  # Set to True if you need to mirror ball's x
CLAMP_ANGLE = True          # Clamp negative angles to positive

# Hard-coded wind and spin parameters (in your units)
HARD_WIND_SPEED = 7.0   # m/s
HARD_WIND_DIR   = 2.0   # degrees (0 means wind blowing in the +X direction)
HARD_SPIN_RPM   = 71.0  # ball backspin in RPM

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# YOLO MODEL LOADING
###############################################################################
model = YOLO(YOLO_MODEL_PATH)
model.fuse()
model.to(device)

###############################################################################
# ADVANCED PREPROCESSING FUNCTION
###############################################################################
def advanced_preprocess(frame):
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
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

    hsv = cv2.cvtColor(frame_sharp, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    tophat = cv2.morphologyEx(hsv, cv2.MORPH_TOPHAT, kernel)
    hsv_combined = cv2.addWeighted(hsv, 1.0, tophat, 0.5, 0)
    processed = cv2.cvtColor(hsv_combined, cv2.COLOR_HSV2BGR)
    return processed

###############################################################################
# DETECTION FUNCTIONS
###############################################################################
def detect_objects(frame):
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

def detect_object_in_roi(frame, roi_center, margin, desired_class):
    h, w, _ = frame.shape
    cx, cy = roi_center
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(w, cx + margin)
    y2 = min(h, cy + margin)
    
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return None
    results = model(roi, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        if cls_id == desired_class:
            bx1, by1, bx2, by2 = map(int, det.xyxy[0].cpu().numpy())
            return [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
    return None

###############################################################################
# BALL TRACKER (For storing ball centers)
###############################################################################
class BallTracker:
    def __init__(self):
        self.positions = []  # list of (x, y)
        self.missed_frames = 0
        self.dynamic_margin = ROI_MARGIN_INITIAL

    def push_position(self, x, y):
        self.positions.append((x, y))

    def get_latest_position(self):
        if self.positions:
            return self.positions[-1]
        return None

    def update_missed(self, found):
        if found:
            self.missed_frames = 0
            self.dynamic_margin = ROI_MARGIN_INITIAL
        else:
            self.missed_frames += 1
            self.dynamic_margin = min(ROI_MARGIN_INITIAL + ROI_MARGIN_GROWTH * self.missed_frames,
                                      ROI_MARGIN_MAX)

###############################################################################
# CLUB TRACKER (For storing club positions)
###############################################################################
class ClubTracker:
    def __init__(self):
        self.path = []
    def update(self, x, y):
        self.path.append((x, y))

###############################################################################
# "JANSSON-LIKE" POLYNOMIAL PREDICTOR (Without LSTM)
###############################################################################
class PolyPredictor:
    def __init__(self, window_size=5, poly_order=2):
        self.window_size = window_size
        self.poly_order = poly_order
        self.points = []  # each is (frame_idx, x, y)
        self.dynamic_margin = ROI_MARGIN_INITIAL
        self.missed_frames = 0

    def push_point(self, frame_idx, x, y):
        self.points.append((frame_idx, x, y))
        if len(self.points) > self.window_size:
            self.points.pop(0)

    def predict_next(self, next_frame_idx):
        if len(self.points) == 0:
            # If no points, return None so we don't update with a default center
            return None
        if len(self.points) < 3:
            return self.points[-1][1], self.points[-1][2]
        frames = np.array([p[0] for p in self.points], dtype=np.float32)
        xs = np.array([p[1] for p in self.points], dtype=np.float32)
        ys = np.array([p[2] for p in self.points], dtype=np.float32)
        try:
            poly_x = Polynomial.fit(frames, xs, self.poly_order)
            poly_y = Polynomial.fit(frames, ys, self.poly_order)
            pred_x = poly_x(next_frame_idx)
            pred_y = poly_y(next_frame_idx)
            return int(pred_x), int(pred_y)
        except:
            return self.points[-1][1], self.points[-1][2]

    def update_missed(self, found):
        if found:
            self.missed_frames = 0
            self.dynamic_margin = ROI_MARGIN_INITIAL
        else:
            self.missed_frames += 1
            self.dynamic_margin = min(ROI_MARGIN_INITIAL + ROI_MARGIN_GROWTH * self.missed_frames,
                                      ROI_MARGIN_MAX)

###############################################################################
# DETECT LAUNCH INDEX (to mark the frame when the ball is launched)
###############################################################################
def detect_launch_index(positions, fps, threshold=LAUNCH_SPEED_THRESHOLD):
    # positions: list of (x, y)
    for i in range(1, len(positions)):
        x0, y0 = positions[i-1]
        x1, y1 = positions[i]
        dt = 1.0 / fps
        dx_m = (x1 - x0) / PIXELS_PER_METER
        dy_m = (y0 - y1) / PIXELS_PER_METER
        speed = math.sqrt(dx_m**2 + dy_m**2) / dt
        if speed > MAX_PHYSICAL_SPEED:
            continue
        if speed > threshold:
            return i
    return None

###############################################################################
# STEP-BASED BALLISTICS WITH SPIN & WIND (Trajectory Simulation)
###############################################################################
def step_based_ballistics_with_spin_and_wind(v0, angle_deg, spin_rpm, wind_speed_m_s, wind_dir_deg, launch_height=0.0):
    if CLAMP_ANGLE and angle_deg < 0:
        angle_deg = abs(angle_deg)
    radius = 0.02135  # m
    Cd = 0.2
    mass = 0.045
    area = math.pi * (radius**2)
    air_density = 1.225
    lift_mag_base = 0.285 * (1 - math.exp(-0.00026 * spin_rpm))
    def deg_to_rad(d):
        return d * math.pi / 180.0
    wind_dir_rad = deg_to_rad(wind_dir_deg)
    wind_vx = wind_speed_m_s * math.cos(wind_dir_rad)
    wind_vy = wind_speed_m_s * math.sin(wind_dir_rad)
    vx = v0 * math.cos(deg_to_rad(angle_deg))
    vy = v0 * math.sin(deg_to_rad(angle_deg))
    sx = 0.0
    sy = launch_height
    dt = 0.01
    traj = [(sx, sy)]
    while sy >= 0:
        rvx = vx - wind_vx
        rvy = vy - wind_vy
        vrel = math.sqrt(rvx**2 + rvy**2)
        F_drag = 0.5 * air_density * Cd * area * (vrel**2)
        if vrel > 0:
            drag_ax = -(F_drag / mass) * (rvx / vrel)
            drag_ay = -(F_drag / mass) * (rvy / vrel)
        else:
            drag_ax = 0
            drag_ay = 0
        if vrel > 0:
            lift_mag = lift_mag_base * vrel
            perp_x = -rvy
            perp_y = rvx
            perp_len = math.sqrt(perp_x**2 + perp_y**2)
            if perp_len > 0:
                perp_x /= perp_len
                perp_y /= perp_len
            lift_ax = (lift_mag / mass) * perp_x
            lift_ay = (lift_mag / mass) * perp_y
        else:
            lift_ax = 0
            lift_ay = 0
        ax = drag_ax + lift_ax
        ay = drag_ay + lift_ay - GRAVITY
        vx += ax * dt
        vy += ay * dt
        sx += vx * dt
        sy += vy * dt
        traj.append((sx, sy))
        if len(traj) > 200000:
            break
    if len(traj) >= 2 and traj[-2][1] > 0 and traj[-1][1] < 0:
        (x_prev, y_prev) = traj[-2]
        (x_last, y_last) = traj[-1]
        alpha = -y_prev / (y_last - y_prev)
        final_x = x_prev + alpha * (x_last - x_prev)
    else:
        final_x = traj[-1][0]
    return traj, final_x

###############################################################################
# PROCESS FRAME
###############################################################################
def process_frame(frame, frame_idx, club_tracker, ball_predictor, in_flight, fps):
    frame_pp = advanced_preprocess(frame)
    frame_resized = cv2.resize(frame_pp, (640, 640))
    annotated = frame_resized.copy()

    # 1) Club detection (full-frame)
    club_bbox = None
    results = model(frame_resized, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        if conf < CONF_THRESHOLD:
            continue
        if cls_id == CLUB_CLASS_ID:
            club_bbox = [x1, y1, x2, y2]
            break
    if club_bbox is not None:
        bx1, by1, bx2, by2 = club_bbox
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        club_tracker.update(cx, cy)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0,0,255), 2)
        cv2.putText(annotated, "club", (bx1, by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    for i in range(1, len(club_tracker.path)):
        pt1 = club_tracker.path[i-1]
        pt2 = club_tracker.path[i]
        cv2.line(annotated, pt1, pt2, (255,0,255), 2)

    # 2) Ball detection / prediction
    ball_bbox = None
    if not in_flight:
        ball_found_full = False
        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            if conf < CONF_THRESHOLD:
                continue
            if cls_id == BALL_CLASS_ID:
                ball_bbox = [x1, y1, x2, y2]
                ball_found_full = True
                break
        if not ball_found_full and len(ball_predictor.points) > 0:
            last_ball = ball_predictor.points[-1][1:3]
            margin = ball_predictor.dynamic_margin
            ball_bbox = detect_object_in_roi(frame_resized, last_ball, margin, BALL_CLASS_ID)
    else:
        pred_ball = ball_predictor.predict_next(frame_idx)
        if pred_ball is not None:
            ball_bbox = detect_object_in_roi(frame_resized, pred_ball, ball_predictor.dynamic_margin, BALL_CLASS_ID)

    if ball_bbox is not None:
        bx1, by1, bx2, by2 = ball_bbox
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        if FLIP_BALL_X_COORDS:
            cx = 640 - cx
        ball_predictor.push_point(frame_idx, cx, cy)
        ball_predictor.update_missed(True)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0,255,0), 2)
        cv2.putText(annotated, "ball", (bx1, by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        ball_predictor.update_missed(False)
        # Do not update if no detection – retain last valid position
        if len(ball_predictor.points) > 0:
            last_pt = ball_predictor.points[-1][1:3]
            cv2.circle(annotated, last_pt, 5, (0,255,0), -1)

    if len(ball_predictor.points) > 1:
        for i in range(1, len(ball_predictor.points)):
            _, x0, y0 = ball_predictor.points[i-1]
            _, x1, y1 = ball_predictor.points[i]
            cv2.line(annotated, (x0, y0), (x1, y1), (0,255,0), 2)

    # If in flight, overlay predicted trajectory (blue)
    if in_flight and len(ball_predictor.points) >= 3:
        # Use the launch speed and angle computed later from the launch event.
        # Here, we assume 'speed' and 'angle_deg' are global variables set at launch.
        traj, _ = step_based_ballistics_with_spin_and_wind(
            speed, angle_deg, HARD_SPIN_RPM, HARD_WIND_SPEED, HARD_WIND_DIR, launch_height=0.0
        )
        # Convert trajectory (in meters) to pixels
        traj_px = [(int(x * PIXELS_PER_METER), int(y * PIXELS_PER_METER)) for (x, y) in traj]
        for i in range(1, len(traj_px)):
            cv2.line(annotated, traj_px[i-1], traj_px[i], (255,0,0), 2)

    return annotated

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open video:", VIDEO_PATH)
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = TARGET_FPS

    out_width, out_height = 640, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("processed_golf_video.mp4", fourcc, input_fps, (out_width, out_height))

    club_tracker = ClubTracker()
    ball_predictor = PolyPredictor(window_size=5, poly_order=2)

    # Hard-coded wind and spin parameters
    wind_speed_m_s = HARD_WIND_SPEED  # 7.0 m/s
    wind_dir_deg = HARD_WIND_DIR        # 2.0 degrees
    spin_rpm = HARD_SPIN_RPM            # 71.0 RPM

    in_flight = False
    frame_idx = 0
    global speed, angle_deg
    speed, angle_deg = 0, 0  # will be set at launch

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame, frame_idx, club_tracker, ball_predictor, in_flight, input_fps)

        # Check for launch based on ball speed in predictor points
        if not in_flight and len(ball_predictor.points) > 1:
            pos_list = [(p[1], p[2]) for p in ball_predictor.points]
            i_launch = detect_launch_index(pos_list, input_fps, LAUNCH_SPEED_THRESHOLD)
            if i_launch is not None:
                print(f"[INFO] Launch detected near frame {frame_idx}.")
                in_flight = True
                # Compute launch speed and angle from last two valid points
                (x0, y0) = pos_list[i_launch - 1]
                (x1, y1) = pos_list[i_launch]
                dt = 1.0 / input_fps
                dx_m = (x1 - x0) / PIXELS_PER_METER
                dy_m = (y0 - y1) / PIXELS_PER_METER
                speed = math.sqrt(dx_m**2 + dy_m**2) / dt
                angle_deg = math.degrees(math.atan2(dy_m, dx_m)) if dx_m != 0 else 90.0
                if CLAMP_ANGLE and angle_deg < 0:
                    angle_deg = abs(angle_deg)
                print(f"[INFO] Launch Speed: {speed:.2f} m/s, Angle: {angle_deg:.2f} deg")

        out.write(annotated_frame)
        cv2.imshow("ProcessedFrame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"[INFO] Processed {frame_idx} frames in {total_time:.2f} seconds.")

    if len(ball_predictor.points) < 2:
        print("[INFO] Not enough ball data => no carry distance.")
        return

    pos_list = [(p[1], p[2]) for p in ball_predictor.points]
    i_launch = detect_launch_index(pos_list, input_fps, LAUNCH_SPEED_THRESHOLD)
    if i_launch is None or i_launch < 1 or i_launch >= len(pos_list):
        print("[INFO] No valid launch => no carry distance.")
        return

    (x0, y0) = pos_list[i_launch - 1]
    (x1, y1) = pos_list[i_launch]
    dt = 1.0 / input_fps
    dx_m = (x1 - x0) / PIXELS_PER_METER
    dy_m = (y0 - y1) / PIXELS_PER_METER
    speed = math.sqrt(dx_m**2 + dy_m**2) / dt
    angle_deg = math.degrees(math.atan2(dy_m, dx_m)) if dx_m != 0 else 90.0
    if CLAMP_ANGLE and angle_deg < 0:
        angle_deg = abs(angle_deg)
    print(f"[INFO] Final Launch Speed: {speed:.2f} m/s, Angle: {angle_deg:.2f} deg")

    traj, final_carry = step_based_ballistics_with_spin_and_wind(
        speed, angle_deg, spin_rpm, wind_speed_m_s, wind_dir_deg, launch_height=0.0
    )

    print("====================================================")
    print(f"[RESULT] Predicted Carry Distance (with spin & wind): {final_carry:.2f} m")
    print("====================================================")

if __name__ == "__main__":
    main()