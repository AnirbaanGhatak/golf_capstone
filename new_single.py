import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
import time

###############################################################################
# CONFIG
###############################################################################
VIDEO_PATH = "Test_videos/TV_main.mp4"
YOLO_MODEL_PATH = "PTs/best_tuning.pt"

# Classes:
BALL_CLASS_ID = 0
CLUB_CLASS_ID = 2

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3

TARGET_FPS = 60
GRAVITY = 9.81
LAUNCH_SPEED_THRESHOLD = 3.0  # m/s => "launch" detection
PIXELS_PER_METER = 50.0

# ROI logic
ROI_MARGIN_INITIAL = 150
ROI_MARGIN_GROWTH = 30
ROI_MARGIN_MAX = 500

# We'll skip "launch" detection if speed > MAX_PHYSICAL_SPEED
MAX_PHYSICAL_SPEED = 100.0

# Only flip the ball's X coordinate, not the club
FLIP_BALL_X_COORDS = True

# If angle<0 => clamp to positive
CLAMP_ANGLE = True

# Real-time slow playback factor (bigger = slower display)
SLOW_DISPLAY_WAIT_MS = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# YOLO LOAD
###############################################################################
model = YOLO(YOLO_MODEL_PATH)
model.fuse()
model.to(device)

###############################################################################
# ADVANCED PREPROCESS
###############################################################################
def advanced_preprocess(frame):
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

    hsv = cv2.cvtColor(frame_sharp, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    tophat = cv2.morphologyEx(hsv, cv2.MORPH_TOPHAT, kernel)
    hsv_combined = cv2.addWeighted(hsv, 1.0, tophat, 0.5, 0)
    processed = cv2.cvtColor(hsv_combined, cv2.COLOR_HSV2BGR)
    return processed

###############################################################################
# DETECTION
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
    if roi.shape[0]==0 or roi.shape[1]==0:
        return None
    results = model(roi, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        if cls_id == desired_class:
            bx1, by1, bx2, by2 = map(int, det.xyxy[0].cpu().numpy())
            return [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
    return None

###############################################################################
# BALL TRACKER
###############################################################################
class BallTracker:
    def __init__(self):
        self.positions = []  # store (x, y)
        self.missed_frames = 0
        self.dynamic_margin = ROI_MARGIN_INITIAL

    def push_position(self, x, y):
        self.positions.append((x, y))

    def get_latest_position(self):
        if len(self.positions)>0:
            return self.positions[-1]
        return (320, 320)  # fallback

    def update_missed(self, found):
        if found:
            self.missed_frames = 0
            self.dynamic_margin = ROI_MARGIN_INITIAL
        else:
            self.missed_frames += 1
            self.dynamic_margin = min(
                ROI_MARGIN_INITIAL + ROI_MARGIN_GROWTH*self.missed_frames,
                ROI_MARGIN_MAX
            )

###############################################################################
# CLUB TRACKER
###############################################################################
class ClubTracker:
    def __init__(self):
        self.path = []
    def update(self, x, y):
        self.path.append((x, y))

###############################################################################
# DETECT LAUNCH
###############################################################################
def detect_launch_index(positions, fps, threshold=LAUNCH_SPEED_THRESHOLD):
    for i in range(1, len(positions)):
        (x0, y0) = positions[i-1]
        (x1, y1) = positions[i]
        dt = 1.0/fps
        dx_m = (x1 - x0)/PIXELS_PER_METER
        dy_m = (y0 - y1)/PIXELS_PER_METER
        speed = math.sqrt(dx_m**2 + dy_m**2)/dt
        if speed>MAX_PHYSICAL_SPEED:
            continue
        if speed>threshold:
            return i
    return None

###############################################################################
# WIND & CARRY
###############################################################################
def degrees_to_radians(deg):
    return deg*math.pi/180.0

def get_wind_components(wind_speed_m_s, wind_dir_deg):
    theta = degrees_to_radians(wind_dir_deg)
    w_x = wind_speed_m_s*math.cos(theta)
    w_y = wind_speed_m_s*math.sin(theta)
    return (w_x, w_y)

def compute_carry_distance_with_wind(v0, angle_deg, wind_speed, wind_dir, launch_height=0.0):
    if CLAMP_ANGLE and angle_deg<0:
        angle_deg = abs(angle_deg)

    angle_rad = math.radians(angle_deg)
    vx_ball = v0*math.cos(angle_rad)
    vy_ball = v0*math.sin(angle_rad)
    w_x, w_y = get_wind_components(wind_speed, wind_dir)
    vx_eff = abs(vx_ball + w_x)
    vy_eff = vy_ball
    a = -0.5*GRAVITY
    b = vy_eff
    c = launch_height
    disc = b**2 - 4*a*c
    if disc<0:
        return 0.0
    t1 = (-b + math.sqrt(disc)) / (2*a)
    t2 = (-b - math.sqrt(disc)) / (2*a)
    T = max(t1, t2) if max(t1, t2)>0 else 0
    return vx_eff*T

###############################################################################
# PROCESS FRAME
###############################################################################
def process_frame(frame, frame_idx, 
                  club_tracker, ball_tracker,
                  in_flight, fps):
    frame_pp = advanced_preprocess(frame)
    h, w, _ = frame_pp.shape
    frame_resized = cv2.resize(frame_pp, (640, 640))

    annotated = frame_resized.copy()

    # 1) Full-frame detection for club
    club_bbox = None
    results = model(frame_resized, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        if conf<CONF_THRESHOLD:
            continue
        if cls_id==CLUB_CLASS_ID:
            club_bbox = [x1, y1, x2, y2]
            break

    if club_bbox is not None:
        bx1, by1, bx2, by2 = club_bbox
        cx = (bx1+bx2)//2
        cy = (by1+by2)//2
        # do NOT flip coords for the club
        club_tracker.update(cx, cy)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0,0,255), 2)
        cv2.putText(annotated, "club", (bx1, by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Draw club path
    for i in range(1, len(club_tracker.path)):
        pt1 = club_tracker.path[i-1]
        pt2 = club_tracker.path[i]
        cv2.line(annotated, pt1, pt2, (255,0,255), 2)

    # 2) Ball logic
    ball_bbox = None

    if not in_flight:
        ball_found_full = False
        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            if conf<CONF_THRESHOLD:
                continue
            if cls_id==BALL_CLASS_ID:
                ball_bbox = [x1, y1, x2, y2]
                ball_found_full = True
                break
        if not ball_found_full:
            last_ball = ball_tracker.get_latest_position()
            margin = ball_tracker.dynamic_margin
            ball_bbox = detect_object_in_roi(
                frame_resized, last_ball, margin, BALL_CLASS_ID
            )
    else:
        pred_ball = ball_tracker.get_latest_position()
        ball_bbox = detect_object_in_roi(
            frame_resized, pred_ball, ball_tracker.dynamic_margin, BALL_CLASS_ID
        )

    if ball_bbox is not None:
        bx1, by1, bx2, by2 = ball_bbox
        cx = (bx1+bx2)//2
        cy = (by1+by2)//2
        if FLIP_BALL_X_COORDS:
            cx = 640 - cx
        ball_tracker.push_position(cx, cy)
        ball_tracker.update_missed(True)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0,255,0), 2)
        cv2.putText(annotated, "ball", (bx1, by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        ball_tracker.update_missed(False)

    # draw ball path
    for i in range(1, len(ball_tracker.positions)):
        pt1 = ball_tracker.positions[i-1]
        pt2 = ball_tracker.positions[i]
        cv2.line(annotated, pt1, pt2, (0,255,0), 2)

    return annotated

###############################################################################
# MAIN
###############################################################################
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open video:", VIDEO_PATH)
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps<=0:
        input_fps = TARGET_FPS

    out_width, out_height = 640, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("processed_golf_video.mp4", fourcc, input_fps, (out_width, out_height))

    club_tracker = ClubTracker()
    ball_tracker = BallTracker()

    in_flight = False
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(
            frame, frame_idx,
            club_tracker, ball_tracker,
            in_flight,
            input_fps
        )

        # check for launch
        if not in_flight and len(ball_tracker.positions)>1:
            i_launch = detect_launch_index(ball_tracker.positions, input_fps, LAUNCH_SPEED_THRESHOLD)
            if i_launch is not None:
                print(f"[INFO] Launch detected near frame {frame_idx}.")
                in_flight = True

        out.write(annotated_frame)
        cv2.imshow("ProcessedFrame", annotated_frame)

        # wait longer for slower real-time display
        if cv2.waitKey(SLOW_DISPLAY_WAIT_MS) & 0xFF==ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time()-start_time
    print(f"[INFO] Processed {frame_idx} frames in {total_time:.2f} seconds.")

    # compute carry distance
    if len(ball_tracker.positions)<2:
        print("[INFO] Not enough ball data => no carry distance.")
        return

    i_launch = detect_launch_index(ball_tracker.positions, input_fps, LAUNCH_SPEED_THRESHOLD)
    if i_launch is None or i_launch<1 or i_launch>=len(ball_tracker.positions):
        print("[INFO] No valid launch => no carry distance.")
        return

    (x0, y0) = ball_tracker.positions[i_launch-1]
    (x1, y1) = ball_tracker.positions[i_launch]
    dt = 1.0/input_fps
    dx_m = (x1 - x0)/PIXELS_PER_METER
    dy_m = (y0 - y1)/PIXELS_PER_METER
    speed = math.sqrt(dx_m**2 + dy_m**2)/dt
    angle_deg = math.degrees(math.atan2(dy_m, dx_m)) if dx_m!=0 else 90.0

    if CLAMP_ANGLE and angle_deg<0:
        angle_deg = abs(angle_deg)

    # compute carry
    def degrees_to_radians(d):
        return d*math.pi/180.0
    def get_wind_components(ws, wd):
        th = degrees_to_radians(wd)
        return (ws*math.cos(th), ws*math.sin(th))
    def compute_carry_distance(v0, ang_deg, w_speed, w_dir, lh=0.0):
        a_rad = math.radians(ang_deg)
        vx_ball = v0*math.cos(a_rad)
        vy_ball = v0*math.sin(a_rad)
        (wx, wy) = get_wind_components(w_speed, w_dir)
        vx_eff = abs(vx_ball + wx)
        vy_eff = vy_ball
        A = -0.5*GRAVITY
        B = vy_eff
        C = lh
        disc = B**2 - 4*A*C
        if disc<0:
            return 0.0
        t1 = (-B + math.sqrt(disc))/(2*A)
        t2 = (-B - math.sqrt(disc))/(2*A)
        T = max(t1, t2) if max(t1, t2)>0 else 0
        return vx_eff*T

    carry = compute_carry_distance(speed, angle_deg, 2.0, 0.0, 0.0)
    print("=================================================")
    print(f"[RESULT] Launch Speed: {speed:.2f} m/s  Angle: {angle_deg:.2f} deg")
    print(f"Carry distance (with wind): {carry:.2f} m")
    print("=================================================")

if __name__=="__main__":
    main()
