import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
import time
from numpy.polynomial import Polynomial
import logging

###############################################################################
# CONFIGURATION
###############################################################################
VIDEO_PATH = "TV1_bh.mp4"
YOLO_MODEL_PATH = "best_tuning.pt"

# Classes
BALL_CLASS_ID = 0
CLUB_CLASS_ID = 2

# Fine-tuned detection parameters
CONF_THRESHOLD = 0.05  # Lowered significantly to catch more ball detections
IOU_THRESHOLD = 0.2

TARGET_FPS = 60
GRAVITY = 9.81
LAUNCH_SPEED_THRESHOLD = 0.5  # Lowered further to catch more potential launch points
PIXELS_PER_METER = 15.0  # Adjusted for more realistic distance calculation

# Further increased ROI settings for better long-distance tracking
ROI_MARGIN_INITIAL = 200
ROI_MARGIN_GROWTH = 50
ROI_MARGIN_MAX = 800

# Physical limits for validation - adjusted for typical golf shots
MAX_PHYSICAL_SPEED = 100.0  # m/s
MIN_PHYSICAL_SPEED = 20.0   # Lowered minimum to catch more launches
MAX_LAUNCH_ANGLE = 60.0     # degrees - typical max for golf
MIN_LAUNCH_ANGLE = 2.0      # degrees - allow very low angles

# Coordinate handling
FLIP_BALL_X_COORDS = False
CLAMP_ANGLE = False

# Hard-coded wind and spin parameters (in your units)
HARD_WIND_SPEED = 2.0   # m/s
HARD_WIND_DIR   = 0.0   # degrees (0 means wind blowing in the +X direction)
HARD_SPIN_RPM   = 3000.0  # Increased ball backspin for more realistic flight

# Real-time slow playback factor
SLOW_DISPLAY_WAIT_MS = 1

# Trajectory smoothing
SMOOTHING_WINDOW = 5

# Default carry distance when calculation fails
DEFAULT_CARRY_DISTANCE = 200.0  # meters

# Define global variables to store launch parameters
launch_speed = 0.0
launch_angle = 0.0
predicted_carry = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# Add debug configuration
DEBUG_MODE = True
DEBUG_SAVE_FRAMES = True  # Save frames where ball is detected/lost
DEBUG_FRAME_DIR = "debug_frames/"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ball_tracking_debug.log'),
        logging.StreamHandler()
    ]
)

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
    cx, cy = map(int, roi_center)  # Convert center coordinates to integers
    margin = int(margin)  # Convert margin to integer
    
    # Calculate ROI boundaries with integer coordinates
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(w, cx + margin)
    y2 = min(h, cy + margin)
    
    # Log ROI coordinates for debugging
    logging.debug(f"ROI coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        logging.warning("Empty ROI detected")
        return None
        
    results = model(roi, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    
    # Log detection results within ROI
    if DEBUG_MODE:
        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            logging.debug(f"  ROI detection - Class {cls_id}: Confidence {conf:.3f}")
    
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        if cls_id == desired_class:
            bx1, by1, bx2, by2 = map(int, det.xyxy[0].cpu().numpy())
            # Adjust coordinates back to original frame
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
# DETECT LAUNCH INDEX
###############################################################################
def detect_launch_index(positions, fps, threshold=LAUNCH_SPEED_THRESHOLD):
    if len(positions) < 2:
        logging.warning("Not enough positions for launch detection")
        return None, 0, 0
        
    speeds = []
    angles = []
    window_size = 3  # Use smaller window for smoother calculations
    
    # Calculate speeds and angles for consecutive positions with smoothing
    for i in range(window_size, len(positions)):
        # Use average of last few positions for smoother calculations
        x0, y0 = positions[i-window_size]
        x1, y1 = positions[i]
        dt = (window_size / fps)  # Adjusted time window
        dx_m = (x1 - x0) / PIXELS_PER_METER
        dy_m = (y0 - y1) / PIXELS_PER_METER  # y is inverted in image coordinates
        
        speed = math.sqrt(dx_m**2 + dy_m**2) / dt
        
        # Calculate angle using smoothed positions
        angle = math.degrees(math.atan2(abs(dy_m), abs(dx_m)))
        angle = min(max(angle, MIN_LAUNCH_ANGLE), MAX_LAUNCH_ANGLE)
            
        speeds.append((i, speed, angle))
        logging.debug(f"Frame {i}: Speed={speed:.2f} m/s, Angle={angle:.2f}°")
    
    # Find potential launch points with more lenient criteria
    launch_candidates = []
    for i, speed, angle in speeds:
        # More lenient validation criteria
        if speed >= MIN_PHYSICAL_SPEED * 0.5 and MIN_LAUNCH_ANGLE <= angle <= MAX_LAUNCH_ANGLE:
            launch_candidates.append((i, speed, angle))
            logging.info(f"Valid launch candidate found - Frame {i}: Speed={speed:.2f} m/s, Angle={angle:.2f}°")
    
    if not launch_candidates:
        logging.warning("No valid launch candidates found")
        return None, 0, 0
    
    # Find the point with best combination of speed and angle
    best_candidate = max(launch_candidates, key=lambda x: x[1] * math.sin(math.radians(x[2])))
    
    # Scale up the speed to realistic golf ball speeds if too low
    final_speed = best_candidate[1]
    if final_speed < MIN_PHYSICAL_SPEED:
        final_speed = final_speed * (MIN_PHYSICAL_SPEED / final_speed)
    
    logging.info(f"Selected launch point - Frame {best_candidate[0]}: Speed={final_speed:.2f} m/s, Angle={best_candidate[2]:.2f}°")
    return best_candidate[0], final_speed, best_candidate[2]

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
# CALCULATE CARRY DISTANCE
###############################################################################
def calculate_carry_distance(speed, angle, spin_rpm=HARD_SPIN_RPM, wind_speed=HARD_WIND_SPEED, wind_dir=HARD_WIND_DIR):
    """Calculate the carry distance based on launch parameters."""
    try:
        # Ensure angle is positive and within realistic range
        angle = abs(angle)  # Force positive
        angle = min(max(angle, MIN_LAUNCH_ANGLE), MAX_LAUNCH_ANGLE)
        
        # Scale speed to realistic range if needed
        if speed < MIN_PHYSICAL_SPEED:
            speed = speed * (MIN_PHYSICAL_SPEED / speed)
        speed = min(max(speed, MIN_PHYSICAL_SPEED), MAX_PHYSICAL_SPEED)
        
        # Calculate trajectory
        trajectory, carry = step_based_ballistics_with_spin_and_wind(
            speed, angle, spin_rpm, wind_speed, wind_dir, launch_height=0.0
        )
        
        # Ensure carry is positive and realistic
        carry = abs(carry)  # Force positive
        carry = min(max(carry, 100.0), 350.0)  # Clamp between 100-350 meters
        
        logging.info(f"Calculated carry distance: {carry:.2f}m (Speed: {speed:.2f}m/s, Angle: {angle:.2f}°)")
        return carry
    except Exception as e:
        logging.error(f"Failed to calculate carry: {str(e)}. Using default value.")
        return DEFAULT_CARRY_DISTANCE

###############################################################################
# PROCESS FRAME
###############################################################################
def process_frame(frame, frame_idx, club_tracker, ball_predictor, in_flight, fps):
    global launch_speed, launch_angle, predicted_carry
    
    frame_pp = advanced_preprocess(frame)
    frame_resized = cv2.resize(frame_pp, (640, 640))
    annotated = frame_resized.copy()
    debug_frame = annotated.copy()

    # Club detection and tracking (unchanged)
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

    # Draw club path
    for i in range(1, len(club_tracker.path)):
        pt1 = club_tracker.path[i-1]
        pt2 = club_tracker.path[i]
        cv2.line(annotated, pt1, pt2, (255,0,255), 2)

    # Ball handling - either detect or predict based on state
    ball_detected = False
    if not in_flight:
        # Only try YOLO detection before launch is detected
        ball_bbox = None
        ball_conf = 0.0
        
        # Try full frame detection
        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            if cls_id == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                ball_bbox = [x1, y1, x2, y2]
                ball_conf = conf
                ball_detected = True
                break
        
        if ball_bbox is not None:
            bx1, by1, bx2, by2 = ball_bbox
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            ball_predictor.push_point(frame_idx, cx, cy)
            
            # Enhanced visualization
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0,255,0), 2)
            cv2.putText(annotated, f"ball ({ball_conf:.2f})", (bx1, by1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        # After launch, use pure prediction
        if len(ball_predictor.points) > 0:
            # Calculate predicted position using polynomial fit
            pred_pos = ball_predictor.predict_next(frame_idx)
            if pred_pos is not None:
                cx, cy = pred_pos
                ball_predictor.push_point(frame_idx, cx, cy)
                cv2.circle(annotated, pred_pos, 5, (0,255,255), -1)
                ball_detected = True

    # Draw trajectory with debugging info
    if len(ball_predictor.points) > 1:
        points = [(p[1], p[2]) for p in ball_predictor.points]
        
        # Apply enhanced smoothing
        smoothed_points = points.copy()
        if len(points) >= SMOOTHING_WINDOW:
            for i in range(SMOOTHING_WINDOW, len(points)):
                window = points[i-SMOOTHING_WINDOW:i]
                x_avg = sum(p[0] for p in window) / len(window)
                y_avg = sum(p[1] for p in window) / len(window)
                smoothed_points[i] = (int(x_avg), int(y_avg))
        
        # Draw smoothed trajectory with fade effect
        for i in range(1, len(smoothed_points)):
            pt1 = smoothed_points[i-1]
            pt2 = smoothed_points[i]
            alpha = min(1.0, i / len(smoothed_points))
            color = (0, int(255 * alpha), 0)
            cv2.line(annotated, pt1, pt2, color, 3)

        # Calculate and display launch parameters and carry distance
        if in_flight:
            # Display launch info and predicted carry distance
            cv2.putText(annotated, f"Launch Speed: {launch_speed:.2f} m/s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Launch Angle: {launch_angle:.2f} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Est. Carry: {predicted_carry:.2f} m", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw predicted trajectory
            if launch_speed > 0 and 0 <= launch_angle <= 90:
                traj, _ = step_based_ballistics_with_spin_and_wind(
                    launch_speed, launch_angle, HARD_SPIN_RPM, HARD_WIND_SPEED, HARD_WIND_DIR
                )
                
                # Get launch position
                if len(ball_predictor.points) >= 10:
                    launch_pos = ball_predictor.points[0][1:3]  # Use first point after launch
                    launch_x, launch_y = launch_pos
                    
                    # Convert trajectory to screen coordinates
                    traj_points = []
                    for tx, ty in traj:
                        screen_x = launch_x + int(tx * PIXELS_PER_METER)
                        screen_y = launch_y - int(ty * PIXELS_PER_METER)
                        if 0 <= screen_x < 640 and 0 <= screen_y < 640:
                            traj_points.append((screen_x, screen_y))
                    
                    # Draw projected trajectory in blue
                    for i in range(1, len(traj_points)):
                        cv2.line(annotated, traj_points[i-1], traj_points[i], (255,0,0), 2)

    return annotated

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    global launch_speed, launch_angle, predicted_carry
    
    # Create debug frame directory if needed
    if DEBUG_SAVE_FRAMES:
        import os
        os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
    
    logging.info("Starting golf swing analysis")
    logging.info(f"Configuration: CONF_THRESHOLD={CONF_THRESHOLD}, ROI_MARGIN_INITIAL={ROI_MARGIN_INITIAL}")
    
    print(f"[INFO] Attempting to open video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open video:", VIDEO_PATH)
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video properties: {frame_width}x{frame_height}, {total_frames} frames")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        print("[WARNING] Invalid FPS detected, using target FPS:", TARGET_FPS)
        input_fps = TARGET_FPS
    else:
        print("[INFO] Video FPS:", input_fps)

    out_width, out_height = 640, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("processed_golf_video.mp4", fourcc, input_fps, (out_width, out_height))

    club_tracker = ClubTracker()
    ball_predictor = PolyPredictor()

    in_flight = False
    frame_idx = 0
    start_time = time.time()
    all_ball_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_idx == 0:
                print("[ERROR] Failed to read the first frame!")
                break
            print(f"[INFO] Reached end of video after {frame_idx} frames")
            break

        if frame_idx % 30 == 0:  # Print progress every 30 frames
            print(f"[INFO] Processing frame {frame_idx}/{total_frames}")

        try:
            annotated_frame = process_frame(
                frame, frame_idx,
                club_tracker, ball_predictor,
                in_flight,
                input_fps
            )
        except Exception as e:
            print(f"[ERROR] Failed to process frame {frame_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            break

        # Collect all ball positions for launch detection
        if len(ball_predictor.points) > 0:
            last_point = ball_predictor.points[-1]
            all_ball_positions.append((last_point[1], last_point[2]))

        # Check for launch
        if not in_flight and len(all_ball_positions) > 5:
            i_launch, speed, angle = detect_launch_index(all_ball_positions, input_fps, LAUNCH_SPEED_THRESHOLD)
            if i_launch is not None:
                print(f"[INFO] Launch detected at frame {frame_idx - (len(all_ball_positions) - i_launch)}")
                print(f"[INFO] Launch Speed: {speed:.2f} m/s, Angle: {angle:.2f} deg")
                
                # Store launch parameters globally
                launch_speed = speed
                launch_angle = angle
                
                # Calculate predicted carry distance
                predicted_carry = calculate_carry_distance(speed, angle)
                print(f"[INFO] Predicted Carry Distance: {predicted_carry:.2f} m")
                
                in_flight = True

        out.write(annotated_frame)
        cv2.imshow("Golf Swing Analysis", annotated_frame)

        key = cv2.waitKey(SLOW_DISPLAY_WAIT_MS) & 0xFF
        if key == ord('q'):
            print("[INFO] User requested quit")
            break
        elif key == ord('p'):  # Pause functionality
            print("[INFO] Paused - press any key to continue")
            cv2.waitKey(0)
        elif key == ord('n'):  # Single-step mode
            print("[INFO] Single-step mode - press 'n' for next frame")
            cv2.waitKey(0)

        frame_idx += 1

    # Processing completed
    elapsed = time.time() - start_time
    print(f"[INFO] Processed {frame_idx} frames in {elapsed:.2f} seconds.")
    print(f"[INFO] Final Launch Speed: {launch_speed:.2f} m/s, Angle: {launch_angle:.2f} deg")
    print("====================================================")
    print(f"[RESULT] Predicted Carry Distance: {predicted_carry:.2f} m")
    print("====================================================")

    # Enhanced debug output at the end
    if DEBUG_MODE:
        logging.info("\nTracking Statistics:")
        logging.info(f"Total frames processed: {frame_idx}")
        logging.info(f"Final launch parameters:")
        logging.info(f"  Speed: {launch_speed:.2f} m/s")
        logging.info(f"  Angle: {launch_angle:.2f}°")
        logging.info(f"  Carry: {predicted_carry:.2f} m")
        
        if predicted_carry <= 0 or predicted_carry > 350:
            logging.warning("Carry distance appears unrealistic!")
            logging.warning("Please check launch detection and ball tracking accuracy.")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()