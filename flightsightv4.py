
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import math
# Load model
model = YOLO("PTs/best_tunning.pt")

# Constants
BALL_CLASS_ID = 0
CLUB_CLASS_ID = 2
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.2
SLOW_MOTION_FACTOR = 16
DISPLAY_FRAMES = True
FRONT_VIDEO_PATH = "Test_videos/archive/TV1_bh.mp4"
SIDE_VIDEO_PATH = "Test_videos/archive/TV1_bs.mp4"

# Camera matrices and calibration offset
CAMERA_MATRIX_FRONT = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
CAMERA_MATRIX_SIDE = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
CAMERA_OFFSET = np.array([5.0, 0.0, 0.0])
PIXELS_PER_METER = 300.0


def center_of_bbox(bbox):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def detect_objects(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    ball_bbox = None
    club_bbox = None
    if hasattr(results[0], 'boxes'):
        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            if cls_id == BALL_CLASS_ID:
                ball_bbox = [x1, y1, x2, y2]
            elif cls_id == CLUB_CLASS_ID:
                club_bbox = [x1, y1, x2, y2]
    return ball_bbox, 1.0, club_bbox, 1.0


class KalmanTracker:
    def __init__(self):
        self.filtered_path = []
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                                 [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def update(self, point):
        if point is None:
            return
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        filtered_point = (int(prediction[0]), int(prediction[1]))
        self.filtered_path.append(filtered_point)
        if len(self.filtered_path) > 100:
            self.filtered_path.pop(0)


class BallTracker3D:
    def __init__(self, calibration):
        self.positions_3d = []
        self.calib = calibration

    def push_position(self, frame_idx, front_pos, side_pos, front_conf=1.0, side_conf=1.0):
        if front_pos and side_pos:
            pt_3d = self.calib.triangulate_point(front_pos, side_pos)
            if self.calib.validate_3d_point(pt_3d):
                self.positions_3d.append(pt_3d)

    def get_3d_positions(self):
        return self.positions_3d


class GolfCameraCalibration:
    def __init__(self, camera_matrix_front, camera_matrix_side, camera_offset):
        self.camera_matrix_front = camera_matrix_front
        self.camera_matrix_side = camera_matrix_side
        self.camera_offset = camera_offset
        self.R_front = np.eye(3)
        self.R_side = cv2.Rodrigues(np.array([0, -np.pi / 2, 0]))[0]
        self.t_front = np.zeros(3)
        self.t_side = camera_offset
        self.P_front = self.camera_matrix_front @ np.hstack((self.R_front, self.t_front.reshape(3, 1)))
        self.P_side = self.camera_matrix_side @ np.hstack((self.R_side, self.t_side.reshape(3, 1)))

    def triangulate_point(self, pt1, pt2):
        pt1_hom = np.array([pt1[0], pt1[1], 1.0])
        pt2_hom = np.array([pt2[0], pt2[1], 1.0])
        A = np.zeros((4, 4))
        A[0] = pt1_hom[0] * self.P_front[2] - self.P_front[0]
        A[1] = pt1_hom[1] * self.P_front[2] - self.P_front[1]
        A[2] = pt2_hom[0] * self.P_side[2] - self.P_side[0]
        A[3] = pt2_hom[1] * self.P_side[2] - self.P_side[1]
        _, _, vh = np.linalg.svd(A)
        pt_3d = vh[-1]
        return pt_3d[:3] / pt_3d[3]

    def validate_3d_point(self, pt, max_distance=20.0):
        return np.linalg.norm(pt) <= max_distance

    def project_3d_to_front(self, pt3d):
        pt3d_hom = np.append(pt3d, 1.0)
        proj = self.P_front @ pt3d_hom
        return proj[:2] / proj[2]

    def project_3d_to_side(self, pt3d):
        pt3d_hom = np.append(pt3d, 1.0)
        proj = self.P_side @ pt3d_hom
        return proj[:2] / proj[2]


def calculate_club_speed_2d(tracker, fps, pixels_per_meter):
    if len(tracker.filtered_path) < 2:
        return 0.0
    (x1, y1) = tracker.filtered_path[-2]
    (x2, y2) = tracker.filtered_path[-1]
    distance_pixels = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    distance_meters = distance_pixels / pixels_per_meter
    time_seconds = 1.0 / (fps * SLOW_MOTION_FACTOR)
    return distance_meters / time_seconds

def estimate_ball_speed(club_speed, smash_factor=1.48):
    return club_speed * smash_factor

def estimate_carry(ball_speed):
    ball_speed_mph = ball_speed * 2.237
    carry_yards = ball_speed_mph * 2.3
    return carry_yards * 0.9144  # Convert to meters

def get_impact_speed_and_carry(club_tracker, fps, pixels_per_meter):
    if len(club_tracker.filtered_path) < 6:
        return 0.0, 0.0, 0.0

    # Use last few velocity samples to detect peak motion
    speeds = []
    for i in range(1, len(club_tracker.filtered_path)):
        x1, y1 = club_tracker.filtered_path[i - 1]
        x2, y2 = club_tracker.filtered_path[i]
        dist_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist_meters = dist_pixels / pixels_per_meter
        time_s = 1.0 / (fps * SLOW_MOTION_FACTOR)
        speeds.append(dist_meters / time_s)

    # Estimate club speed at impact
    club_speed = max(speeds[-5:])  # peak in last 5 frames

    # Estimate ball speed
    smash_factor = 1.48
    ball_speed = club_speed * smash_factor

    # Estimate carry distance from ball speed
    ball_speed_mph = ball_speed * 2.237
    carry_yards = ball_speed_mph * 2.3
    carry_meters = carry_yards * 0.9144

    return club_speed, ball_speed, carry_meters


def process_frames(front_frame, side_frame, frame_idx, ball_tracker, front_kf, side_kf, calibration, fps):
    front_annotated = cv2.resize(front_frame, (640, 480))
    side_annotated = cv2.resize(side_frame, (640, 480))

    front_ball_bbox, _, front_club_bbox, _ = detect_objects(front_annotated)
    side_ball_bbox, _, side_club_bbox, _ = detect_objects(side_annotated)

    front_ball_pos = center_of_bbox(front_ball_bbox)
    side_ball_pos = center_of_bbox(side_ball_bbox)
    front_club_pos = center_of_bbox(front_club_bbox)
    side_club_pos = center_of_bbox(side_club_bbox)

    if front_club_pos:
        front_kf.update(front_club_pos)
    if side_club_pos:
        side_kf.update(side_club_pos)

    ball_tracker.push_position(frame_idx, front_ball_pos, side_ball_pos)

    if len(front_kf.filtered_path) > 1:
        for i in range(1, len(front_kf.filtered_path)):
            cv2.line(front_annotated, front_kf.filtered_path[i-1], front_kf.filtered_path[i], (255, 0, 0), 2)

    if len(side_kf.filtered_path) > 1:
        for i in range(1, len(side_kf.filtered_path)):
            cv2.line(side_annotated, side_kf.filtered_path[i-1], side_kf.filtered_path[i], (255, 0, 0), 2)

    for i in range(1, len(ball_tracker.positions_3d)):
        p1f = calibration.project_3d_to_front(ball_tracker.positions_3d[i-1])
        p2f = calibration.project_3d_to_front(ball_tracker.positions_3d[i])
        cv2.line(front_annotated, tuple(map(int, p1f)), tuple(map(int, p2f)), (0, 255, 0), 2)

        p1s = calibration.project_3d_to_side(ball_tracker.positions_3d[i-1])
        p2s = calibration.project_3d_to_side(ball_tracker.positions_3d[i])
        cv2.line(side_annotated, tuple(map(int, p1s)), tuple(map(int, p2s)), (0, 255, 0), 2)

    # Calculate and display speeds
    club_speed = calculate_club_speed_2d(front_kf, fps, 150)
    ball_speed = estimate_ball_speed(club_speed)
    carry = estimate_carry(ball_speed)

    cv2.putText(front_annotated, f"Club Speed: {club_speed:.1f} m/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(front_annotated, f"Ball Speed: {ball_speed:.1f} m/s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(front_annotated, f"Carry: {carry:.1f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 1)

    combined = np.zeros((480, 1280, 3), dtype=np.uint8)
    combined[:, :640] = front_annotated
    combined[:, 640:] = side_annotated

    club_speed, ball_speed, carry = get_impact_speed_and_carry(front_kf, fps, PIXELS_PER_METER)

    print(f"Club Speed at impact: {club_speed:.2f} m/s")
    print(f"Ball Speed: {ball_speed:.2f} m/s")
    print(f"Estimated Carry: {carry:.2f} m")
    return combined

def main():
    front_cap = cv2.VideoCapture(FRONT_VIDEO_PATH)
    side_cap = cv2.VideoCapture(SIDE_VIDEO_PATH)

    calibration = GolfCameraCalibration(CAMERA_MATRIX_FRONT, CAMERA_MATRIX_SIDE, CAMERA_OFFSET)
    ball_tracker = BallTracker3D(calibration)
    front_kf = KalmanTracker()
    side_kf = KalmanTracker()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(f"output_{timestamp}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480))

    frame_idx = 0
    while True:
        ret1, f1 = front_cap.read()
        ret2, f2 = side_cap.read()
        if not ret1 or not ret2:
            break
        frame = process_frames(f1, f2, frame_idx, ball_tracker, front_kf, side_kf, calibration, 30)
        out.write(frame)
        if DISPLAY_FRAMES:
            cv2.imshow("FlightSight Pro", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break
        frame_idx += 1

    front_cap.release()
    side_cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
