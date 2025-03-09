"""
FlightSight Pro - Advanced Golf Shot Analysis System
Dual-Angle Ball and Club Tracking with 3D Trajectory Reconstruction

This system uses computer vision and machine learning to track golf shots from two camera angles,
calculate shot parameters, and visualize the ball trajectory in 3D space.
"""

# GPU optimization
import os
# Set environment variables for optimal GPU performance
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Enable memory growth to avoid OOM errors
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
import time
from numpy.polynomial import Polynomial
import logging
import os.path
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from datetime import datetime
import gc  # For manual garbage collection

###############################################################################
# CONFIGURATION
###############################################################################
# Upload flag - set to False since we're using local files
UPLOAD_VIDEOS = False

# Video paths
FRONT_VIDEO_PATH = "Test_videos/archive/TV2_bh.mp4"  # Behind golfer view
SIDE_VIDEO_PATH = "Test_videos/archive/TV2_bs.mp4"    # Side view

# YOLO model
YOLO_MODEL_PATH = "PTs/best_tunning.pt"  # YOLO model

# Classes for YOLO
BALL_CLASS_ID = 0    # Index for ball in your YOLO model
CLUB_CLASS_ID = 2    # Index for club in your YOLO model

# Detection parameters
CONF_THRESHOLD = 0.38
IOU_THRESHOLD = 0.2

# Display control
DISPLAY_FRAMES = True  # Set to False to disable frame display during processing
PROCESS_INTERVAL = 1    # Process every frame (set higher for faster preview)
BATCH_SIZE = 1          # No batching

# Target FPS for processing
TARGET_FPS = 30

# Physical parameters
GRAVITY = 9.81
LAUNCH_SPEED_THRESHOLD = 0.5
PIXELS_PER_METER_FRONT = 150.0  # Scaling factor
PIXELS_PER_METER_SIDE = 150.0   # Scaling factor

# Smash factor (ratio of ball speed to club head speed)
SMASH_FACTOR = 1.48  # Typical driver smash factor

# Distance calculation constants
CARRY_FACTOR = 2.3  # Yards of carry per mph of ball speed (approximation)
METERS_PER_YARD = 0.9144  # Conversion factor

# ROI tracking settings
ROI_MARGIN_INITIAL = 200
ROI_MARGIN_GROWTH = 50
ROI_MARGIN_MAX = 800

# Physical limits for validation
MAX_PHYSICAL_SPEED = 100.0  # m/s (~223 mph)
MIN_PHYSICAL_SPEED = 20.0   # m/s (~45 mph)
MAX_LAUNCH_ANGLE = 60.0     # degrees
MIN_LAUNCH_ANGLE = 2.0      # degrees

# Direction handling
AUTO_DETECT_DIRECTION = True
FORCE_DIRECTION_RIGHT = True
FORCE_DIRECTION_LEFT = False

# Wind and spin parameters
WIND_SPEED = 2.0       # m/s
WIND_DIRECTION = 0.0   # degrees (0 = tailwind)
BACKSPIN_RPM = 3000.0  # Initial backspin
SIDESPIN_RPM = 0.0     # Initial sidespin

# Default carry distance when calculation fails
DEFAULT_CARRY_DISTANCE = 200.0  # meters

SLOW_MOTION_FACTOR = 8.0  

# Frame window adjustments for slow motion analysis
FRAME_WINDOW_SIZE = int(5 * SLOW_MOTION_FACTOR)  # Wider window for trajectory analysis
IMPACT_DETECTION_WINDOW = int(20 * SLOW_MOTION_FACTOR)  # More frames to detect impact

# Detection thresholds adjusted for slower apparent movement
MIN_CLUB_SPEED_THRESHOLD = 5.0 / SLOW_MOTION_FACTOR  # Lower threshold for swing detection
MIN_MOVEMENT_THRESHOLD = 3.0 / SLOW_MOTION_FACTOR  # Smaller movement per frame threshold

# Global variables to store launch parameters
launch_speed = 0.0
launch_angle_vertical = 0.0
launch_angle_horizontal = 0.0
predicted_carry = 0.0
shot_direction = 1  # Default right direction
club_speed = 0.0    # Club head speed

# Camera calibration parameters
CAMERA_MATRIX_FRONT = np.array([
    [1000, 0, 320],  # fx, 0, cx
    [0, 1000, 240],  # 0, fy, cy
    [0, 0, 1]        # 0, 0, 1
])

CAMERA_MATRIX_SIDE = np.array([
    [1000, 0, 320],  # fx, 0, cx
    [0, 1000, 240],  # 0, fy, cy
    [0, 0, 1]        # 0, 0, 1
])

# Camera offset (side camera position relative to front camera)
# For golf analysis with perpendicular views, side camera ~3-5m to right
CAMERA_OFFSET = np.array([5.0, 0.0, 0.0])  # X, Y, Z in meters

# Debug configuration
DEBUG_MODE = True
DEBUG_SAVE_FRAMES = True
DEBUG_FRAME_DIR = "debug_frames/"  # Local directory for debug frames

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Just log to stdout for notebooks
    ]
)

# GPU configuration for stability
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Found {device_count} GPUs! Using primary GPU for stability.")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # Use only the first GPU to avoid the batch size error with YOLO
        device = "0"  # Use single GPU for YOLO inference
    else:
        print(f"Found 1 GPU: {torch.cuda.get_device_name(0)}")
        device = "0"  # Use single GPU
else:
    print("No GPU found. Using CPU.")
    device = "cpu"

###############################################################################
# DISPLAY HELPER FUNCTIONS
###############################################################################

def draw_metrics_box(img, x, y, width, height, alpha=0.5):
    """
    Draw a semi-transparent background box for metrics display.
    
    Args:
        img: OpenCV image to draw on
        x, y: Top-left corner coordinates
        width, height: Box dimensions
        alpha: Transparency level (0-1)
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


###############################################################################
# YOLO MODEL LOADING
###############################################################################

# Global model variable
model = None  

def load_yolo_model():
    """
    Load the YOLO model and verify inference works.
    Handles fallback to CPU if GPU inference fails.
    """
    global model
    
    # Load the YOLO model
    try:
        model = YOLO(YOLO_MODEL_PATH)
        # Fuse layers if available for better performance
        if hasattr(model, 'fuse') and callable(getattr(model, 'fuse')):
            model.fuse()
        
        logging.info(f"YOLO model loaded (will use {device} device)")
        
        # Test inference with small batch to verify everything works
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        
        # Handle potential API changes in YOLO
        try:
            # First try with standard YOLOv8-style params
            _ = model(dummy_image_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        except TypeError:
            # Fall back to simpler params if API changed
            _ = model(dummy_image_rgb, conf=CONF_THRESHOLD)
            
        logging.info("YOLO model test inference successful")
    except Exception as e:
        logging.error(f"YOLO model test inference failed: {e}")
        logging.warning("Falling back to CPU if GPU inference fails")
        
        # If GPU inference fails, try on CPU
        if device != "cpu" and torch.cuda.is_available():
            try:
                # Try again with CPU
                model = YOLO(YOLO_MODEL_PATH)
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
                try:
                    _ = model(dummy_image_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device="cpu")
                except TypeError:
                    _ = model(dummy_image_rgb, conf=CONF_THRESHOLD, device="cpu")
                    
                logging.info("YOLO model CPU inference successful (fallback)")
            except Exception as e2:
                logging.error(f"YOLO model CPU inference also failed: {e2}")
                raise RuntimeError("Failed to initialize YOLO model for inference")


###############################################################################
# CAMERA CALIBRATION AND 3D RECONSTRUCTION
###############################################################################

class GolfCameraCalibration:
    """
    Enhanced camera calibration specifically for golf shot analysis with behind and side views.
    Handles triangulation of 3D points from dual-view 2D coordinates.
    """
    
    def __init__(self, camera_matrix_front, camera_matrix_side, camera_offset):
        """
        Initialize camera calibration with camera matrices and physical setup.
        
        Args:
            camera_matrix_front: 3x3 intrinsic matrix for front camera
            camera_matrix_side: 3x3 intrinsic matrix for side camera
            camera_offset: 3D offset (x,y,z) of side camera relative to front
        """
        self.camera_matrix_front = camera_matrix_front
        self.camera_matrix_side = camera_matrix_side
        self.camera_offset = camera_offset
        
        # For golf shots, define orientation better:
        # Front camera (behind golfer) looks down the target line
        self.R_front = np.eye(3)
        
        # Side camera is rotated 90 degrees around y-axis to view from side
        # For right-handed golf setup, side camera is typically on the right side
        self.R_side = Rotation.from_euler('y', -90, degrees=True).as_matrix()
        
        # Translation vectors
        self.t_front = np.zeros(3)  # Front camera at origin
        self.t_side = camera_offset  # Side camera offset
        
        # Projection matrices
        self.P_front = self.camera_matrix_front @ np.hstack((self.R_front, self.t_front.reshape(3, 1)))
        self.P_side = self.camera_matrix_side @ np.hstack((self.R_side, self.t_side.reshape(3, 1)))
        
        # For golf, we need appropriate scaling to ensure realistic distances
        # A typical driver shot might travel 200-250 yards
        self.distance_scale_factor = 15.0  # Adjusted for golf distances
        
        logging.info("Golf-specific camera calibration initialized")
    
    def triangulate_point(self, point_front, point_side):
        """
        Triangulate a 3D point from corresponding points in two views.
        
        Args:
            point_front: (x, y) in front view
            point_side: (x, y) in side view
            
        Returns:
            np.array: 3D point (x, y, z) in world coordinates
        """
        # Convert to homogeneous coordinates
        point_front_hom = np.array([point_front[0], point_front[1], 1.0])
        point_side_hom = np.array([point_side[0], point_side[1], 1.0])
        
        # Create the DLT matrix
        A = np.zeros((4, 4))
        A[0] = point_front_hom[0] * self.P_front[2] - self.P_front[0]
        A[1] = point_front_hom[1] * self.P_front[2] - self.P_front[1]
        A[2] = point_side_hom[0] * self.P_side[2] - self.P_side[0]
        A[3] = point_side_hom[1] * self.P_side[2] - self.P_side[1]
        
        # Solve using SVD
        _, _, vh = np.linalg.svd(A)
        point_3d_hom = vh[-1]
        
        # Convert from homogeneous to 3D coordinates
        point_3d = point_3d_hom[:3] / point_3d_hom[3]
        
        # Apply scaling factor to ensure realistic distances for golf shots
        point_3d = point_3d * self.distance_scale_factor
        
        return point_3d
    
    def project_3d_to_front(self, point_3d):
        """
        Project a 3D point onto the front view.
        
        Args:
            point_3d: 3D point (x, y, z)
            
        Returns:
            tuple: (x, y) coordinates in front view
        """
        point_3d_hom = np.append(point_3d, 1.0)
        point_front_hom = self.P_front @ point_3d_hom
        point_front = point_front_hom[:2] / point_front_hom[2]
        return point_front
    
    def project_3d_to_side(self, point_3d):
        """
        Project a 3D point onto the side view.
        
        Args:
            point_3d: 3D point (x, y, z)
            
        Returns:
            tuple: (x, y) coordinates in side view
        """
        point_3d_hom = np.append(point_3d, 1.0)
        point_side_hom = self.P_side @ point_3d_hom
        point_side = point_side_hom[:2] / point_side_hom[2]
        return point_side
    
    def validate_3d_point(self, point_3d, max_distance=10.0):
        """
        Check if a 3D point is valid (within reasonable range).
        
        Args:
            point_3d: 3D point coordinates
            max_distance: Maximum allowable distance from origin
            
        Returns:
            bool: True if point is valid, False otherwise
        """
        distance = np.linalg.norm(point_3d)
        return distance <= max_distance


###############################################################################
# IMPROVED BALL DETECTION
###############################################################################

def detect_ball_with_motion(frame, prev_frame, background_frame):
    """
    Detect the ball using motion and shape properties.
    Uses frame differencing and logical AND to isolate moving ball.
    
    Args:
        frame: Current frame
        prev_frame: Previous frame
        background_frame: Background model frame
    
    Returns:
        tuple: (ball_position, confidence) or (None, 0.0) if no ball detected
    """
    if prev_frame is None or background_frame is None:
        return None, 0.0
    
    try:
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame differences
        diff1 = cv2.absdiff(gray, prev_gray)
        diff2 = cv2.absdiff(gray, bg_gray)
        
        # Threshold the differences
        _, thresh1 = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)
        
        # Logical AND to find regions present in both differences
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the most ball-like contour (small, circular, fast-moving)
        best_ball = None
        best_conf = 0.0
        
        for contour in contours:
            # Check size
            area = cv2.contourArea(contour)
            if area < 10 or area > 500:  # Too small or too large
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # Not circular enough
                continue
            
            # Calculate position
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate confidence based on circularity and size
            confidence = circularity * min(1.0, area / 100.0)
            
            if confidence > best_conf:
                best_ball = (cx, cy)
                best_conf = confidence
        
        return best_ball, best_conf
        
    except Exception as e:
        logging.error(f"Error in motion-based ball detection: {e}")
        return None, 0.0


###############################################################################
# DETECTION FUNCTIONS
###############################################################################

def advanced_preprocess(frame):
    """
    Enhance frame for better object detection with contrast and edge enhancement.
    
    Args:
        frame: Input OpenCV frame (BGR format)
    
    Returns:
        Preprocessed frame or None if input is None
    """
    # Skip preprocessing if frame is None
    if frame is None:
        return None
        
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


def detect_objects(frame):
    """
    Detect ball and club in a frame using YOLO.
    
    Args:
        frame: Input frame for detection
    
    Returns:
        tuple: (ball_bbox, ball_conf, club_bbox, club_conf) - bounding boxes and confidence scores
    """
    if frame is None:
        return None, 0.0, None, 0.0
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Handle potential API changes in YOLO
    try:
        # First try with standard YOLOv8-style params
        results = model(frame_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    except TypeError:
        # Fall back to simpler params if API changed
        results = model(frame_rgb, conf=CONF_THRESHOLD)
    
    ball_bbox = None
    ball_conf = 0.0
    club_bbox = None
    club_conf = 0.0
    
    # Handle potential changes in results structure
    try:
        # Try standard YOLOv8 result format first
        if hasattr(results[0], 'boxes'):
            for det in results[0].boxes:
                cls_id = int(det.cls[0])
                conf = float(det.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                    
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                
                if cls_id == BALL_CLASS_ID and conf > ball_conf:
                    ball_bbox = [x1, y1, x2, y2]
                    ball_conf = conf
                elif cls_id == CLUB_CLASS_ID and conf > club_conf:
                    club_bbox = [x1, y1, x2, y2]
                    club_conf = conf
        # Alternative structure that might be used in YOLO
        elif hasattr(results, 'xyxy'):
            for i, det in enumerate(results.xyxy[0]):
                cls_id = int(det[-1])
                conf = float(det[-2])
                if conf < CONF_THRESHOLD:
                    continue
                    
                x1, y1, x2, y2 = map(int, det[:4].cpu().numpy())
                
                if cls_id == BALL_CLASS_ID and conf > ball_conf:
                    ball_bbox = [x1, y1, x2, y2]
                    ball_conf = conf
                elif cls_id == CLUB_CLASS_ID and conf > club_conf:
                    club_bbox = [x1, y1, x2, y2]
                    club_conf = conf
        # If structure completely different, try to adapt
        else:
            logging.warning("Unrecognized YOLO result format - trying to adapt")
            # Try to find detections in results
            if hasattr(results, 'pred') and len(results.pred) > 0:
                for det in results.pred[0]:
                    if len(det) >= 6:  # x1,y1,x2,y2,conf,cls
                        cls_id = int(det[5])
                        conf = float(det[4])
                        if conf < CONF_THRESHOLD:
                            continue
                            
                        x1, y1, x2, y2 = map(int, det[:4].cpu().numpy())
                        
                        if cls_id == BALL_CLASS_ID and conf > ball_conf:
                            ball_bbox = [x1, y1, x2, y2]
                            ball_conf = conf
                        elif cls_id == CLUB_CLASS_ID and conf > club_conf:
                            club_bbox = [x1, y1, x2, y2]
                            club_conf = conf
    except Exception as e:
        logging.error(f"Error parsing YOLO results: {e}")
    
    return ball_bbox, ball_conf, club_bbox, club_conf


def detect_object_in_roi(frame, roi_center, margin, desired_class):
    """
    Detect a specific object within a region of interest using YOLO.
    
    Args:
        frame: Input frame
        roi_center: (x, y) center of ROI
        margin: Pixel margin around center to define ROI
        desired_class: Class ID to detect
    
    Returns:
        tuple: (bbox, confidence) - Bounding box and confidence score, or (None, 0.0) if not found
    """
    try:
        if frame is None or roi_center is None:
            return None, 0.0
            
        h, w, _ = frame.shape
        cx, cy = map(int, roi_center)
        margin = int(margin)
        
        # Calculate ROI boundaries
        x1 = max(0, cx - margin)
        y1 = max(0, cy - margin)
        x2 = min(w, cx + margin)
        y2 = min(h, cy + margin)
        
        # Ensure ROI is valid
        if x2 <= x1 or y2 <= y1:
            logging.warning(f"Invalid ROI dimensions: width={x2-x1}, height={y2-y1}")
            return None, 0.0
        
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            logging.warning("Empty ROI detected")
            return None, 0.0
        
        # Convert BGR to RGB for YOLO
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Run detection on ROI with YOLO compatibility
        try:
            # Try standard params first
            results = model(roi_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        except TypeError:
            # Fall back to simpler params if API changed
            results = model(roi_rgb, conf=CONF_THRESHOLD)
        
        # Find best match for desired class
        best_det = None
        best_conf = -1
        
        # Handle potential changes in results structure
        try:
            # Standard YOLOv8 format
            if hasattr(results[0], 'boxes'):
                for det in results[0].boxes:
                    cls_id = int(det.cls[0])
                    conf = float(det.conf[0])
                    
                    if cls_id == desired_class and conf > best_conf:
                        best_det = det
                        best_conf = conf
                
                if best_det is not None:
                    bx1, by1, bx2, by2 = map(int, best_det.xyxy[0].cpu().numpy())
                    # Adjust coordinates back to original frame
                    return [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1], best_conf
            
            # Alternative YOLO format
            elif hasattr(results, 'xyxy'):
                for i, det in enumerate(results.xyxy[0]):
                    cls_id = int(det[-1])
                    conf = float(det[-2])
                    
                    if cls_id == desired_class and conf > best_conf:
                        best_conf = conf
                        x1_roi, y1_roi, x2_roi, y2_roi = map(int, det[:4].cpu().numpy())
                        # Adjust coordinates back to original frame
                        return [x1_roi + x1, y1_roi + y1, x2_roi + x1, y2_roi + y1], best_conf
            
            # Try to adapt to other potential formats
            else:
                logging.warning("Unrecognized YOLO result format in ROI detection")
                # Try to find detections in results
                if hasattr(results, 'pred') and len(results.pred) > 0:
                    for det in results.pred[0]:
                        if len(det) >= 6:  # x1,y1,x2,y2,conf,cls
                            cls_id = int(det[5])
                            conf = float(det[4])
                            
                            if cls_id == desired_class and conf > best_conf:
                                best_conf = conf
                                x1_roi, y1_roi, x2_roi, y2_roi = map(int, det[:4].cpu().numpy())
                                # Adjust coordinates back to original frame
                                return [x1_roi + x1, y1_roi + y1, x2_roi + x1, y2_roi + y1], best_conf
        
        except Exception as e:
            logging.error(f"Error parsing YOLO ROI results: {e}")
        
        return None, 0.0
        
    except Exception as e:
        logging.error(f"Error in ROI detection: {str(e)}")
        return None, 0.0


###############################################################################
# CLUB TRACKING CLASSES
###############################################################################

class EnhancedClubTracker:
    """
    Improved club tracker with temporal filtering and noise reduction.
    """
    
    def __init__(self):
        """Initialize club tracker with filtering parameters."""
        self.raw_path = []     # Raw club positions (x, y, confidence)
        self.filtered_path = [] # Filtered club positions
        self.velocity = []     # Club head velocity for smash factor calculation
        self.max_path_length = 100  # Increased from 30 to show more of the path
        self.min_movement_threshold = 3.0  # Minimum movement to register (pixels)
        self.confidence_threshold = 0.3  # Minimum confidence to accept detection
    
    def update(self, x, y, confidence=1.0):
        """
        Add new club position with temporal filtering.
        
        Args:
            x, y: Club position coordinates
            confidence: Detection confidence (0-1)
        """
        # Skip if position is None
        if x is None or y is None:
            return
            
        # Add raw position with confidence
        self.raw_path.append((x, y, confidence))
        
        # Trim path if too long
        if len(self.raw_path) > self.max_path_length:
            self.raw_path.pop(0)
        
        # Apply filtering only if we have enough points
        if len(self.raw_path) >= 3:
            # Get recent positions
            recent = self.raw_path[-3:]
            
            # Calculate weighted average based on confidence
            total_weight = sum(p[2] for p in recent)
            if total_weight > 0:
                avg_x = sum(p[0] * p[2] for p in recent) / total_weight
                avg_y = sum(p[1] * p[2] for p in recent) / total_weight
                
                # Only add if we have at least 2 points in filtered path
                if len(self.filtered_path) < 2 or self.is_significant_movement(avg_x, avg_y):
                    self.filtered_path.append((int(avg_x), int(avg_y)))
                    
                    # Calculate velocity if we have at least 2 filtered points
                    if len(self.filtered_path) >= 2:
                        p1 = self.filtered_path[-2]
                        p2 = self.filtered_path[-1]
                        vx = p2[0] - p1[0]
                        vy = p2[1] - p1[1]
                        self.velocity.append((vx, vy))
            
            # Trim filtered path if too long
            if len(self.filtered_path) > self.max_path_length:
                self.filtered_path.pop(0)
                
            # Trim velocity history
            if len(self.velocity) > self.max_path_length:
                self.velocity.pop(0)
        
        elif len(self.raw_path) > 0:
            # If not enough points, just add the raw position
            x, y, _ = self.raw_path[-1]
            if not self.filtered_path or self.is_significant_movement(x, y):
                self.filtered_path.append((int(x), int(y)))
    
    def is_significant_movement(self, x, y):
        """
        Check if movement from last position is significant.
        
        Args:
            x, y: Current position coordinates
            
        Returns:
            bool: True if movement is significant, False otherwise
        """
        if not self.filtered_path:
            return True
        
        last_x, last_y = self.filtered_path[-1]
        distance = ((x - last_x)**2 + (y - last_y)**2)**0.5
        return distance >= self.min_movement_threshold
    
    def get_path(self):
        """
        Get filtered club path.
        
        Returns:
            list: List of (x, y) coordinates representing the club path
        """
        return self.filtered_path
    
    def get_average_velocity(self, n_samples=5):
        """
        Calculate average velocity from recent samples.
        
        Args:
            n_samples: Number of recent samples to average
            
        Returns:
            tuple: (avg_vx, avg_vy) - Average velocity components
        """
        if len(self.velocity) < n_samples:
            return (0, 0)
        
        recent = self.velocity[-n_samples:]
        avg_vx = sum(v[0] for v in recent) / len(recent)
        avg_vy = sum(v[1] for v in recent) / len(recent)
        return (avg_vx, avg_vy)
    
    def get_club_speed(self, fps, pixels_per_meter):
        """
        Calculate club head speed with slow motion compensation.
        
        Args:
            fps: Frame rate of slow motion video
            pixels_per_meter: Conversion factor from pixels to meters
            
        Returns:
            float: Club speed in m/s (real-world value)
        """
        if len(self.velocity) < 3:
            return 0.0
        
        # Calculate average velocity from recent samples
        recent = self.velocity[-3:]
        avg_vx = sum(v[0] for v in recent) / len(recent)
        avg_vy = sum(v[1] for v in recent) / len(recent)
        
        # Convert to m/s with slow motion compensation
        speed_pixels = (avg_vx**2 + avg_vy**2)**0.5
        speed_m_s = speed_pixels / pixels_per_meter * fps * SLOW_MOTION_FACTOR
        return speed_m_s
    
    def is_in_backswing(self):
        """
        Determine if club is in backswing phase, with slow motion adjustments.
        
        Returns:
            bool: True if in backswing, False otherwise
        """
        # Use wider window for slow motion detection
        window_size = FRAME_WINDOW_SIZE
        if len(self.filtered_path) < window_size:
            return False
        
        # Analyze movement over window
        recent = self.filtered_path[-window_size:]
        start_x, start_y = recent[0]
        end_x, end_y = recent[-1]
        
        # Calculate movement
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Scaled thresholds for slow motion
        threshold_x = 10 / SLOW_MOTION_FACTOR  # Right movement threshold
        threshold_y = -10 / SLOW_MOTION_FACTOR  # Upward movement threshold
        
        return dx > threshold_x and dy < threshold_y
    
    def is_in_downswing(self):
        """
        Determine if club is in downswing phase, with slow motion adjustments.
        
        Returns:
            bool: True if in downswing, False otherwise
        """
        # Use wider window for slow motion detection
        window_size = FRAME_WINDOW_SIZE
        if len(self.filtered_path) < window_size:
            return False
        
        # Analyze movement over window
        recent = self.filtered_path[-window_size:]
        start_x, start_y = recent[0]
        end_x, end_y = recent[-1]
        
        # Calculate movement
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Scaled thresholds for slow motion
        threshold_x = -15 / SLOW_MOTION_FACTOR  # Left movement threshold
        threshold_y = 10 / SLOW_MOTION_FACTOR   # Downward movement threshold
        
        return dx < threshold_x and dy > threshold_y


class DualViewEnhancedClubTracker:
    """
    Enhanced club tracker for dual-view setup with 3D reconstruction.
    """
    
    def __init__(self, calibration):
        """
        Initialize dual-view club tracker with calibration.
        
        Args:
            calibration: GolfCameraCalibration instance for 3D reconstruction
        """
        self.front_tracker = EnhancedClubTracker()
        self.side_tracker = EnhancedClubTracker()
        self.calib = calibration
        self.positions_3d = []  # 3D club head positions (frame_idx, x, y, z)
        self.max_3d_length = 100  # Maximum number of 3D positions to store
    
    def update(self, frame_idx, front_pos, front_conf, side_pos, side_conf):
        """
        Update club positions in both views and reconstruct 3D position if possible.
        
        Args:
            frame_idx: Current frame index
            front_pos: Club position in front view (x, y)
            front_conf: Detection confidence in front view
            side_pos: Club position in side view (x, y)
            side_conf: Detection confidence in side view
            
        Returns:
            bool: True if 3D position was successfully triangulated, False otherwise
        """
        if front_pos is not None:
            self.front_tracker.update(front_pos[0], front_pos[1], front_conf)
        
        if side_pos is not None:
            self.side_tracker.update(side_pos[0], side_pos[1], side_conf)
        
        # If we have both views with good confidence, triangulate 3D position
        if (front_pos is not None and side_pos is not None and 
            front_conf > 0.3 and side_conf > 0.3):
            try:
                point_3d = self.calib.triangulate_point(front_pos, side_pos)
                
                # Validate the 3D point
                if self.calib.validate_3d_point(point_3d):
                    self.positions_3d.append((frame_idx, point_3d[0], point_3d[1], point_3d[2]))
                    
                    # Trim 3D positions list if needed
                    if len(self.positions_3d) > self.max_3d_length:
                        self.positions_3d.pop(0)
                    
                    return True
            except Exception as e:
                logging.error(f"Club triangulation error: {e}")
        
        return False
    
    def get_club_speed_3d(self, fps):
        """
        Calculate 3D club head speed with slow motion compensation.
        
        Args:
            fps: Frame rate of slow motion video
            
        Returns:
            float: Club head speed in m/s (real-world value)
        """
        if len(self.positions_3d) < 3:
            return 0.0
        
        # Use more positions for better averaging in slow motion
        recent = self.positions_3d[-10:]
        
        # Calculate speed between consecutive positions
        speeds = []
        for i in range(1, len(recent)):
            p1 = np.array(recent[i-1][1:4])  # 3D point (x, y, z)
            p2 = np.array(recent[i][1:4])
            dt = (recent[i][0] - recent[i-1][0]) / fps  # Time in seconds
            
            if dt > 0:
                distance = np.linalg.norm(p2 - p1)  # 3D distance
                # Scale up for slow motion compensation
                speed = distance / dt * SLOW_MOTION_FACTOR
                speeds.append(speed)
        
        # Return average of highest speeds
        if speeds:
            top_speeds = sorted(speeds, reverse=True)[:3]
            avg_speed = sum(top_speeds) / len(top_speeds)
            return avg_speed
        return 0.0
    
    def get_recent_3d_positions(self, n_samples=5):
        """
        Get n most recent 3D positions.
        
        Args:
            n_samples: Number of recent samples to return
            
        Returns:
            list: List of recent 3D positions
        """
        if len(self.positions_3d) < n_samples:
            return self.positions_3d
        return self.positions_3d[-n_samples:]
    
    def is_in_swing(self):
        """
        Determine if player is actively swinging.
        
        Returns:
            bool: True if in active swing, False otherwise
        """
        return (self.front_tracker.is_in_backswing() or 
                self.front_tracker.is_in_downswing() or
                self.side_tracker.is_in_backswing() or
                self.side_tracker.is_in_downswing())


###############################################################################
# BALL TRACKING CLASSES
###############################################################################

class BallTracker:
    """
    Base class for tracking ball centers in a single view.
    """
    
    def __init__(self):
        """Initialize ball tracker with default parameters."""
        self.positions = []  # list of (frame_idx, x, y, confidence)
        self.missed_frames = 0
        self.dynamic_margin = ROI_MARGIN_INITIAL

    def push_position(self, frame_idx, x, y, confidence=1.0):
        """
        Add a new ball position.
        
        Args:
            frame_idx: Frame index
            x, y: Ball coordinates
            confidence: Detection confidence (0-1)
        """
        self.positions.append((frame_idx, x, y, confidence))

    def get_latest_position(self):
        """
        Get most recent ball position.
        
        Returns:
            tuple: (x, y) position or None if no positions
        """
        if self.positions:
            return self.positions[-1][1:3]  # Return (x, y)
        return None

    def update_missed(self, found):
        """
        Update missed frames counter and adjust ROI size accordingly.
        
        Args:
            found: True if ball was found in current frame, False otherwise
        """
        if found:
            self.missed_frames = 0
            self.dynamic_margin = ROI_MARGIN_INITIAL
        else:
            self.missed_frames += 1
            self.dynamic_margin = min(ROI_MARGIN_INITIAL + ROI_MARGIN_GROWTH * self.missed_frames,
                                      ROI_MARGIN_MAX)


class DualViewBallTracker:
    """
    Integrates tracking from two camera views for 3D ball tracking.
    """
    
    def __init__(self, camera_calibration):
        """
        Initialize dual-view ball tracker.
        
        Args:
            camera_calibration: GolfCameraCalibration instance for 3D reconstruction
        """
        self.front_tracker = BallTracker()
        self.side_tracker = BallTracker()
        self.calib = camera_calibration
        self.positions_3d = []  # 3D positions (frame_idx, x, y, z, confidence)
        self.missed_frames = 0
    
    def push_position(self, frame_idx, front_pos, side_pos, front_conf=1.0, side_conf=1.0):
        """
        Add corresponding positions from both views and calculate 3D position.
        
        Args:
            frame_idx: Current frame index
            front_pos: Ball position in front view (x, y)
            front_conf: Detection confidence in front view
            side_pos: Ball position in side view (x, y)
            side_conf: Detection confidence in side view
            
        Returns:
            bool: True if 3D position was successfully triangulated, False otherwise
        """
        # Add to individual trackers
        if front_pos is not None:
            self.front_tracker.push_position(frame_idx, front_pos[0], front_pos[1], front_conf)
        
        if side_pos is not None:
            self.side_tracker.push_position(frame_idx, side_pos[0], side_pos[1], side_conf)
        
        # If we have both views, triangulate 3D position
        if front_pos is not None and side_pos is not None:
            try:
                point_3d = self.calib.triangulate_point(front_pos, side_pos)
                
                # Validate the 3D point
                if self.calib.validate_3d_point(point_3d):
                    confidence = (front_conf + side_conf) / 2.0
                    self.positions_3d.append((frame_idx, point_3d[0], point_3d[1], point_3d[2], confidence))
                    self.missed_frames = 0
                    return True
                else:
                    logging.warning(f"Invalid 3D position at frame {frame_idx}: {point_3d}")
            except Exception as e:
                logging.error(f"Triangulation error at frame {frame_idx}: {e}")
        
        # If we couldn't add a 3D position
        self.missed_frames += 1
        return False
    
    def get_latest_3d_position(self):
        """
        Get the most recent 3D position.
        
        Returns:
            tuple: (x, y, z) coordinates or None if no positions
        """
        if self.positions_3d:
            return self.positions_3d[-1][1:4]  # Return (x, y, z)
        return None
    
    def get_3d_positions(self):
        """
        Get all 3D positions.
        
        Returns:
            list: List of (x, y, z) tuples for all tracked positions
        """
        return [(p[1], p[2], p[3]) for p in self.positions_3d]
    
    def predict_next_3d(self, frame_idx, window=5, order=2):
        """
        Predict next 3D position using polynomial fitting.
        
        Args:
            frame_idx: Frame index to predict for
            window: Window size for fitting
            order: Polynomial order
            
        Returns:
            tuple: Predicted 3D position (x, y, z) or last known position
        """
        if len(self.positions_3d) < 3:
            return self.get_latest_3d_position()
        
        try:
            # Get recent positions (up to window size)
            recent = self.positions_3d[-window:]
            frames = np.array([p[0] for p in recent])
            x_vals = np.array([p[1] for p in recent])
            y_vals = np.array([p[2] for p in recent])
            z_vals = np.array([p[3] for p in recent])
            
            # Fit polynomials
            poly_x = Polynomial.fit(frames, x_vals, order)
            poly_y = Polynomial.fit(frames, y_vals, order)
            poly_z = Polynomial.fit(frames, z_vals, order)
            
            # Predict next position
            pred_x = poly_x(frame_idx)
            pred_y = poly_y(frame_idx)
            pred_z = poly_z(frame_idx)
            
            return (pred_x, pred_y, pred_z)
        except Exception as e:
            logging.warning(f"3D prediction error: {e}")
            return self.get_latest_3d_position()


###############################################################################
# SWING STATE DETECTION
###############################################################################

class SwingStateDetector:
    """
    Advanced swing state detector using club and ball tracking.
    Detects setup, backswing, downswing, impact, and follow-through.
    """
    
    def __init__(self):
        """Initialize swing state detector with state definitions."""
        self.states = ["SETUP", "BACKSWING", "DOWNSWING", "IMPACT", "FOLLOW_THROUGH"]
        self.current_state = "SETUP"
        self.state_history = []  # For temporal filtering
        self.history_length = 5  # History length for state smoothing
        self.state_confidence = 1.0  # Confidence in current state
        self.time_in_state = 0  # Frames spent in current state
    
    def update(self, club_tracker, ball_tracker, frame_idx):
        """
        Update swing state based on club and ball tracking.
        
        Args:
            club_tracker: DualViewEnhancedClubTracker instance
            ball_tracker: DualViewBallTracker instance
            frame_idx: Current frame index
            
        Returns:
            tuple: (state, confidence)
        """
        # Get club data
        front_club = club_tracker.front_tracker
        side_club = club_tracker.side_tracker
        
        # Check if ball has moved (potential impact)
        ball_moved = False
        if len(ball_tracker.positions_3d) >= 3:
            first_pos = np.array(ball_tracker.positions_3d[0][1:4])
            current_pos = np.array(ball_tracker.positions_3d[-1][1:4])
            movement = np.linalg.norm(current_pos - first_pos)
            ball_moved = movement > 0.05  # 5cm threshold
        
        # Determine raw state
        raw_state = self.current_state
        
        if ball_moved:
            if self.current_state in ["IMPACT", "FOLLOW_THROUGH"]:
                raw_state = "FOLLOW_THROUGH"
            else:
                raw_state = "IMPACT"
        
        elif front_club.is_in_backswing() or side_club.is_in_backswing():
            if self.current_state in ["SETUP", "BACKSWING"]:
                raw_state = "BACKSWING"
        
        elif front_club.is_in_downswing() or side_club.is_in_downswing():
            if self.current_state in ["BACKSWING", "DOWNSWING"]:
                raw_state = "DOWNSWING"
        
        elif club_tracker.is_in_swing():
            # General swing detected but not clearly backswing or downswing
            if self.current_state == "SETUP":
                raw_state = "BACKSWING"
            elif self.current_state == "FOLLOW_THROUGH":
                raw_state = "FOLLOW_THROUGH"  # Stay in follow-through
            elif self.current_state == "IMPACT":
                raw_state = "FOLLOW_THROUGH"  # Progress to follow-through after impact
        
        else:
            # No significant movement
            if self.current_state == "FOLLOW_THROUGH":
                # Stay in follow-through once we've reached it
                raw_state = "FOLLOW_THROUGH"
            elif self.current_state == "IMPACT":
                raw_state = "FOLLOW_THROUGH"  # Progress after impact
            else:
                raw_state = "SETUP"
        
        # Add to state history
        self.state_history.append(raw_state)
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)
        
        # Determine most common state in history (temporal filtering)
        state_counts = {}
        for state in self.states:
            state_counts[state] = self.state_history.count(state)
        
        # Get most common state
        filtered_state = max(state_counts, key=state_counts.get)
        
        # Only change state if it's been consistent
        if (filtered_state != self.current_state and 
            state_counts[filtered_state] >= self.history_length * 0.6):
            
            # Update state
            self.current_state = filtered_state
            self.time_in_state = 0
            
            # Set confidence based on how dominant this state is in history
            self.state_confidence = state_counts[filtered_state] / self.history_length
        else:
            # Stay in current state but update time and confidence
            self.time_in_state += 1
            
            # Set confidence based on how dominant this state is in history
            if self.current_state in state_counts:
                self.state_confidence = state_counts[self.current_state] / self.history_length
        
        return self.current_state, self.state_confidence


###############################################################################
# TRAJECTORY CALCULATION FUNCTIONS
###############################################################################

def calculate_trajectory_polynomial(ball_positions, frame_rate):
    """
    Calculate trajectory using polynomial approximation method.
    Split trajectory into segments and use least-squares fitting.
    
    Args:
        ball_positions: List of 3D ball positions [(x, y, z), ...]
        frame_rate: Frame rate of video
        
    Returns:
        tuple: (smooth_trajectory, carry_distance, smooth_times)
    """
    if len(ball_positions) < 5:
        return None, 0, None
    
    try:
        # Convert positions to numpy arrays for computation
        positions = np.array(ball_positions)
        
        # Create time array (assuming constant frame rate)
        times = np.arange(len(positions)) / (frame_rate * SLOW_MOTION_FACTOR)
        
        # Split trajectory into pre-impact and post-impact segments
        # Find point of maximum angle change in trajectory
        max_angle_diff = 0
        split_idx = 0
        
        for i in range(2, len(positions) - 1):
            v1 = positions[i] - positions[i-1]
            v2 = positions[i+1] - positions[i]
            
            # Calculate angle between consecutive velocity vectors
            dot_product = np.dot(v1, v2)
            magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if magnitudes > 0:
                angle = np.arccos(min(1.0, max(-1.0, dot_product / magnitudes)))
                if angle > max_angle_diff:
                    max_angle_diff = angle
                    split_idx = i
        
        # Fit polynomials to each segment (use higher degree for better accuracy)
        if split_idx > 2 and split_idx < len(positions) - 2:
            # Pre-impact segment
            pre_impact = positions[:split_idx+1]
            pre_times = times[:split_idx+1]
            
            # Post-impact segment
            post_impact = positions[split_idx:]
            post_times = times[split_idx:]
            
            # Fit 3D polynomials (one for each coordinate)
            pre_x = np.polyfit(pre_times, pre_impact[:, 0], 2)
            pre_y = np.polyfit(pre_times, pre_impact[:, 1], 2)
            pre_z = np.polyfit(pre_times, pre_impact[:, 2], 2)
            
            post_x = np.polyfit(post_times, post_impact[:, 0], 2)
            post_y = np.polyfit(post_times, post_impact[:, 1], 2)
            post_z = np.polyfit(post_times, post_impact[:, 2], 2)
            
            # Generate smooth trajectory
            smooth_times = np.linspace(times[0], times[-1] + 2.0, 100)  # Extend prediction
            smooth_trajectory = []
            
            for t in smooth_times:
                if t <= times[split_idx]:
                    # Pre-impact
                    x = np.polyval(pre_x, t)
                    y = np.polyval(pre_y, t)
                    z = np.polyval(pre_z, t)
                else:
                    # Post-impact
                    x = np.polyval(post_x, t)
                    y = np.polyval(post_y, t)
                    z = np.polyval(post_z, t)
                
                smooth_trajectory.append([x, y, z])
            
            # Calculate carry distance (flight distance to ground impact)
            # Find where z becomes <= 0
            for i in range(1, len(smooth_trajectory)):
                if smooth_trajectory[i][2] <= 0:
                    impact_point = smooth_trajectory[i-1]
                    # Calculate ground distance from origin
                    carry = np.sqrt(impact_point[0]**2 + impact_point[1]**2)
                    return smooth_trajectory, carry, smooth_times
            
            # If no ground impact found, use final point
            final_point = smooth_trajectory[-1]
            carry = np.sqrt(final_point[0]**2 + final_point[1]**2)
            return smooth_trajectory, carry, smooth_times
        
        # If splitting fails, fit a single polynomial to the whole trajectory
        x_poly = np.polyfit(times, positions[:, 0], 3)
        y_poly = np.polyfit(times, positions[:, 1], 3)
        z_poly = np.polyfit(times, positions[:, 2], 3)
        
        # Generate smooth trajectory
        smooth_times = np.linspace(times[0], times[-1] + 2.0, 100)  # Extend forecast
        smooth_trajectory = []
        
        for t in smooth_times:
            x = np.polyval(x_poly, t)
            y = np.polyval(y_poly, t)
            z = np.polyval(z_poly, t)
            smooth_trajectory.append([x, y, z])
            
            # Stop if predicted point hits ground
            if z <= 0 and t > times[-1]:
                # Calculate carry distance
                carry = np.sqrt(x**2 + y**2)
                return smooth_trajectory[:len(smooth_trajectory)-1], carry, smooth_times[:len(smooth_trajectory)-1]
        
        # If no ground impact predicted, use final recorded position for carry estimate
        final_point = positions[-1]
        carry = np.sqrt(final_point[0]**2 + final_point[1]**2) * 1.5  # Scale by 1.5 for realistic carry
        return smooth_trajectory, carry, smooth_times
        
    except Exception as e:
        logging.error(f"Error in polynomial trajectory calculation: {e}")
        return None, 0, None


def calculate_launch_parameters_from_trajectory(ball_positions, fps):
    """
    Calculate launch parameters from 3D trajectory data.
    
    Args:
        ball_positions: List of 3D ball positions [(x, y, z), ...]
        fps: Frame rate of the slow motion video
        
    Returns:
        tuple: (speed, vertical_angle, horizontal_angle)
    """
    if len(ball_positions) < 5:
        return 0.0, 0.0, 0.0
    
    try:
        # Get first positions after impact
        positions = np.array(ball_positions[:5])
        
        # Calculate velocities between consecutive positions
        velocities = []
        for i in range(1, len(positions)):
            # Scale by SLOW_MOTION_FACTOR to get real-world velocity
            v = (positions[i] - positions[i-1]) * fps * SLOW_MOTION_FACTOR
            velocities.append(v)
        
        # Average velocity vector
        avg_velocity = np.mean(velocities, axis=0)
        
        # Calculate speed (magnitude of velocity)
        speed = np.linalg.norm(avg_velocity)
        
        # Calculate vertical launch angle (angle with XY plane)
        vertical_angle = np.degrees(np.arctan2(avg_velocity[2], np.sqrt(avg_velocity[0]**2 + avg_velocity[1]**2)))
        
        # Calculate horizontal launch angle (angle in XY plane)
        horizontal_angle = np.degrees(np.arctan2(avg_velocity[1], avg_velocity[0]))
        
        return speed, vertical_angle, horizontal_angle
        
    except Exception as e:
        logging.error(f"Error calculating launch parameters: {e}")
        return 45.0, 12.0, 0.0  # Default values


def calculate_carry_from_ball_speed(ball_speed_ms):
    """
    Calculate carry distance from ball speed using a simple empirical model.
    
    Args:
        ball_speed_ms: Ball speed in meters per second
        
    Returns:
        float: Estimated carry distance in meters
    """
    # Convert m/s to mph
    ball_speed_mph = ball_speed_ms * 2.237
    
    # Use empirical formula: approximately 2.3 yards of carry per 1 mph of ball speed
    # This is a simplification but works decently for drivers
    carry_yards = ball_speed_mph * CARRY_FACTOR
    
    # Convert to meters
    carry_meters = carry_yards * METERS_PER_YARD
    
    logging.info(f"Calculated carry from ball speed: {ball_speed_ms:.2f}m/s → {ball_speed_mph:.2f}mph → {carry_yards:.2f}yd → {carry_meters:.2f}m")
    
    return carry_meters


def estimate_carry_using_smash_factor(club_speed, smash_factor=SMASH_FACTOR):
    """
    Estimate carry distance using club speed and smash factor.
    
    Args:
        club_speed: Club head speed in m/s
        smash_factor: Ratio of ball speed to club head speed (default: 1.48 for driver)
        
    Returns:
        tuple: (ball_speed, carry_distance) in m/s and meters
    """
    # Calculate ball speed using smash factor
    ball_speed = club_speed * smash_factor
    
    # Calculate carry distance from ball speed
    carry = calculate_carry_from_ball_speed(ball_speed)
    
    logging.info(f"Estimated using smash factor: Club speed {club_speed:.2f}m/s * {smash_factor} = Ball speed {ball_speed:.2f}m/s → Carry {carry:.2f}m")
    
    return ball_speed, carry


def step_based_ballistics_with_spin_and_wind(v0, angle_deg, spin_rpm, wind_speed_m_s, wind_dir_deg, launch_height=0.0):
    """
    Physics-based ball trajectory simulation with spin and wind effects.
    
    Args:
        v0: Initial velocity (m/s)
        angle_deg: Launch angle in degrees
        spin_rpm: Ball spin rate in RPM
        wind_speed_m_s: Wind speed in m/s
        wind_dir_deg: Wind direction in degrees (0 = tailwind)
        launch_height: Initial height above ground (m)
        
    Returns:
        tuple: (trajectory_points, carry_distance)
    """
    # Handle negative angles if needed
    CLAMP_ANGLE = False  # Define this based on your requirements
    if CLAMP_ANGLE and angle_deg < 0:
        angle_deg = abs(angle_deg)
    
    # Ball properties
    radius = 0.02135  # m
    Cd = 0.2
    mass = 0.045
    area = math.pi * (radius**2)
    air_density = 1.225
    lift_mag_base = 0.285 * (1 - math.exp(-0.00026 * spin_rpm))
    
    # Convert degrees to radians
    def deg_to_rad(d):
        return d * math.pi / 180.0
    
    # Wind components
    wind_dir_rad = deg_to_rad(wind_dir_deg)
    wind_vx = wind_speed_m_s * math.cos(wind_dir_rad)
    wind_vy = wind_speed_m_s * math.sin(wind_dir_rad)
    
    # Initial velocity components
    vx = v0 * math.cos(deg_to_rad(angle_deg))
    vy = v0 * math.sin(deg_to_rad(angle_deg))
    
    # Initial position
    sx = 0.0
    sy = launch_height
    
    # Time step for simulation
    dt = 0.01
    
    # Store trajectory
    traj = [(sx, sy)]
    
    # Run simulation until ball hits ground
    while sy >= 0:
        # Relative velocity (ball - wind)
        rvx = vx - wind_vx
        rvy = vy - wind_vy
        vrel = math.sqrt(rvx**2 + rvy**2)
        
        # Drag force
        F_drag = 0.5 * air_density * Cd * area * (vrel**2)
        
        if vrel > 0:
            drag_ax = -(F_drag / mass) * (rvx / vrel)
            drag_ay = -(F_drag / mass) * (rvy / vrel)
        else:
            drag_ax = 0
            drag_ay = 0
        
        # Magnus effect (lift from spin)
        if vrel > 0:
            lift_mag = lift_mag_base * vrel
            
            # Vector perpendicular to relative velocity
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
        
        # Total acceleration
        ax = drag_ax + lift_ax
        ay = drag_ay + lift_ay - GRAVITY  # Gravity acts downwards
        
        # Update velocity
        vx += ax * dt
        vy += ay * dt
        
        # Update position
        sx += vx * dt
        sy += vy * dt
        
        # Add current position to trajectory
        traj.append((sx, sy))
        
        # Safety to prevent infinite loops
        if len(traj) > 200000:
            break
    
    # Calculate final landing position by interpolation
    if len(traj) >= 2 and traj[-2][1] > 0 and traj[-1][1] < 0:
        (x_prev, y_prev) = traj[-2]
        (x_last, y_last) = traj[-1]
        alpha = -y_prev / (y_last - y_prev)
        final_x = x_prev + alpha * (x_last - x_prev)
    else:
        final_x = traj[-1][0]
    
    return traj, final_x


###############################################################################
# VISUALIZATION FUNCTIONS
###############################################################################

def draw_enhanced_trajectory(image, traj_points, color=(255, 0, 0), thickness=3, 
                            draw_landing=True, landing_color=(0, 255, 255)):
    """
    Draw trajectory with enhanced visibility and landing point.
    
    Args:
        image: Image to draw on
        traj_points: List of (x, y) trajectory points
        color: Line color (default: blue)
        thickness: Line thickness
        draw_landing: Whether to highlight landing point
        landing_color: Color for landing point
    """
    if len(traj_points) < 2:
        return
    
    # Draw main trajectory line
    for i in range(1, len(traj_points)):
        pt1 = traj_points[i-1]
        pt2 = traj_points[i]
        
        # Ensure points are valid screen coordinates
        if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
            0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
            
            # Draw with increasing thickness for perspective effect
            progress = i / len(traj_points)
            current_thickness = max(1, int(thickness * (1.0 - progress * 0.5)))
            
            # Draw line segment
            cv2.line(image, pt1, pt2, color, current_thickness)
    
    # Draw landing point
    if draw_landing and len(traj_points) >= 3:
        landing_point = traj_points[-1]
        
        # Ensure point is on screen
        if (0 <= landing_point[0] < image.shape[1] and 
            0 <= landing_point[1] < image.shape[0]):
            
            # Draw landing marker (circle with cross)
            cv2.circle(image, landing_point, 7, landing_color, 2)
            cv2.line(image, 
                     (landing_point[0] - 5, landing_point[1]),
                     (landing_point[0] + 5, landing_point[1]),
                     landing_color, 2)
            cv2.line(image, 
                     (landing_point[0], landing_point[1] - 5),
                     (landing_point[0], landing_point[1] + 5),
                     landing_color, 2)


def add_carry_distance_marker(image, landing_point, distance, 
                              font=cv2.FONT_HERSHEY_SIMPLEX, 
                              color=(255, 255, 0)):
    """
    Add distance marker at landing point.
    
    Args:
        image: Image to draw on
        landing_point: (x, y) landing position
        distance: Distance in meters
        font: Font to use
        color: Text color
    """
    if not (0 <= landing_point[0] < image.shape[1] and 
            0 <= landing_point[1] < image.shape[0]):
        return
    
    # Draw distance text
    text = f"{distance:.1f}m"
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, 0.6, 2)
    
    # Draw box background
    box_margin = 5
    box_x1 = landing_point[0] - text_width//2 - box_margin
    box_y1 = landing_point[1] - text_height - box_margin - 15
    box_x2 = landing_point[0] + text_width//2 + box_margin
    box_y2 = landing_point[1] - box_margin - 5
    
    # Ensure box is on screen
    box_x1 = max(0, min(box_x1, image.shape[1] - 1))
    box_y1 = max(0, min(box_y1, image.shape[0] - 1))
    box_x2 = max(0, min(box_x2, image.shape[1] - 1))
    box_y2 = max(0, min(box_y2, image.shape[0] - 1))
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    
    # Draw text
    text_x = landing_point[0] - text_width//2
    text_y = landing_point[1] - 15
    cv2.putText(image, text, (text_x, text_y), font, 0.6, color, 2)


def plot_3d_trajectory(trajectory, title="Golf Ball 3D Trajectory"):
    """
    Create a 3D plot of the ball trajectory.
    
    Args:
        trajectory: List of 3D points [(x, y, z), ...]
        title: Plot title
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = [p[0] for p in trajectory]
    y = [p[1] for p in trajectory]
    z = [p[2] for p in trajectory]
    
    # Plot trajectory
    ax.plot(x, y, z, 'b-', linewidth=2)
    
    # Add start and end points
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Launch')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='Landing')
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(0, max_range)
    
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_launch_parameters(club_speed, ball_speed, smash_factor, vert_angle, horiz_angle, carry_distance):
    """
    Create a visualization of the launch parameters with club speed and smash factor.
    
    Args:
        club_speed: Club head speed in m/s
        ball_speed: Ball speed in m/s
        smash_factor: Ratio of ball speed to club speed
        vert_angle: Vertical launch angle in degrees
        horiz_angle: Horizontal launch angle in degrees
        carry_distance: Carry distance in meters
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First plot: Speeds and angles
    params1 = ['Club Speed', 'Ball Speed', 'Vertical Angle', 'Horizontal Angle']
    values1 = [club_speed, ball_speed, vert_angle, horiz_angle]
    units1 = ['m/s', 'm/s', 'degrees', 'degrees']
    alt_values1 = [club_speed * 2.237, ball_speed * 2.237, vert_angle, horiz_angle]
    alt_units1 = ['mph', 'mph', 'degrees', 'degrees']
    colors1 = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    positions1 = range(len(params1))
    for i, (param, value, unit, alt_value, alt_unit, color) in enumerate(
            zip(params1, values1, units1, alt_values1, alt_units1, colors1)):
        ax1.bar(i, value, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.text(i, value + max(values1)*0.03, f"{value:.1f} {unit}\n({alt_value:.1f} {alt_unit})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(positions1)
    ax1.set_xticklabels(params1, fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(values1) * 1.2)
    ax1.set_title('Speed and Direction Parameters', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Second plot: Smash factor and carry
    params2 = ['Smash Factor', 'Carry Distance']
    values2 = [smash_factor, carry_distance]
    units2 = ['ratio', 'meters']
    alt_values2 = [smash_factor, carry_distance / METERS_PER_YARD]
    alt_units2 = ['ratio', 'yards']
    colors2 = ['#9b59b6', '#1abc9c']
    
    positions2 = range(len(params2))
    for i, (param, value, unit, alt_value, alt_unit, color) in enumerate(
            zip(params2, values2, units2, alt_values2, alt_units2, colors2)):
        ax2.bar(i, value, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        if i == 0:  # Special formatting for smash factor
            ax2.text(i, value + max(values2)*0.03, f"{value:.2f} {unit}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax2.text(i, value + max(values2)*0.03, f"{value:.1f} {unit}\n({alt_value:.1f} {alt_unit})", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(positions2)
    ax2.set_xticklabels(params2, fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(values2) * 1.2)
    ax2.set_title('Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle('Golf Shot Analysis Results', fontsize=20, fontweight='bold', y=1.05)
    plt.show()


###############################################################################
# MAIN FRAME PROCESSING FUNCTION
###############################################################################

def process_frames(front_frame, side_frame, frame_idx, ball_tracker, club_tracker, 
                  swing_detector, calibration, in_flight, impact_detected, fps,
                  front_bg=None, prev_front=None, prev_side=None):
    """
    Process a pair of frames from front and side views.
    
    Args:
        front_frame: Frame from front (behind golfer) camera
        side_frame: Frame from side camera
        frame_idx: Current frame index
        ball_tracker: DualViewBallTracker instance
        club_tracker: DualViewEnhancedClubTracker instance
        swing_detector: SwingStateDetector instance
        calibration: GolfCameraCalibration instance
        in_flight: Whether ball is in flight
        impact_detected: Whether impact has been detected
        fps: Frame rate
        front_bg: Background frame for front view
        prev_front: Previous front frame
        prev_side: Previous side frame
        
    Returns:
        tuple: (combined_frame, ball_detected)
    """
    global launch_speed, launch_angle_vertical, launch_angle_horizontal, predicted_carry, club_speed
    
    #----------------------------------------------------------------------
    # 1. INITIALIZATION AND FRAME SETUP
    #----------------------------------------------------------------------
    
    # Create standard-sized output frames for processing
    standard_width = 640
    standard_height = 480
    
    # Initialize club and ball position variables
    front_club_pos = None
    side_club_pos = None
    front_ball_pos = None
    side_ball_pos = None
    
    # Initialize detection confidence values
    front_ball_conf = 0.0
    side_ball_conf = 0.0
    front_club_conf = 0.0
    side_club_conf = 0.0
    
    # Initialize flag for ball detection
    ball_detected = False
    
    # Create copies of frames for annotation with standard size
    if front_frame is not None:
        # Get original frame dimensions
        h_front, w_front = front_frame.shape[:2]
        front_resized = cv2.resize(front_frame, (standard_width, standard_height))
        front_annotated = front_resized.copy()
    else:
        front_annotated = np.zeros((standard_height, standard_width, 3), dtype=np.uint8)
        w_front, h_front = standard_width, standard_height
        
    if side_frame is not None:
        # Get original frame dimensions
        h_side, w_side = side_frame.shape[:2]
        side_resized = cv2.resize(side_frame, (standard_width, standard_height))
        side_annotated = side_resized.copy()
    else:
        side_annotated = np.zeros((standard_height, standard_width, 3), dtype=np.uint8)
        w_side, h_side = standard_width, standard_height
    
    #----------------------------------------------------------------------
    # 2. PREPROCESS AND DETECT OBJECTS
    #----------------------------------------------------------------------
    
    # Preprocess frames at standard resolution
    front_pp = advanced_preprocess(front_resized) if front_frame is not None else None
    side_pp = advanced_preprocess(side_resized) if side_frame is not None else None
    
    # Detect objects at standard resolution
    front_ball_bbox, front_ball_conf, front_club_bbox, front_club_conf = detect_objects(front_pp)
    side_ball_bbox, side_ball_conf, side_club_bbox, side_club_conf = detect_objects(side_pp)
    
    #----------------------------------------------------------------------
    # 3. ENHANCED BALL DETECTION WITH MOTION
    #----------------------------------------------------------------------
    
    # Use enhanced ball detection based on research paper approach
    if prev_front is not None and front_bg is not None:
        # Resize previous frames to standard resolution
        prev_front_resized = cv2.resize(prev_front, (standard_width, standard_height))
        front_bg_resized = cv2.resize(front_bg, (standard_width, standard_height))
        
        front_ball_pos_motion, front_ball_conf_motion = detect_ball_with_motion(
            front_pp, prev_front_resized, front_bg_resized
        )
        
        # If motion-based detection found a ball
        if front_ball_pos_motion is not None:
            # Convert to bbox format [x1, y1, x2, y2] for consistency
            x, y = front_ball_pos_motion
            radius = 10  # Approximate radius for golf ball
            front_ball_motion_bbox = [x-radius, y-radius, x+radius, y+radius]
            
            # Use motion-based detection if confidence is higher or YOLO didn't find anything
            if front_ball_conf_motion > front_ball_conf or front_ball_bbox is None:
                front_ball_bbox = front_ball_motion_bbox
                front_ball_conf = max(front_ball_conf, front_ball_conf_motion)
    
    if prev_side is not None and front_bg is not None:
        # Resize previous frames to standard resolution
        prev_side_resized = cv2.resize(prev_side, (standard_width, standard_height))
        side_bg_resized = cv2.resize(front_bg, (standard_width, standard_height))  # Using front_bg as placeholder
        
        side_ball_pos_motion, side_ball_conf_motion = detect_ball_with_motion(
            side_pp, prev_side_resized, side_bg_resized
        )
        
        # If motion-based detection found a ball
        if side_ball_pos_motion is not None:
            # Convert to bbox format [x1, y1, x2, y2] for consistency
            x, y = side_ball_pos_motion
            radius = 10  # Approximate radius for golf ball
            side_ball_motion_bbox = [x-radius, y-radius, x+radius, y+radius]
            
            # Use motion-based detection if confidence is higher or YOLO didn't find anything
            if side_ball_conf_motion > side_ball_conf or side_ball_bbox is None:
                side_ball_bbox = side_ball_motion_bbox
                side_ball_conf = max(side_ball_conf, side_ball_conf_motion)
    
    #----------------------------------------------------------------------
    # 4. CLUB DETECTION AND VISUALIZATION
    #----------------------------------------------------------------------
    
    # Process club detections
    if front_club_bbox is not None:
        bx1, by1, bx2, by2 = front_club_bbox
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        front_club_pos = (cx, cy)
        cv2.rectangle(front_annotated, (bx1, by1), (bx2, by2), (0,0,255), 2)
        cv2.putText(front_annotated, "club", (bx1, by1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    if side_club_bbox is not None:
        bx1, by1, bx2, by2 = side_club_bbox
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        side_club_pos = (cx, cy)
        cv2.rectangle(side_annotated, (bx1, by1), (bx2, by2), (0,0,255), 2)
        cv2.putText(side_annotated, "club", (bx1, by1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    # Update club tracker with new positions
    club_tracker.update(frame_idx, 
                      front_club_pos, front_club_conf,
                      side_club_pos, side_club_conf)
    
    # Update swing state
    swing_state, state_confidence = swing_detector.update(club_tracker, ball_tracker, frame_idx)
    
    # Get filtered club paths
    front_club_path = club_tracker.front_tracker.get_path()
    side_club_path = club_tracker.side_tracker.get_path()
    
    # Always calculate club speed for display and metrics estimation
    if club_speed == 0 and len(club_tracker.positions_3d) >= 3:
        new_club_speed = club_tracker.get_club_speed_3d(fps)
        if new_club_speed > 5.0:  # Only update if reasonable value
            club_speed = new_club_speed
            
            # Pre-calculate estimated values based on club speed for early display
            ball_speed_est = club_speed * SMASH_FACTOR
            carry_est = calculate_carry_from_ball_speed(ball_speed_est)
            
            # Set global values if they haven't been set yet
            if launch_speed == 0:
                launch_speed = ball_speed_est
            if predicted_carry == 0:
                predicted_carry = carry_est
            if launch_angle_vertical == 0:
                launch_angle_vertical = 12.0  # Typical default launch angle
    
    # Draw club paths with thicker lines and gradient coloring
    # Front view - draw complete club path with increased thickness
    for i in range(1, len(front_club_path)):
        pt1 = front_club_path[i-1]
        pt2 = front_club_path[i]
        # Use fading color for temporal effect - more recent is brighter
        alpha = min(1.0, i / len(front_club_path))
        # Gradient from blue (oldest) to bright magenta (newest)
        color = (int(200 * alpha), 0, int(100 + 155 * alpha))
        # Increase thickness for all segments, with even thicker lines for recent path
        thickness = 2 if i < len(front_club_path) - 10 else 3
        cv2.line(front_annotated, pt1, pt2, color, thickness)
    
    # Side view - draw complete club path with increased thickness
    for i in range(1, len(side_club_path)):
        pt1 = side_club_path[i-1]
        pt2 = side_club_path[i]
        alpha = min(1.0, i / len(side_club_path))
        # Use same color scheme as front view
        color = (int(200 * alpha), 0, int(100 + 155 * alpha))
        # Increase thickness for all segments, with even thicker lines for recent path
        thickness = 2 if i < len(side_club_path) - 10 else 3
        cv2.line(side_annotated, pt1, pt2, color, thickness)
    
    #----------------------------------------------------------------------
    # 5. BALL DETECTION AND VISUALIZATION
    #----------------------------------------------------------------------
    
    # Process ball detections
    if front_ball_bbox is not None:
        bx1, by1, bx2, by2 = front_ball_bbox
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        front_ball_pos = (cx, cy)
        cv2.rectangle(front_annotated, (bx1, by1), (bx2, by2), (0,255,0), 2)
        status = f"ball ({front_ball_conf:.2f})"
        cv2.putText(front_annotated, status, (bx1, by1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        ball_detected = True
    
    if side_ball_bbox is not None:
        bx1, by1, bx2, by2 = side_ball_bbox
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        side_ball_pos = (cx, cy)
        cv2.rectangle(side_annotated, (bx1, by1), (bx2, by2), (0,255,0), 2)
        status = f"ball ({side_ball_conf:.2f})"
        cv2.putText(side_annotated, status, (bx1, by1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        ball_detected = True
    
    # For in-flight ball, use ROI-based detection with prediction
    if in_flight and not ball_detected:
        # Use 3D prediction for next ball position
        pred_3d = ball_tracker.predict_next_3d(frame_idx)
        
        if pred_3d is not None:
            # Project 3D prediction to both views
            front_pred = calibration.project_3d_to_front(pred_3d)
            side_pred = calibration.project_3d_to_side(pred_3d)
            
            # Convert to integers
            front_pred = (int(front_pred[0]), int(front_pred[1]))
            side_pred = (int(side_pred[0]), int(side_pred[1]))
            
            # ROI margins
            front_margin = ball_tracker.front_tracker.dynamic_margin
            side_margin = ball_tracker.side_tracker.dynamic_margin
            
            # Try ROI detection in front view
            front_roi_bbox, front_roi_conf = detect_object_in_roi(
                front_pp, front_pred, front_margin, BALL_CLASS_ID
            )
            
            if front_roi_bbox is not None:
                bx1, by1, bx2, by2 = front_roi_bbox
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                front_ball_pos = (cx, cy)
                cv2.rectangle(front_annotated, (bx1, by1), (bx2, by2), (0,255,0), 2)
                cv2.putText(front_annotated, f"ball (ROI: {front_roi_conf:.2f})", (bx1, by1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # Use prediction in front view
                cv2.circle(front_annotated, front_pred, 5, (0,255,255), -1)
                cv2.putText(front_annotated, "ball (pred)", (front_pred[0]-30, front_pred[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                front_ball_pos = front_pred
            
            # Try ROI detection in side view
            side_roi_bbox, side_roi_conf = detect_object_in_roi(
                side_pp, side_pred, side_margin, BALL_CLASS_ID
            )
            
            if side_roi_bbox is not None:
                bx1, by1, bx2, by2 = side_roi_bbox
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                side_ball_pos = (cx, cy)
                cv2.rectangle(side_annotated, (bx1, by1), (bx2, by2), (0,255,0), 2)
                cv2.putText(side_annotated, f"ball (ROI: {side_roi_conf:.2f})", (bx1, by1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # Use prediction in side view
                cv2.circle(side_annotated, side_pred, 5, (0,255,255), -1)
                cv2.putText(side_annotated, "ball (pred)", (side_pred[0]-30, side_pred[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                side_ball_pos = side_pred
    
    # Update ball tracker with new positions
    if front_ball_pos is not None or side_ball_pos is not None:
        added_3d = ball_tracker.push_position(
            frame_idx, front_ball_pos, side_ball_pos, 
            front_ball_conf if front_ball_pos is not None else 0.0,
            side_ball_conf if side_ball_pos is not None else 0.0
        )
        
        if added_3d:
            ball_detected = True
    
    #----------------------------------------------------------------------
    # 6. TRAJECTORY CALCULATION AND VISUALIZATION
    #----------------------------------------------------------------------
    
    # Get 3D positions for display
    ball_positions_3d = ball_tracker.get_3d_positions()
    
    # Calculate trajectory using polynomial method
    if len(ball_positions_3d) >= 5 and (impact_detected or in_flight):
        # Apply trajectory calculation method
        smooth_trajectory, calculated_carry, _ = calculate_trajectory_polynomial(
            ball_positions_3d, fps
        )
        
        if smooth_trajectory is not None and calculated_carry > 0:
            # Update launch parameters
            ls, la_v, la_h = calculate_launch_parameters_from_trajectory(ball_positions_3d, fps)
            
            # Only update if values are reasonable
            if ls > 10.0:  # Minimum reasonable ball speed
                launch_speed = ls
                launch_angle_vertical = la_v
                launch_angle_horizontal = la_h
                predicted_carry = calculated_carry
                
                logging.info(f"Updated launch parameters using trajectory: speed={launch_speed:.2f}m/s, "
                             f"vertical angle={launch_angle_vertical:.2f}°, "
                             f"horizontal angle={launch_angle_horizontal:.2f}°, "
                             f"carry={predicted_carry:.2f}m")
    
    # Draw actual 3D trajectory if available
    if len(ball_positions_3d) > 1:
        # Project 3D points back to both views
        front_traj = []
        side_traj = []
        
        for pos_3d in ball_positions_3d:
            # Project 3D point to front view
            front_point = calibration.project_3d_to_front(pos_3d)
            front_traj.append((int(front_point[0]), int(front_point[1])))
            
            # Project 3D point to side view
            side_point = calibration.project_3d_to_side(pos_3d)
            side_traj.append((int(side_point[0]), int(side_point[1])))
        
        # Draw 3D trajectory on both views with thicker lines
        for i in range(1, len(front_traj)):
            pt1 = front_traj[i-1]
            pt2 = front_traj[i]
            # Fade-in effect
            alpha = min(1.0, i / len(front_traj))
            color = (0, int(255 * alpha), 0)
            # Use thicker lines for trajectory
            cv2.line(front_annotated, pt1, pt2, color, 3)
        
        for i in range(1, len(side_traj)):
            pt1 = side_traj[i-1]
            pt2 = side_traj[i]
            # Fade-in effect
            alpha = min(1.0, i / len(side_traj))
            color = (0, int(255 * alpha), 0)
            # Use thicker lines for trajectory
            cv2.line(side_annotated, pt1, pt2, color, 3)
        
        # Add landing point markers for better visibility
        if len(front_traj) > 2:
            cv2.circle(front_annotated, front_traj[-1], 5, (0, 255, 255), -1)
        if len(side_traj) > 2:
            cv2.circle(side_annotated, side_traj[-1], 5, (0, 255, 255), -1)
    
    # If we don't have actual trajectory but have ball position and club speed,
    # draw a predicted trajectory for visual feedback
    elif ball_tracker.get_latest_3d_position() is not None and club_speed > 0:
        # Generate a simple simulated trajectory based on club speed
        start_pos = ball_tracker.get_latest_3d_position()
        
        # Simple simulated trajectory
        default_angle = 12  # degrees
        est_ball_speed = club_speed * SMASH_FACTOR
        
        # Calculate simple parabolic trajectory
        simulated_traj = []
        for t in np.linspace(0, 3, 30):  # 3 second flight, 30 points
            # Simple physics model
            x = start_pos[0] + est_ball_speed * math.cos(math.radians(default_angle)) * t
            y = start_pos[1]  # Assume straight
            z = start_pos[2] + est_ball_speed * math.sin(math.radians(default_angle)) * t - 0.5 * GRAVITY * t**2
            if z < 0:  # Stop at ground level
                break
            simulated_traj.append((x, y, z))
        
        # Project and draw simulated trajectory
        if simulated_traj:
            front_sim_traj = []
            side_sim_traj = []
            
            for pos_3d in simulated_traj:
                # Project 3D point to both views
                front_point = calibration.project_3d_to_front(pos_3d)
                front_sim_traj.append((int(front_point[0]), int(front_point[1])))
                
                side_point = calibration.project_3d_to_side(pos_3d)
                side_sim_traj.append((int(side_point[0]), int(side_point[1])))
            
            # Draw simulated trajectory as dotted line
            for i in range(1, len(front_sim_traj)):
                if i % 2 == 0:  # Skip every other point for dotted effect
                    continue
                pt1 = front_sim_traj[i-1]
                pt2 = front_sim_traj[i]
                # Use distinct color for simulated trajectory
                cv2.line(front_annotated, pt1, pt2, (100, 100, 255), 2)
            
            for i in range(1, len(side_sim_traj)):
                if i % 2 == 0:  # Skip every other point for dotted effect
                    continue
                pt1 = side_sim_traj[i-1]
                pt2 = side_sim_traj[i]
                # Use distinct color for simulated trajectory
                cv2.line(side_annotated, pt1, pt2, (100, 100, 255), 2)
            
            # Mark landing points
            if len(front_sim_traj) > 2:
                cv2.drawMarker(front_annotated, front_sim_traj[-1], (100, 100, 255), 
                              markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            if len(side_sim_traj) > 2:
                cv2.drawMarker(side_annotated, side_sim_traj[-1], (100, 100, 255), 
                              markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                
            # Add a note that this is predicted trajectory
            mid_idx = len(front_sim_traj)//2
            if 0 <= mid_idx < len(front_sim_traj):
                cv2.putText(front_annotated, "Predicted Flight", front_sim_traj[mid_idx], 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            
            mid_idx = len(side_sim_traj)//2
            if 0 <= mid_idx < len(side_sim_traj):
                cv2.putText(side_annotated, "Predicted Flight", side_sim_traj[mid_idx], 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
    
    #----------------------------------------------------------------------
    # 7. METRICS DISPLAY
    #----------------------------------------------------------------------
    
    # Create background boxes for metrics display
    metrics_height = 130
    metrics_width = 250
    
    # Draw background boxes for better readability
    draw_metrics_box(front_annotated, 10, 10, metrics_width, metrics_height)
    draw_metrics_box(side_annotated, 10, 10, metrics_width, metrics_height)
    
    # ALWAYS display metrics if we have club speed
    if club_speed > 0:
        # Format horizontal angle correctly
        horiz_angle_text = f"{launch_angle_horizontal:.1f}°"
        
        # Label showing if carry is estimated or measured
        status_text = "ESTIMATED" if not impact_detected else "MEASURED"
        
        # Front view metrics
        cv2.putText(front_annotated, f"Club Speed: {club_speed:.1f} m/s ({club_speed*2.237:.0f}mph)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(front_annotated, f"Ball Speed: {launch_speed:.1f} m/s ({launch_speed*2.237:.0f}mph)", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(front_annotated, f"Smash Factor: {SMASH_FACTOR:.2f}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(front_annotated, f"Horiz. Angle: {horiz_angle_text}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Make carry display more prominent - ALWAYS shown
        cv2.putText(front_annotated, f"CARRY ({status_text}): {predicted_carry:.1f}m ({predicted_carry/METERS_PER_YARD:.0f}yd)", (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
        
        # Side view metrics
        cv2.putText(side_annotated, f"Club Speed: {club_speed:.1f} m/s ({club_speed*2.237:.0f}mph)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(side_annotated, f"Ball Speed: {launch_speed:.1f} m/s ({launch_speed*2.237:.0f}mph)", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(side_annotated, f"Launch Angle: {launch_angle_vertical:.1f}°", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(side_annotated, f"Smash Factor: {SMASH_FACTOR:.2f}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Make carry display more prominent - ALWAYS shown
        cv2.putText(side_annotated, f"CARRY ({status_text}): {predicted_carry:.1f}m ({predicted_carry/METERS_PER_YARD:.0f}yd)", (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
    else:
        # Display placeholder with more informative message
        cv2.putText(front_annotated, "Calculating club speed and carry...", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(side_annotated, "Calculating club speed and carry...", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    #----------------------------------------------------------------------
    # 8. STATUS DISPLAYS
    #----------------------------------------------------------------------
    
    # Display tracking status on both views
    status_msg = "Tracking: " + ("Active" if ball_detected else f"Lost ({ball_tracker.missed_frames})")
    status_color = (0, 255, 0) if ball_detected else (255, 165, 0)
    cv2.putText(front_annotated, status_msg, (10, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(side_annotated, status_msg, (10, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Show impact status with better visibility
    if impact_detected:
        cv2.putText(front_annotated, "IMPACT DETECTED", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # Orange-yellow for visibility
        cv2.putText(side_annotated, "IMPACT DETECTED", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(front_annotated, "Waiting for impact...", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)  # Grey for less prominence
        cv2.putText(side_annotated, "Waiting for impact...", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Display swing state on both views
    if swing_state == "SETUP":
        setup_color = (0, 255, 255)  # Yellow
    else:
        setup_color = (128, 128, 0)  # Dark yellow/gold
    
    cv2.putText(front_annotated, swing_state, (10, 220),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, setup_color, 2)
    cv2.putText(side_annotated, swing_state, (10, 220),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, setup_color, 2)
    
    #----------------------------------------------------------------------
    # 9. COMBINED VIEW CREATION
    #----------------------------------------------------------------------
    
    # Create combined view with standard size
    combined_width = standard_width * 2  # 640 * 2 = 1280
    combined = np.zeros((standard_height, combined_width, 3), dtype=np.uint8)
    
    # Place annotated frames in combined view
    combined[:, :standard_width] = front_annotated
    combined[:, standard_width:] = side_annotated
    
    # Add separating line
    cv2.line(combined, (standard_width, 0), (standard_width, standard_height), (200, 200, 200), 2)
    
    # Add view labels
    cv2.putText(combined, "Behind View", (20, standard_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Side View", (standard_width + 20, standard_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined, ball_detected


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    """
    Main function that orchestrates the entire golf shot analysis process.
    Handles video loading, shot detection, trajectory visualization, and result output.
    """
    global launch_speed, launch_angle_vertical, launch_angle_horizontal, predicted_carry, club_speed
    
    #----------------------------------------------------------------------
    # 1. INITIALIZATION AND SETUP
    #----------------------------------------------------------------------
    
    # Start timing for performance measurement
    start_time = time.time()
    
    # Display welcome message
    print("="*80)
    print("FlightSight Pro Dual-Angle Golf Analysis")
    print("="*80)
    
    # Initialize with placeholder values - these will be replaced
    # with calculated values once impact is detected
    launch_speed = 0.0 
    launch_angle_vertical = 0.0
    launch_angle_horizontal = 0.0
    predicted_carry = 0.0
    club_speed = 0.0
    
    # Create debug directory
    if DEBUG_SAVE_FRAMES:
        os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
    
    logging.info("Starting FlightSight Pro Dual-Angle Golf Analysis")
    
    #----------------------------------------------------------------------
    # 2. LOAD MODEL AND CHECK VIDEOS
    #----------------------------------------------------------------------
    
    # Load YOLO model
    load_yolo_model()
    
    # Create camera calibration
    calibration = GolfCameraCalibration(
        CAMERA_MATRIX_FRONT, CAMERA_MATRIX_SIDE, CAMERA_OFFSET
    )
    
    # Check input videos
    if not os.path.isfile(FRONT_VIDEO_PATH):
        logging.error(f"Front view video not found: {FRONT_VIDEO_PATH}")
        return
    
    if not os.path.isfile(SIDE_VIDEO_PATH):
        logging.error(f"Side view video not found: {SIDE_VIDEO_PATH}")
        return
    
    #----------------------------------------------------------------------
    # 3. OPEN VIDEOS AND SETUP ANALYSIS
    #----------------------------------------------------------------------
    
    # Open video captures
    front_cap = cv2.VideoCapture(FRONT_VIDEO_PATH)
    side_cap = cv2.VideoCapture(SIDE_VIDEO_PATH)
    
    if not front_cap.isOpened():
        logging.error(f"Could not open front video: {FRONT_VIDEO_PATH}")
        return
    
    if not side_cap.isOpened():
        logging.error(f"Could not open side video: {SIDE_VIDEO_PATH}")
        front_cap.release()
        return
    
    # Get video properties
    front_fps = front_cap.get(cv2.CAP_PROP_FPS)
    side_fps = side_cap.get(cv2.CAP_PROP_FPS)
    
    # Use minimum FPS for synchronization
    fps = min(front_fps, side_fps)
    if fps <= 0:
        fps = TARGET_FPS
    
    # Setup output video with standard dimensions
    standard_height = 480
    standard_width = 640
    combined_width = standard_width * 2  # 1280 pixels wide (2 views side by side)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"flightsight_pro_analysis_{timestamp}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, standard_height))
    
    # Create trackers
    ball_tracker = DualViewBallTracker(calibration)
    club_tracker = DualViewEnhancedClubTracker(calibration)
    swing_detector = SwingStateDetector()
    
    # Initialize state variables
    in_flight = False
    impact_detected = False
    impact_frame = 0
    frame_idx = 0
    all_ball_positions_3d = []
    
    # Variables for frame differencing (research paper method)
    front_background = None
    side_background = None
    prev_front_frame = None
    prev_side_frame = None
    
    # Count total frames for progress reporting
    total_frames = int(min(front_cap.get(cv2.CAP_PROP_FRAME_COUNT), 
                          side_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    #----------------------------------------------------------------------
    # 4. INITIALIZE PROGRESS TRACKING
    #----------------------------------------------------------------------
    
    # Process videos
    print("Processing videos... This may take a few minutes. Progress will be reported every 30 frames.")
    
    #----------------------------------------------------------------------
    # 5. MAIN PROCESSING LOOP
    #----------------------------------------------------------------------
    
    while True:
        # Clear GPU cache periodically to avoid memory fragmentation
        if frame_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Read frames from both videos
        front_ret, front_frame = front_cap.read()
        side_ret, side_frame = side_cap.read()
        
        # Check if we've reached the end
        if not front_ret or not side_ret:
            logging.info(f"Reached end of videos at frame {frame_idx}")
            break
        
        # Store background frames for motion detection
        if frame_idx == 0:
            front_background = front_frame.copy()
            side_background = side_frame.copy()
        
        # Print progress
        if frame_idx % 30 == 0:
            percent_done = min(100, int(frame_idx / total_frames * 100))
            print(f"Processing: {percent_done}% complete ({frame_idx}/{total_frames} frames)")
            
            # Display memory usage on GPU if available
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"  GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        try:
            # Skip frames for faster processing if needed (but don't skip at the beginning)
            if PROCESS_INTERVAL > 1 and frame_idx % PROCESS_INTERVAL != 0 and frame_idx > 10:
                frame_idx += 1
                continue
            
            # Process frames (passing previous frames for motion detection)
            combined_frame, ball_detected = process_frames(
                front_frame, side_frame, frame_idx,
                ball_tracker, club_tracker, swing_detector, calibration,
                in_flight, impact_detected, fps,
                front_background, prev_front_frame, prev_side_frame
            )
            
            # Update previous frames
            prev_front_frame = front_frame.copy() if front_frame is not None else None
            prev_side_frame = side_frame.copy() if side_frame is not None else None
            
            # Save 3D positions for further analysis
            latest_3d = ball_tracker.get_latest_3d_position()
            if latest_3d is not None:
                all_ball_positions_3d.append(latest_3d)
            
            #----------------------------------------------------------------------
            # 6. IMPACT DETECTION (IMPROVED)
            #----------------------------------------------------------------------
            
            if not impact_detected:
                # Use wider frame window for slow motion
                if frame_idx > IMPACT_DETECTION_WINDOW:
                    # Calculate club head speed
                    club_speed_calculated = club_tracker.get_club_speed_3d(fps)
                    if club_speed_calculated > 0:
                        club_speed = max(club_speed, club_speed_calculated)
                    
                    # Multiple impact detection methods with slow motion sensitivity:
                    impact_conditions = []
                    
                    # 1. Lower club speed threshold for slow motion
                    if club_speed > MIN_CLUB_SPEED_THRESHOLD:
                        if (club_tracker.front_tracker.is_in_downswing() or 
                            club_tracker.side_tracker.is_in_downswing()):
                            impact_conditions.append("Club downswing detected")
                    
                    # 2. Lower confidence requirement for ball detection
                    front_ball_bbox, front_ball_conf, _, _ = detect_objects(front_frame)
                    side_ball_bbox, side_ball_conf, _, _ = detect_objects(side_frame)
                    
                    if (front_ball_bbox is not None and side_ball_bbox is not None and
                        front_ball_conf > 0.3 and side_ball_conf > 0.3):
                        impact_conditions.append("Ball detected in both views")
                    
                    # 3. Check for sustained swing state - more reliable in slow motion
                    state_duration_threshold = int(3 * SLOW_MOTION_FACTOR)
                    if (swing_detector.current_state in ["DOWNSWING"] and 
                        swing_detector.time_in_state > state_duration_threshold):
                        impact_conditions.append("Sustained downswing state")
                    
                    # 4. Detect even small ball movement in slow motion
                    if len(ball_tracker.positions_3d) >= 3:
                        first_pos = np.array(ball_tracker.positions_3d[0][1:4])
                        current_pos = np.array(ball_tracker.positions_3d[-1][1:4])
                        movement = np.linalg.norm(current_pos - first_pos)
                        
                        # Very sensitive threshold for slow motion
                        if movement > 0.01:
                            impact_conditions.append("Ball movement detected")
                    
                    # Impact detected if ANY condition met
                    if impact_conditions:
                        impact_detected = True
                        impact_frame = frame_idx
                        print(f"\nIMPACT DETECTED IN SLOW MOTION: {', '.join(impact_conditions)}")
                        print(f"Club speed (adjusted): {club_speed:.2f} m/s ({club_speed*2.237:.1f} mph)")
                        logging.info(f"Impact detected at frame {impact_frame}: {', '.join(impact_conditions)}")
                        
            #----------------------------------------------------------------------
            # 7. FLIGHT DETECTION AND PARAMETER CALCULATION
            #----------------------------------------------------------------------
            
            # Check for launch after impact
            if impact_detected and not in_flight:
                # Calculate club head speed if not already done
                if club_speed < 1.0:
                    club_speed = max(club_tracker.get_club_speed_3d(fps), 30.0)  # Use at least 30 m/s (~67mph)
                
                # Use club head speed to calculate ball speed and carry via smash factor
                ball_speed, carry_distance = estimate_carry_using_smash_factor(club_speed)
                
                # Update global parameters
                launch_speed = ball_speed
                launch_angle_vertical = 12.0  # Typical driver launch angle
                launch_angle_horizontal = -5.0  # Slight draw bias
                predicted_carry = carry_distance
                
                print(f"\nGolf Shot Parameters Calculated:")
                print(f"  Club Speed: {club_speed:.2f}m/s ({club_speed*2.237:.1f}mph)")
                print(f"  Ball Speed: {ball_speed:.2f}m/s ({ball_speed*2.237:.1f}mph)")
                print(f"  Smash Factor: {SMASH_FACTOR:.2f}")
                print(f"  Launch Angle: {launch_angle_vertical:.1f}° vertical, {launch_angle_horizontal:.1f}° horizontal")
                print(f"  Predicted Carry: {predicted_carry:.2f}m ({predicted_carry/METERS_PER_YARD:.1f}yd)")
                
                in_flight = True
            
            # Even if impact not detected yet, still update metrics for display
            # whenever we have a good club speed value
            if not impact_detected and club_speed > 0 and launch_speed == 0:
                # Pre-calculate estimated ball speed and carry
                ball_speed_est = club_speed * SMASH_FACTOR
                carry_est = calculate_carry_from_ball_speed(ball_speed_est)
                
                # Set initial values
                launch_speed = ball_speed_est
                launch_angle_vertical = 12.0  # Typical default
                launch_angle_horizontal = 0.0  # Neutral default
                predicted_carry = carry_est
                
                print(f"\nESTIMATED Shot Parameters:")
                print(f"  Club Speed: {club_speed:.2f}m/s ({club_speed*2.237:.1f}mph)")
                print(f"  Est. Ball Speed: {launch_speed:.2f}m/s ({launch_speed*2.237:.1f}mph)")
                print(f"  Est. Carry: {predicted_carry:.2f}m ({predicted_carry/METERS_PER_YARD:.1f}yd)")
            
            #----------------------------------------------------------------------
            # 8. VIDEO SAVING AND DISPLAY
            #----------------------------------------------------------------------
            
            # Write frame to output video
            out.write(combined_frame)
            
            # Save debug frame if enabled
            if DEBUG_SAVE_FRAMES and frame_idx % 10 == 0:
                debug_frame_path = os.path.join(DEBUG_FRAME_DIR, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(debug_frame_path, combined_frame)
            
            # If displaying frames is enabled
            if DISPLAY_FRAMES:
                cv2.imshow("FlightSight Pro Analysis", combined_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                    break
            
            frame_idx += 1
            
        except Exception as e:
            logging.error(f"Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to continue with next frame
            frame_idx += 1
            continue
    
    #----------------------------------------------------------------------
    # 9. CLEANUP AND RESULTS DISPLAY
    #----------------------------------------------------------------------
    
    # Clean up
    front_cap.release()
    side_cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Report processing statistics
    end_time = time.time()
    total_time = end_time - start_time
    processing_fps = frame_idx / max(1, total_time)
    print(f"\nProcessing complete: {frame_idx} frames in {total_time:.1f} seconds ({processing_fps:.1f} fps)")
    print(f"Output video saved to: {output_path}")
    
    # Save a final snapshot with all metrics
    if frame_idx > 0:
        snapshot_filename = f"final_analysis_{timestamp}.jpg"
        cv2.imwrite(snapshot_filename, combined_frame)
        print(f"Final analysis snapshot saved to {snapshot_filename}")
    
    #----------------------------------------------------------------------
    # 10. RESULTS SUMMARY AND VISUALIZATION
    #----------------------------------------------------------------------
    
    # Display final results as text
    print("\n" + "="*50)
    print("Golf Shot Analysis Results")
    print("="*50)
    print(f"Club Speed: {club_speed:.2f} m/s ({club_speed*2.237:.2f} mph)")
    print(f"Ball Speed: {launch_speed:.2f} m/s ({launch_speed*2.237:.2f} mph)")
    print(f"Smash Factor: {SMASH_FACTOR:.2f}")
    print(f"Vertical Launch Angle: {launch_angle_vertical:.2f}°")
    print(f"Horizontal Launch Angle: {launch_angle_horizontal:.2f}°")
    print(f"Estimated Carry Distance: {predicted_carry:.2f} m ({predicted_carry/METERS_PER_YARD:.2f} yards)")
    print("="*50)
    
    # Calculate and visualize trajectory if enough data is available
    if len(all_ball_positions_3d) >= 5:
        smooth_trajectory, final_carry, _ = calculate_trajectory_polynomial(
            all_ball_positions_3d, fps
        )
        
        if smooth_trajectory is not None:
            print(f"\nTrajectory calculation complete.")
            print(f"Final estimated carry: {final_carry:.2f}m ({final_carry/METERS_PER_YARD:.2f} yards)")
            
            # Plot 3D trajectory
            plot_3d_trajectory(smooth_trajectory, 
                             title=f"Golf Ball Trajectory - Carry: {final_carry:.1f}m ({final_carry/METERS_PER_YARD:.1f} yards)")
    
    # Even if we don't have actual trajectory data, we can still show metrics visualization
    plot_launch_parameters(
        club_speed,
        launch_speed,
        SMASH_FACTOR,
        launch_angle_vertical,
        launch_angle_horizontal,
        predicted_carry
    )
    
    return {
        'club_speed_ms': club_speed,
        'club_speed_mph': club_speed * 2.237,
        'ball_speed_ms': launch_speed,
        'ball_speed_mph': launch_speed * 2.237,
        'smash_factor': SMASH_FACTOR,
        'vertical_angle': launch_angle_vertical,
        'horizontal_angle': launch_angle_horizontal,
        'carry_distance_m': predicted_carry,
        'carry_distance_yards': predicted_carry/METERS_PER_YARD
    }
    

if __name__ == "__main__":
    main()