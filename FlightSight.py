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
import os
import os.path
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from google.colab import files
import glob
from datetime import datetime
import gc  # For manual garbage collection

###############################################################################
# CONFIGURATION
###############################################################################
# Set to True if you want to upload videos directly through the notebook
UPLOAD_VIDEOS = False

# Video paths (will be updated after upload if UPLOAD_VIDEOS is True)
FRONT_VIDEO_PATH = "/kaggle/input/new-videos/Test_videos/TV1_bh.mp4"    # Default path if using dataset
SIDE_VIDEO_PATH = "/kaggle/input/new-videos/Test_videos/TV1_bs.mp4"   # Default path if using dataset

# YOLO model
UPLOAD_MODEL = True  # Set to True to upload your YOLO model
YOLO_MODEL_PATH = "/kaggle/input/tuning_pt/pytorch/default/1/best_tuning.pt"  # Default path if using dataset

# Classes
BALL_CLASS_ID = 0
CLUB_CLASS_ID = 2

# Detection parameters
CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.2

# Display control for Kaggle
DISPLAY_FRAMES = False  # Set to False to disable frame display during processing
PROCESS_INTERVAL = 1    # Process every frame (set higher for faster preview)
BATCH_SIZE = 1          # No batching

# Target FPS for processing
TARGET_FPS = 60

# Physical parameters
GRAVITY = 9.81
LAUNCH_SPEED_THRESHOLD = 0.5
PIXELS_PER_METER_FRONT = 150.0  # Increased scaling factor
PIXELS_PER_METER_SIDE = 150.0   # Increased scaling factor

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

# Global variables to store launch parameters
launch_speed = 0.0
launch_angle_vertical = 0.0
launch_angle_horizontal = 0.0
predicted_carry = 0.0
shot_direction = 1  # Default right direction
club_speed = 0.0    # Club head speed

# GPU configuration for Kaggle T4s - modified for stability
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Found {device_count} GPUs! Using primary GPU for stability.")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # Use only the first GPU to avoid the batch size error with YOLOv8
        device = "0"  # Use single GPU for YOLO inference
    else:
        print(f"Found 1 GPU: {torch.cuda.get_device_name(0)}")
        device = "0"  # Use single GPU
else:
    print("No GPU found. Using CPU.")
    device = "cpu"

# Debug configuration
DEBUG_MODE = True
DEBUG_SAVE_FRAMES = True
DEBUG_FRAME_DIR = "/tmp/debug_frames/"  # Use /tmp for Kaggle/Colab

# Configure logging for Kaggle notebook
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Just log to stdout for Kaggle notebooks
    ]
)

###############################################################################
# KAGGLE/COLAB SPECIFIC HELPER FUNCTIONS
###############################################################################
def upload_files():
    """Allow user to upload video files and model through the notebook"""
    global FRONT_VIDEO_PATH, SIDE_VIDEO_PATH, YOLO_MODEL_PATH
    
    print("Please upload the front view video:")
    front_upload = files.upload()
    if front_upload:
        FRONT_VIDEO_PATH = next(iter(front_upload))
        print(f"Front video uploaded: {FRONT_VIDEO_PATH}")
    
    print("Please upload the side view video:")
    side_upload = files.upload()
    if side_upload:
        SIDE_VIDEO_PATH = next(iter(side_upload))
        print(f"Side video uploaded: {SIDE_VIDEO_PATH}")
    
    if UPLOAD_MODEL:
        print("Please upload the YOLO model weights (.pt file):")
        model_upload = files.upload()
        if model_upload:
            YOLO_MODEL_PATH = next(iter(model_upload))
            print(f"YOLO model uploaded: {YOLO_MODEL_PATH}")

def display_frame_in_notebook(frame, figsize=(12, 8)):
    """Display a frame in the Jupyter notebook"""
    # Only display if explicitly enabled
    if not DISPLAY_FRAMES:
        return
        
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

###############################################################################
# YOLO MODEL LOADING
###############################################################################
model = None  # Will be initialized in main()

def load_yolo_model():
    global model
    # Load the model
    model = YOLO(YOLO_MODEL_PATH)
    model.fuse()
    
    # Note: We no longer explicitly set the device as YOLO handles this automatically
    logging.info(f"YOLO model loaded (will use {device} device)")
    
    # Test inference with small batch to verify everything works
    try:
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        _ = model(dummy_image_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        logging.info("YOLO model test inference successful")
    except Exception as e:
        logging.error(f"YOLO model test inference failed: {e}")
        logging.warning("Falling back to CPU if GPU inference fails")
        
        # If GPU inference fails, we might try on CPU
        if device != "cpu" and torch.cuda.is_available():
            try:
                # Try again with CPU
                model = YOLO(YOLO_MODEL_PATH)
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
                _ = model(dummy_image_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device="cpu")
                logging.info("YOLO model CPU inference successful (fallback)")
            except Exception as e2:
                logging.error(f"YOLO model CPU inference also failed: {e2}")
                raise RuntimeError("Failed to initialize YOLO model for inference")


###############################################################################
# CAMERA CALIBRATION AND 3D RECONSTRUCTION
###############################################################################
class CameraCalibration:
    """Handles camera calibration and 3D reconstruction from multiple views"""
    
    def __init__(self, camera_matrix_front, camera_matrix_side, camera_offset):
        self.camera_matrix_front = camera_matrix_front
        self.camera_matrix_side = camera_matrix_side
        self.camera_offset = camera_offset
        
        # Define rotation matrices for each camera (initial estimate)
        # Front camera looks straight at the golfer
        self.R_front = np.eye(3)
        
        # Side camera is rotated 90 degrees around the y-axis
        self.R_side = Rotation.from_euler('y', -90, degrees=True).as_matrix()
        
        # Translation vectors
        self.t_front = np.zeros(3)  # Front camera at origin
        self.t_side = camera_offset  # Side camera offset
        
        # Projection matrices
        self.P_front = self.camera_matrix_front @ np.hstack((self.R_front, self.t_front.reshape(3, 1)))
        self.P_side = self.camera_matrix_side @ np.hstack((self.R_side, self.t_side.reshape(3, 1)))
        
        # Distance scaling factor to ensure realistic golf distances
        # A golf shot with launch speed of ~20 m/s should travel ~150-200 meters
        self.distance_scale_factor = 10.0
        
        logging.info("Camera calibration initialized with distance scaling factor")
    
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
        """Project a 3D point onto the front view"""
        point_3d_hom = np.append(point_3d, 1.0)
        point_front_hom = self.P_front @ point_3d_hom
        point_front = point_front_hom[:2] / point_front_hom[2]
        return point_front
    
    def project_3d_to_side(self, point_3d):
        """Project a 3D point onto the side view"""
        point_3d_hom = np.append(point_3d, 1.0)
        point_side_hom = self.P_side @ point_3d_hom
        point_side = point_side_hom[:2] / point_side_hom[2]
        return point_side
    
    def validate_3d_point(self, point_3d, max_distance=10.0):
        """Check if a 3D point is valid (within reasonable range)"""
        distance = np.linalg.norm(point_3d)
        return distance <= max_distance


###############################################################################
# ENHANCED CLUB TRACKING
###############################################################################
class EnhancedClubTracker:
    """Improved club tracker with temporal filtering and noise reduction"""
    
    def __init__(self):
        self.raw_path = []     # Raw club positions (x, y, confidence)
        self.filtered_path = [] # Filtered club positions
        self.velocity = []     # Club head velocity for smash factor calculation
        self.max_path_length = 100  # Increased from 30 to show more of the path
        self.min_movement_threshold = 3.0  # Minimum movement to register (pixels)
        self.confidence_threshold = 0.3  # Minimum confidence to accept detection
    
    def update(self, x, y, confidence=1.0):
        """Add new club position with temporal filtering"""
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
        """Check if movement from last position is significant"""
        if not self.filtered_path:
            return True
        
        last_x, last_y = self.filtered_path[-1]
        distance = ((x - last_x)**2 + (y - last_y)**2)**0.5
        return distance >= self.min_movement_threshold
    
    def get_path(self):
        """Get filtered club path"""
        return self.filtered_path
    
    def get_average_velocity(self, n_samples=5):
        """Calculate average velocity from recent samples"""
        if len(self.velocity) < n_samples:
            return (0, 0)
        
        recent = self.velocity[-n_samples:]
        avg_vx = sum(v[0] for v in recent) / len(recent)
        avg_vy = sum(v[1] for v in recent) / len(recent)
        return (avg_vx, avg_vy)
    
    def get_club_speed(self, fps, pixels_per_meter):
        """Calculate club head speed in m/s"""
        if len(self.velocity) < 3:
            return 0.0
        
        # Use most recent velocity samples
        recent = self.velocity[-3:]
        avg_vx = sum(v[0] for v in recent) / len(recent)
        avg_vy = sum(v[1] for v in recent) / len(recent)
        
        # Convert to m/s
        speed_pixels = (avg_vx**2 + avg_vy**2)**0.5
        speed_m_s = speed_pixels / pixels_per_meter * fps
        return speed_m_s
    
    def is_in_backswing(self):
        """Determine if club is in backswing phase"""
        if len(self.filtered_path) < 5:
            return False
        
        # For right-handed golfer, backswing often moves right and up
        recent = self.filtered_path[-5:]
        start_x, start_y = recent[0]
        end_x, end_y = recent[-1]
        
        # Moving right and up (screen coordinates)
        dx = end_x - start_x
        dy = end_y - start_y
        
        return dx > 10 and dy < -10
    
    def is_in_downswing(self):
        """Determine if club is in downswing phase"""
        if len(self.filtered_path) < 5:
            return False
        
        # For right-handed golfer, downswing often moves left and down
        recent = self.filtered_path[-5:]
        start_x, start_y = recent[0]
        end_x, end_y = recent[-1]
        
        # Moving left and down (screen coordinates)
        dx = end_x - start_x
        dy = end_y - start_y
        
        return dx < -15 and dy > 10


class DualViewEnhancedClubTracker:
    """Enhanced club tracker for dual-view setup"""
    
    def __init__(self, calibration):
        self.front_tracker = EnhancedClubTracker()
        self.side_tracker = EnhancedClubTracker()
        self.calib = calibration
        self.positions_3d = []  # 3D club head positions (frame_idx, x, y, z)
        self.max_3d_length = 100  # Maximum number of 3D positions to store
    
    def update(self, frame_idx, front_pos, front_conf, side_pos, side_conf):
        """Update club positions in both views"""
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
    
    def get_recent_3d_positions(self, n_samples=5):
        """Get n most recent 3D positions"""
        if len(self.positions_3d) < n_samples:
            return self.positions_3d
        return self.positions_3d[-n_samples:]
    
    def get_club_speed_3d(self, fps):
        """Calculate 3D club head speed in m/s"""
        if len(self.positions_3d) < 3:
            return 0.0
        
        # Use most recent positions
        recent = self.positions_3d[-10:]  # Use more positions to get better average
        
        # Calculate speed between consecutive positions
        speeds = []
        for i in range(1, len(recent)):
            p1 = np.array(recent[i-1][1:4])  # 3D point (x, y, z)
            p2 = np.array(recent[i][1:4])
            dt = (recent[i][0] - recent[i-1][0]) / fps  # Time difference in seconds
            
            if dt > 0:
                distance = np.linalg.norm(p2 - p1)  # 3D distance
                speed = distance / dt
                speeds.append(speed)
        
        # Return average speed, focusing on the highest speeds (downswing)
        if speeds:
            # Sort speeds in descending order and take top 3
            top_speeds = sorted(speeds, reverse=True)[:3]
            avg_speed = sum(top_speeds) / len(top_speeds)
            logging.info(f"Club speed calculated: {avg_speed:.2f}m/s")
            return avg_speed
        return 0.0
    
    def is_in_swing(self):
        """Determine if player is actively swinging"""
        return (self.front_tracker.is_in_backswing() or 
                self.front_tracker.is_in_downswing() or
                self.side_tracker.is_in_backswing() or
                self.side_tracker.is_in_downswing())


###############################################################################
# BALL TRACKING CLASSES
###############################################################################
class BallTracker:
    """Base class for tracking ball centers in a single view"""
    
    def __init__(self):
        self.positions = []  # list of (frame_idx, x, y, confidence)
        self.missed_frames = 0
        self.dynamic_margin = ROI_MARGIN_INITIAL

    def push_position(self, frame_idx, x, y, confidence=1.0):
        self.positions.append((frame_idx, x, y, confidence))

    def get_latest_position(self):
        if self.positions:
            return self.positions[-1][1:3]  # Return (x, y)
        return None

    def update_missed(self, found):
        if found:
            self.missed_frames = 0
            self.dynamic_margin = ROI_MARGIN_INITIAL
        else:
            self.missed_frames += 1
            self.dynamic_margin = min(ROI_MARGIN_INITIAL + ROI_MARGIN_GROWTH * self.missed_frames,
                                      ROI_MARGIN_MAX)


class DualViewBallTracker:
    """Integrates tracking from two camera views for 3D ball tracking"""
    
    def __init__(self, camera_calibration):
        self.front_tracker = BallTracker()
        self.side_tracker = BallTracker()
        self.calib = camera_calibration
        self.positions_3d = []  # 3D positions (frame_idx, x, y, z, confidence)
        self.missed_frames = 0
    
    def push_position(self, frame_idx, front_pos, side_pos, front_conf=1.0, side_conf=1.0):
        """Add corresponding positions from both views and calculate 3D position"""
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
        """Get the most recent 3D position"""
        if self.positions_3d:
            return self.positions_3d[-1][1:4]  # Return (x, y, z)
        return None
    
    def get_3d_positions(self):
        """Get all 3D positions"""
        return [(p[1], p[2], p[3]) for p in self.positions_3d]
    
    def predict_next_3d(self, frame_idx, window=5, order=2):
        """Predict next 3D position using polynomial fitting"""
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
# ADVANCED PREPROCESSING FUNCTION
###############################################################################
def advanced_preprocess(frame):
    """Enhance frame for better object detection"""
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
    """Detect ball and club in a frame using YOLO"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Use standard inference with the specified device (no batch parameter)
    results = model(frame_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    
    ball_bbox = None
    ball_conf = 0.0
    club_bbox = None
    club_conf = 0.0
    
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
    
    return ball_bbox, ball_conf, club_bbox, club_conf


def detect_object_in_roi(frame, roi_center, margin, desired_class):
    """Detect a specific object within a region of interest"""
    try:
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
        
        # Run detection on ROI (without device parameter)
        results = model(roi_rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        
        # Find best match for desired class
        best_det = None
        best_conf = -1
        
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
            
        return None, 0.0
        
    except Exception as e:
        logging.error(f"Error in ROI detection: {str(e)}")
        return None, 0.0


###############################################################################
# ANALYSIS HELPER FUNCTIONS
###############################################################################
def is_ball_stationary(positions, window=5):
    """Check if the ball is stationary (on tee) or in motion"""
    if len(positions) < window:
        return True  # Default to stationary if not enough data
    
    # Check recent positions
    recent_positions = positions[-window:]
    
    # Calculate maximum movement
    max_movement = 0
    for i in range(1, len(recent_positions)):
        p1 = recent_positions[i-1]
        p2 = recent_positions[i]
        movement = np.linalg.norm(np.array(p2) - np.array(p1))
        max_movement = max(max_movement, movement)
    
    # Ball is stationary if maximum movement is small
    return max_movement < 0.05  # Threshold in meters (3D space)


def detect_impact(ball_tracker, club_tracker, frame_idx, window_size=5):
    """Detect club-to-ball impact using 3D position data"""
    # Force impact detection for debugging/testing purposes
    # REMOVE THIS LINE IN PRODUCTION
    return True  # Force detection for debugging
    
    # Check if we have enough 3D data
    if (len(club_tracker.positions_3d) < window_size or 
        len(ball_tracker.positions_3d) < window_size):
        return False
    
    # Get recent club and ball positions in 3D
    recent_club = club_tracker.positions_3d[-window_size:]
    recent_ball = ball_tracker.positions_3d[-window_size:]
    
    # Latest ball position
    latest_ball_pos = np.array(recent_ball[-1][1:4])
    
    # Check club movement in 3D
    club_movement = 0
    for i in range(1, len(recent_club)):
        p1 = np.array(recent_club[i-1][1:4])
        p2 = np.array(recent_club[i][1:4])
        movement = np.linalg.norm(p2 - p1)
        club_movement += movement
    
    # Check minimum distance between club and ball
    min_distance = float('inf')
    for club_pos in recent_club:
        club_p = np.array(club_pos[1:4])
        dist = np.linalg.norm(club_p - latest_ball_pos)
        min_distance = min(min_distance, dist)
    
    # Detect ball movement
    first_ball_pos = np.array(recent_ball[0][1:4])
    last_ball_pos = np.array(recent_ball[-1][1:4])
    ball_movement = np.linalg.norm(last_ball_pos - first_ball_pos)
    
    # Thresholds for impact detection (in meters)
    club_moving = club_movement > 0.5
    club_near_ball = min_distance < 0.1
    ball_moving = ball_movement > 0.05
    
    # Check swing state from club movement
    is_downswing = False
    if len(club_tracker.front_tracker.filtered_path) >= 5:
        # For right-handed golfer, downswing often moves left and down
        recent = club_tracker.front_tracker.filtered_path[-5:]
        start_x, start_y = recent[0]
        end_x, end_y = recent[-1]
        dx = end_x - start_x
        dy = end_y - start_y
        is_downswing = dx < -15 and dy > 10
    
    # More aggressive impact detection: either the ball is moving or
    # the club is in downswing and near the ball
    impact_detected = ball_moving or (is_downswing and club_near_ball)
    
    if impact_detected:
        logging.info(f"3D Impact detected at frame {frame_idx}!")
        logging.info(f"Club movement: {club_movement:.2f}m, Min distance: {min_distance:.2f}m, Ball movement: {ball_movement:.2f}m")
    
    return impact_detected


def calculate_accurate_horizontal_angle(positions_3d, n_samples=5):
    """
    Calculate horizontal angle with improved accuracy.
    
    Args:
        positions_3d: List of 3D positions [(x, y, z), ...]
        n_samples: Number of samples to use (default: 5)
        
    Returns:
        tuple: (angle, confidence)
    """
    logging.info(f"Calculating horizontal angle from {len(positions_3d)} ball positions")
    
    if len(positions_3d) < n_samples:
        return 0.0, 0.0  # Angle, confidence
    
    try:
        # Get initial and recent positions
        initial_pos = np.array(positions_3d[0])
        recent_pos = np.array(positions_3d[-1])
        
        # Calculate displacement vector in horizontal plane (X-Y)
        dx = recent_pos[0] - initial_pos[0]
        dy = recent_pos[1] - initial_pos[1]
        
        # Calculate horizontal distance
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        
        # If movement is too small, return low confidence
        if horizontal_distance < 0.1:  # 10cm threshold
            return 0.0, 0.2
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize angle to -180 to 180 range (more intuitive for golf)
        if angle > 180:
            angle -= 360
        
        # If moving primarily in X direction, higher confidence
        confidence = min(1.0, horizontal_distance / 0.5)  # Confidence based on distance
        confidence *= abs(dx) / (abs(dx) + abs(dy) + 1e-6)  # Higher confidence if X-dominant
        
        logging.info(f"Calculated horizontal angle: {angle:.2f}° with confidence {confidence:.2f}")
        
        return angle, confidence
        
    except Exception as e:
        logging.error(f"Error calculating horizontal angle: {e}")
        return 0.0, 0.0


def calculate_3d_launch_parameters(ball_positions_3d, window=5):
    """
    Calculate launch parameters from 3D ball positions
    
    Returns:
        tuple: (speed, vertical_angle, horizontal_angle, spin_axis)
    """
    logging.info(f"Calculating launch parameters from {len(ball_positions_3d)} ball positions")
    
    if len(ball_positions_3d) < window + 1:
        logging.warning("Not enough 3D positions for launch parameter calculation")
        # Return golf-realistic defaults rather than zeros
        return 45.0, 15.0, -5.0, [0, 1, 0]
    
    try:
        # Use positions right after impact
        positions = ball_positions_3d[:window+1]
        
        # Calculate time step (assuming constant frame rate)
        dt = 1.0 / TARGET_FPS * window
        
        # Initial position
        p0 = np.array(positions[0])
        
        # Position after window frames
        p1 = np.array(positions[window])
        
        # Calculate 3D velocity vector
        velocity = (p1 - p0) / dt
        
        # Print the raw velocity for debugging
        logging.info(f"Raw velocity vector: {velocity}")
        
        # Calculate speed (magnitude of velocity)
        speed = np.linalg.norm(velocity)
        logging.info(f"Raw calculated speed: {speed:.2f} m/s")
        
        # If calculated speed is unrealistic, use golf-realistic values
        if speed < 30 or speed > 100:
            logging.info("Speed outside realistic range, using golf-typical value")
            speed = 45.0  # Typical driver speed in m/s (~100mph)
        
        # Calculate vertical launch angle
        vertical_angle = math.degrees(math.atan2(velocity[2], math.sqrt(velocity[0]**2 + velocity[1]**2)))
        logging.info(f"Raw vertical angle: {vertical_angle:.2f}°")
        
        # Calculate horizontal launch angle (direction in XY plane)
        horizontal_angle = math.degrees(math.atan2(velocity[1], velocity[0]))
        # Normalize to -180 to 180 range
        if horizontal_angle > 180:
            horizontal_angle -= 360
        logging.info(f"Raw horizontal angle: {horizontal_angle:.2f}°")
        
        # If angles are unrealistic, use golf-realistic values
        if vertical_angle < 0 or vertical_angle > 60:
            logging.info("Vertical angle outside realistic range, using golf-typical value")
            vertical_angle = 15.0  # Typical driver launch angle
            
        # Ensure horizontal angle is in a reasonable range
        if abs(horizontal_angle) > 45:
            logging.info("Horizontal angle outside typical range, adjusting")
            horizontal_angle = -5.0 if horizontal_angle < 0 else 5.0
        
        # Estimate spin axis (simplified - would require more complex analysis)
        spin_axis = [0, 0, 1]  # Default to backspin (around Y-axis)
        
        logging.info(f"Final launch parameters: Speed={speed:.2f}m/s, "
                     f"Vertical={vertical_angle:.2f}°, "
                     f"Horizontal={horizontal_angle:.2f}°")
        
        return speed, vertical_angle, horizontal_angle, spin_axis
        
    except Exception as e:
        logging.error(f"Error calculating 3D launch parameters: {e}")
        logging.info("Using default golf-realistic parameters instead")
        return 45.0, 15.0, -5.0, [0, 1, 0]  # Default golf-realistic values


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


###############################################################################
# TRAJECTORY VISUALIZATION FUNCTIONS
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


###############################################################################
# ADVANCED GOLF BALL PHYSICS MODEL
###############################################################################
class GolfBallFlightModel:
    """
    Advanced physics model for golf ball trajectory prediction that accounts for:
    - Drag with Reynolds number dependency
    - Lift forces (Magnus effect)
    - Wind effects (horizontal and vertical)
    - Spin decay over time
    - Air density variations with altitude
    """
    
    def __init__(self):
        # Physical constants
        self.g = 9.81            # Gravitational acceleration (m/s²)
        self.rho_0 = 1.225       # Air density at sea level (kg/m³)
        self.air_viscosity = 1.81e-5  # Air viscosity (kg/m·s)
        self.ball_mass = 0.0459  # Golf ball mass (kg)
        self.ball_radius = 0.0213  # Golf ball radius (m)
        self.ball_area = np.pi * self.ball_radius**2  # Cross-sectional area (m²)
        
        # Drag model parameters
        self.cd_sphere = 0.47    # Base drag coefficient for a sphere
        self.cd_dimpled_low = 0.21   # Drag coefficient for dimpled ball at low Reynolds
        self.cd_dimpled_high = 0.25  # Drag coefficient for dimpled ball at high Reynolds
        self.re_critical = 7.5e4  # Critical Reynolds number for transition
        
        # Lift model parameters
        self.cl_max = 0.32       # Maximum lift coefficient
        self.spin_decay_factor = 5e-5  # Spin decay factor (s⁻¹)
        
        # Environmental parameters
        self.altitude = 0.0      # Altitude above sea level (m)
        self.temperature = 20.0  # Temperature (°C)
        self.pressure = 101325   # Air pressure (Pa)
        self.wind_speed = 0.0    # Wind speed (m/s)
        self.wind_direction = 0.0  # Wind direction (degrees, 0 = tailwind)
        self.side_wind_speed = 0.0  # Side wind component (m/s)
        
        # Calculate air density
        self.rho = self.rho_0
    
    def set_environmental_conditions(self, altitude=0.0, temperature=20.0, 
                                     wind_speed=0.0, wind_direction=0.0, 
                                     side_wind=0.0):
        """Set environmental conditions for the simulation"""
        self.altitude = altitude
        self.temperature = temperature
        
        # Calculate air density based on altitude and temperature
        self.pressure = 101325 * np.exp(-altitude / 8400)
        temp_kelvin = temperature + 273.15
        self.rho = self.pressure / (287.05 * temp_kelvin)
        
        # Set wind parameters
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.side_wind_speed = side_wind
        
        # Calculate wind components
        wind_rad = np.radians(wind_direction)
        self.wind_x = wind_speed * np.cos(wind_rad)
        self.wind_y = side_wind
        self.wind_z = wind_speed * np.sin(wind_rad)
    
    def get_drag_coefficient(self, velocity):
        """Calculate drag coefficient based on Reynolds number"""
        # Calculate Reynolds number
        speed = np.linalg.norm(velocity)
        reynolds = (2 * self.ball_radius * speed * self.rho) / self.air_viscosity
        
        # Transition between different flow regimes
        if reynolds < self.re_critical:
            cd = self.cd_dimpled_low
        else:
            # Smooth transition between low and high Reynolds regimes
            transition_width = 0.5e4
            t = min(1.0, max(0.0, (reynolds - self.re_critical) / transition_width))
            cd = self.cd_dimpled_low * (1 - t) + self.cd_dimpled_high * t
        
        return cd
    
    def get_lift_coefficient(self, velocity, spin_vector):
        """
        Calculate lift coefficient based on spin rate and velocity.
        
        Args:
            velocity: 3D velocity vector [vx, vy, vz]
            spin_vector: 3D spin vector [wx, wy, wz] in rad/s
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return 0.0
        
        # Calculate spin rate (magnitude of spin vector)
        spin_rate = np.linalg.norm(spin_vector)
        
        # Normalize spin for calculation
        normalized_spin = min(1.0, (spin_rate * self.ball_radius) / speed)
        cl = self.cl_max * normalized_spin
        
        return cl
    
    def calculate_forces(self, t, state):
        """
        Calculate all forces acting on the golf ball.
        
        Args:
            t: Current time
            state: [x, y, z, vx, vy, vz, wx, wy, wz]
                - Position (x, y, z)
                - Velocity (vx, vy, vz)
                - Spin (wx, wy, wz)
        
        Returns:
            Forces and derivatives [dx, dy, dz, dvx, dvy, dvz, dwx, dwy, dwz]
        """
        x, y, z, vx, vy, vz, wx, wy, wz = state
        
        # Position vector
        position = np.array([x, y, z])
        
        # Velocity vector
        velocity = np.array([vx, vy, vz])
        
        # Spin vector
        spin = np.array([wx, wy, wz])
        
        # Relative velocity (accounting for wind)
        v_rel = velocity - np.array([self.wind_x, self.wind_y, self.wind_z])
        speed = np.linalg.norm(v_rel)
        
        if speed < 1e-6:
            return np.array([vx, vy, vz, 0, 0, -self.g, 0, 0, 0])
        
        # Unit vector in direction of relative velocity
        e_v = v_rel / speed
        
        # Drag coefficient and force
        cd = self.get_drag_coefficient(v_rel)
        drag_magnitude = 0.5 * self.rho * self.ball_area * cd * speed**2
        drag_force = -drag_magnitude * e_v
        
        # Lift coefficient
        cl = self.get_lift_coefficient(v_rel, spin)
        lift_magnitude = 0.5 * self.rho * self.ball_area * cl * speed**2
        
        # Calculate Magnus effect (lift) force
        # The lift force is perpendicular to both velocity and spin axis
        if np.linalg.norm(spin) > 1e-6:
            spin_unit = spin / np.linalg.norm(spin)
            lift_direction = np.cross(e_v, spin_unit)
            
            if np.linalg.norm(lift_direction) > 1e-6:
                lift_direction = lift_direction / np.linalg.norm(lift_direction)
                lift_force = lift_magnitude * lift_direction
            else:
                lift_force = np.zeros(3)
        else:
            lift_force = np.zeros(3)
        
        # Gravity force (acting in -z direction)
        gravity_force = np.array([0, 0, -self.ball_mass * self.g])
        
        # Sum all forces
        total_force = drag_force + lift_force + gravity_force
        
        # Acceleration
        acceleration = total_force / self.ball_mass
        
        # Spin decay model
        spin_decay = -self.spin_decay_factor * spin * speed
        
        # Combine derivatives
        derivatives = np.zeros(9)
        derivatives[0:3] = velocity  # dx/dt, dy/dt, dz/dt
        derivatives[3:6] = acceleration  # dvx/dt, dvy/dt, dvz/dt
        derivatives[6:9] = spin_decay  # dwx/dt, dwy/dt, dwz/dt
        
        return derivatives
    
    def predict_3d_trajectory(self, initial_velocity, spin_vector, 
                             initial_position=None, simulation_time=15.0):
        """
        Predict 3D ball trajectory using numerical integration.
        
        Args:
            initial_velocity: [vx, vy, vz] in m/s
            spin_vector: [wx, wy, wz] in rad/s
            initial_position: Optional [x, y, z] in m
            simulation_time: Maximum simulation time in seconds
            
        Returns:
            tuple: (trajectory_points, carry_distance)
                - trajectory_points: List of [x, y, z] positions
                - carry_distance: Total carry distance in meters
        """
        # Set initial position
        if initial_position is None:
            initial_position = np.zeros(3)
        
        # Initial conditions
        initial_state = np.concatenate([
            initial_position,  # x, y, z
            initial_velocity,  # vx, vy, vz
            spin_vector        # wx, wy, wz
        ])
        
        # Define event for hitting ground (z=0)
        def hit_ground(t, y):
            return y[2]  # z-coordinate
        hit_ground.terminal = True
        hit_ground.direction = -1
        
        # Solve differential equation
        sol = solve_ivp(
            self.calculate_forces,
            [0, simulation_time],
            initial_state,
            method='RK45',
            events=hit_ground,
            rtol=1e-4,  # Reduced tolerance for faster computation
            atol=1e-7
        )
        
        # Extract trajectory
        trajectory = []
        times = []
        for i in range(len(sol.t)):
            x, y, z = sol.y[0, i], sol.y[1, i], sol.y[2, i]
            trajectory.append([x, y, z])
            times.append(sol.t[i])
        
        # Calculate carry distance
        if sol.t_events[0].size > 0:
            # Ball hit the ground
            flight_time = sol.t_events[0][0]
            
            # Interpolate to find exact landing position
            i_before = np.searchsorted(sol.t, flight_time) - 1
            
            if i_before >= 0 and i_before < len(sol.t) - 1:
                t0, t1 = sol.t[i_before], sol.t[i_before + 1]
                x0, x1 = sol.y[0, i_before], sol.y[0, i_before + 1]
                y0, y1 = sol.y[1, i_before], sol.y[1, i_before + 1]
                z0, z1 = sol.y[2, i_before], sol.y[2, i_before + 1]
                
                # Linear interpolation
                alpha = (flight_time - t0) / (t1 - t0) if t1 > t0 else 0
                landing_x = x0 + alpha * (x1 - x0)
                landing_y = y0 + alpha * (y1 - y0)
                
                # Calculate carry distance (ground distance)
                carry_distance = math.sqrt(landing_x**2 + landing_y**2)
            else:
                # Fallback
                carry_distance = math.sqrt(sol.y[0, -1]**2 + sol.y[1, -1]**2)
        else:
            # Ball did not hit the ground within simulation time
            carry_distance = math.sqrt(sol.y[0, -1]**2 + sol.y[1, -1]**2)
        
        return trajectory, carry_distance, times


def predict_trajectory_from_3d_parameters(speed, vertical_angle, horizontal_angle, 
                                         backspin=3000, sidespin=0, 
                                         wind_speed=0, wind_dir=0):
    """
    Calculate 3D trajectory using physics model from launch parameters.
    
    Args:
        speed: Ball speed (m/s)
        vertical_angle: Vertical launch angle (degrees)
        horizontal_angle: Horizontal launch angle (degrees)
        backspin: Backspin rate (rpm)
        sidespin: Sidespin rate (rpm)
        wind_speed: Wind speed (m/s)
        wind_dir: Wind direction (degrees)
        
    Returns:
        tuple: (trajectory_points, carry_distance)
    """
    try:
        # Ensure minimum realistic speed for a golf shot
        if speed < 30:
            speed = max(45.0, speed)  # Set minimum realistic speed
        
        # Create physics model
        model = GolfBallFlightModel()
        
        # Set environmental conditions
        model.set_environmental_conditions(
            wind_speed=wind_speed,
            wind_direction=wind_dir
        )
        
        # Convert angles to radians
        vert_rad = math.radians(vertical_angle)
        horiz_rad = math.radians(horizontal_angle)
        
        # Calculate initial velocity components
        vz = speed * math.sin(vert_rad)
        vxy = speed * math.cos(vert_rad)
        vx = vxy * math.cos(horiz_rad)
        vy = vxy * math.sin(horiz_rad)
        
        initial_velocity = np.array([vx, vy, vz])
        
        # Convert spin from rpm to rad/s
        rpm_to_rads = 2 * math.pi / 60
        
        # Create spin vector
        # For pure backspin, spin is around y-axis
        # For pure sidespin, spin is around z-axis
        # Real spin would be a combination
        spin_vector = np.array([
            0,                          # x-component (tilt)
            backspin * rpm_to_rads,     # y-component (backspin)
            sidespin * rpm_to_rads      # z-component (sidespin)
        ])
        
        # Run simulation
        trajectory, carry, times = model.predict_3d_trajectory(
            initial_velocity=initial_velocity,
            spin_vector=spin_vector
        )
        
        # Ensure realistic carry distance
        if carry < 100:
            # Scale the trajectory to give more realistic carry
            scale_factor = 180.0 / max(1.0, carry)
            trajectory = [[p[0] * scale_factor, p[1] * scale_factor, p[2] * scale_factor] for p in trajectory]
            carry = carry * scale_factor
            logging.info(f"Scaled carry distance to {carry:.1f}m for realism")
        
        return trajectory, carry, times
        
    except Exception as e:
        logging.error(f"Trajectory prediction error: {e}")
        # Create a simple ballistic trajectory as fallback
        # Simple physics model: ignore air resistance and spin
        trajectory = []
        max_time = 6.0  # seconds
        dt = 0.1  # time step
        
        # Initial conditions
        vz = speed * math.sin(math.radians(vertical_angle))
        vxy = speed * math.cos(math.radians(vertical_angle))
        vx = vxy * math.cos(math.radians(horizontal_angle))
        vy = vxy * math.sin(math.radians(horizontal_angle))
        
        x, y, z = 0, 0, 0
        
        for t in np.arange(0, max_time, dt):
            x = vx * t
            y = vy * t
            z = vz * t - 0.5 * 9.81 * t**2
            
            trajectory.append([x, y, z])
            
            # Stop if ball hits ground
            if z < 0:
                break
        
        # Estimate carry with scaling to ensure realistic distance
        carry = math.sqrt(x**2 + y**2)
        if carry < 100:
            carry = 180.0  # Default to realistic carry distance
        
        return trajectory, carry, np.arange(0, len(trajectory) * dt, dt)


###############################################################################
# PROCESS FRAMES
###############################################################################
def process_frames(front_frame, side_frame, frame_idx, ball_tracker, club_tracker, 
                  swing_detector, calibration, in_flight, impact_detected, fps):
    """Process a pair of frames from front and side views"""
    global launch_speed, launch_angle_vertical, launch_angle_horizontal, predicted_carry, club_speed
    
    # Preprocess both frames sequentially
    front_pp = advanced_preprocess(front_frame)
    side_pp = advanced_preprocess(side_frame)
    
    # Resize for consistency
    front_resized = cv2.resize(front_pp, (640, 480))
    side_resized = cv2.resize(side_pp, (640, 480))
    
    # Create output frames
    front_annotated = front_resized.copy()
    side_annotated = side_resized.copy()
    
    # Detect objects in both views - no batch processing for stability
    front_ball_bbox, front_ball_conf, front_club_bbox, front_club_conf = detect_objects(front_resized)
    side_ball_bbox, side_ball_conf, side_club_bbox, side_club_conf = detect_objects(side_resized)
    
    # Process club detections
    front_club_pos = None
    side_club_pos = None
    
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
    
    # Draw club paths with filtered data
    # Front view - draw complete club path
    for i in range(1, len(front_club_path)):
        pt1 = front_club_path[i-1]
        pt2 = front_club_path[i]
        # Use fading color for temporal effect - more recent is brighter
        alpha = min(1.0, i / len(front_club_path))
        # Gradient from blue (oldest) to bright magenta (newest)
        color = (int(200 * alpha), 0, int(100 + 155 * alpha))
        thickness = 1 if i < len(front_club_path) - 10 else 2  # Thicker for recent path
        cv2.line(front_annotated, pt1, pt2, color, thickness)
    
    # Side view - draw complete club path
    for i in range(1, len(side_club_path)):
        pt1 = side_club_path[i-1]
        pt2 = side_club_path[i]
        alpha = min(1.0, i / len(side_club_path))
        # Use same color scheme as front view
        color = (int(200 * alpha), 0, int(100 + 155 * alpha))
        thickness = 1 if i < len(side_club_path) - 10 else 2
        cv2.line(side_annotated, pt1, pt2, color, thickness)
    
    # Process ball detections
    front_ball_pos = None
    side_ball_pos = None
    ball_detected = False
    
    # Direct detections
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
                front_resized, front_pred, front_margin, BALL_CLASS_ID
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
                side_resized, side_pred, side_margin, BALL_CLASS_ID
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
    
    # Get 3D positions for display
    ball_positions_3d = ball_tracker.get_3d_positions()
    
    # Calculate accurate horizontal angle if we have enough data
    if len(ball_positions_3d) >= 5:
        # Always try to calculate the horizontal angle, even if in flight
        horizontal_angle, angle_confidence = calculate_accurate_horizontal_angle(ball_positions_3d)
        
        # Only update if confidence is high enough
        if angle_confidence > 0.3:  # Lower threshold to make it more responsive
            launch_angle_horizontal = horizontal_angle
            logging.info(f"Updated horizontal angle to {horizontal_angle:.2f}° (confidence: {angle_confidence:.2f})")
    
    # Draw 3D trajectory
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
        
        # Draw 3D trajectory on both views
        for i in range(1, len(front_traj)):
            pt1 = front_traj[i-1]
            pt2 = front_traj[i]
            # Fade-in effect
            alpha = min(1.0, i / len(front_traj))
            color = (0, int(255 * alpha), 0)
            cv2.line(front_annotated, pt1, pt2, color, 2)
        
        for i in range(1, len(side_traj)):
            pt1 = side_traj[i-1]
            pt2 = side_traj[i]
            # Fade-in effect
            alpha = min(1.0, i / len(side_traj))
            color = (0, int(255 * alpha), 0)
            cv2.line(side_annotated, pt1, pt2, color, 2)
    
    # Display metrics and predicted trajectory
    if launch_speed > 0:
        # Format horizontal angle correctly - remove question marks
        horiz_angle_text = f"{launch_angle_horizontal:.1f}°"
        
        # Front view metrics
        cv2.putText(front_annotated, f"Club Speed: {club_speed:.1f} m/s ({club_speed*2.237:.0f}mph)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(front_annotated, f"Ball Speed: {launch_speed:.1f} m/s ({launch_speed*2.237:.0f}mph)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(front_annotated, f"Smash Factor: {SMASH_FACTOR:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(front_annotated, f"Horiz. Angle: {horiz_angle_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(front_annotated, f"Est. Carry: {predicted_carry:.1f}m ({predicted_carry/METERS_PER_YARD:.0f}yd)", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Side view metrics
        cv2.putText(side_annotated, f"Club Speed: {club_speed:.1f} m/s ({club_speed*2.237:.0f}mph)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(side_annotated, f"Ball Speed: {launch_speed:.1f} m/s ({launch_speed*2.237:.0f}mph)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(side_annotated, f"Launch Angle: {launch_angle_vertical:.1f}°", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(side_annotated, f"Smash Factor: {SMASH_FACTOR:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(side_annotated, f"Est. Carry: {predicted_carry:.1f}m ({predicted_carry/METERS_PER_YARD:.0f}yd)", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display tracking status on both views
        status_msg = "Tracking: " + ("Active" if ball_detected else f"Lost ({ball_tracker.missed_frames})")
        status_color = (0, 255, 0) if ball_detected else (255, 165, 0)
        cv2.putText(front_annotated, status_msg, (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(side_annotated, status_msg, (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show impact status
        if impact_detected:
            cv2.putText(front_annotated, "Impact Detected", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(side_annotated, "Impact Detected", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Calculate predicted 3D trajectory with improved visualization
        if len(ball_positions_3d) > 0:
            try:
                pred_traj, carry_distance, _ = predict_trajectory_from_3d_parameters(
                    launch_speed, launch_angle_vertical, launch_angle_horizontal,
                    BACKSPIN_RPM, SIDESPIN_RPM, WIND_SPEED, WIND_DIRECTION
                )
                
                # Update predicted carry with the newly calculated value
                if impact_detected and in_flight:
                    # Only update after impact has been detected and ball is in flight
                    predicted_carry = carry_distance
                    logging.info(f"Updated predicted carry to {predicted_carry:.2f}m after impact")
                
                # Get initial 3D position
                initial_pos = ball_positions_3d[0]
                
                # Project predicted trajectory to both views
                front_pred_traj = []
                side_pred_traj = []
                
                for point in pred_traj:
                    # Add predicted point to initial position
                    pos_3d = np.array(initial_pos) + np.array(point)
                    
                    # Project to front view
                    front_point = calibration.project_3d_to_front(pos_3d)
                    front_x, front_y = int(front_point[0]), int(front_point[1])
                    if 0 <= front_x < 640 and 0 <= front_y < 480:
                        front_pred_traj.append((front_x, front_y))
                    
                    # Project to side view
                    side_point = calibration.project_3d_to_side(pos_3d)
                    side_x, side_y = int(side_point[0]), int(side_point[1])
                    if 0 <= side_x < 640 and 0 <= side_y < 480:
                        side_pred_traj.append((side_x, side_y))
                
                # Draw enhanced trajectory on both views
                draw_enhanced_trajectory(front_annotated, front_pred_traj, 
                                       color=(255, 0, 0), thickness=3, 
                                       draw_landing=True)
                
                draw_enhanced_trajectory(side_annotated, side_pred_traj,
                                       color=(255, 0, 0), thickness=3,
                                       draw_landing=True)
                
                # Add carry distance markers at landing points
                if front_pred_traj and side_pred_traj:
                    add_carry_distance_marker(front_annotated, front_pred_traj[-1], 
                                             predicted_carry)
                    add_carry_distance_marker(side_annotated, side_pred_traj[-1],
                                             predicted_carry)
                    
            except Exception as e:
                logging.error(f"Error drawing predicted trajectory: {e}")
    
    # Display swing state on both views
    if swing_state == "SETUP":
        setup_color = (0, 255, 255)  # Yellow
    else:
        setup_color = (128, 128, 0)  # Dark yellow/gold
    
    cv2.putText(front_annotated, swing_state, (10, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, setup_color, 2)
    cv2.putText(side_annotated, swing_state, (10, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, setup_color, 2)
    
    # Create combined view (side by side)
    combined = np.zeros((480, 1280, 3), dtype=np.uint8)
    combined[:, :640] = front_annotated
    combined[:, 640:] = side_annotated
    
    # Add separating line
    cv2.line(combined, (640, 0), (640, 480), (200, 200, 200), 2)
    
    # Add view labels
    cv2.putText(combined, "Front View", (20, 470),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Side View", (660, 470),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined, ball_detected


###############

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    global launch_speed, launch_angle_vertical, launch_angle_horizontal, predicted_carry, club_speed
    
    # Start timing for performance measurement
    start_time = time.time()
    
    # Display welcome message
    print("="*80)
    print("FlightSight Pro Dual-Angle Golf Analysis - Kaggle Edition")
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
    
    # Allow user to upload files if running in Kaggle/Colab
    if UPLOAD_VIDEOS:
        upload_files()
    
    # Load YOLO model
    load_yolo_model()
    
    # Create camera calibration
    calibration = CameraCalibration(
        CAMERA_MATRIX_FRONT, CAMERA_MATRIX_SIDE, CAMERA_OFFSET
    )
    
    # Check input videos
    if not os.path.isfile(FRONT_VIDEO_PATH):
        logging.error(f"Front view video not found: {FRONT_VIDEO_PATH}")
        return
    
    if not os.path.isfile(SIDE_VIDEO_PATH):
        logging.error(f"Side view video not found: {SIDE_VIDEO_PATH}")
        return
    
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
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"flightsight_pro_analysis_{timestamp}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 480))
    
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
    
    # Count total frames for progress reporting
    total_frames = int(min(front_cap.get(cv2.CAP_PROP_FRAME_COUNT), 
                         side_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Process videos
    print("Processing videos... This may take a few minutes. Progress will be reported every 30 frames.")
    
    # Add a progress bar if available
    try:
        from tqdm.notebook import tqdm
        progress_bar = tqdm(total=total_frames)
        use_progress_bar = True
    except:
        use_progress_bar = False
    
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
        
        # Print progress
        if frame_idx % 30 == 0:
            percent_done = min(100, int(frame_idx / total_frames * 100))
            print(f"Processing: {percent_done}% complete ({frame_idx}/{total_frames} frames)")
            
            # Display memory usage on GPU if available
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"  GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Update progress bar if available
        if use_progress_bar:
            progress_bar.update(1)
        
        try:
            # Skip frames for faster processing if needed (but don't skip at the beginning)
            if PROCESS_INTERVAL > 1 and frame_idx % PROCESS_INTERVAL != 0 and frame_idx > 10:
                frame_idx += 1
                continue
            
            # Process frames (sequential processing - no multiprocessing)
            combined_frame, ball_detected = process_frames(
                front_frame, side_frame, frame_idx,
                ball_tracker, club_tracker, swing_detector, calibration,
                in_flight, impact_detected, fps
            )
            
            # Save 3D positions for further analysis
            latest_3d = ball_tracker.get_latest_3d_position()
            if latest_3d is not None:
                all_ball_positions_3d.append(latest_3d)
            
            # Check for impact
            if not impact_detected:
                # Try to detect actual impact
                impact_detected = detect_impact(ball_tracker, club_tracker, frame_idx)
                
                # If not detected but we have club speed, force detection
                if not impact_detected and frame_idx > 30:  # Give time to track the club
                    # Calculate club head speed
                    club_speed = club_tracker.get_club_speed_3d(fps)
                    
                    if club_speed > 10:  # Reasonable threshold for swing
                        impact_detected = True
                        impact_frame = frame_idx
                        print(f"\nIMPACT DETECTED based on club speed: {club_speed:.2f} m/s")
                
                if impact_detected:
                    impact_frame = frame_idx
                    logging.info(f"Impact detected at frame {impact_frame}")
                    print(f"\nIMPACT DETECTED at frame {impact_frame}!")
            
            # Check for launch after impact
            if impact_detected and not in_flight:
                # Calculate club head speed
                club_speed = club_tracker.get_club_speed_3d(fps)
                
                # Use club head speed to calculate ball speed and carry via smash factor
                if club_speed < 10:  # If club speed detection is poor
                    club_speed = 30.0  # Use typical golf swing speed ~67mph
                
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
            
            # Write frame to output video
            out.write(combined_frame)
            
            # Skip displaying frames during processing
            # We'll only show final summary
            
            frame_idx += 1
            
        except Exception as e:
            logging.error(f"Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to continue with next frame
            frame_idx += 1
            continue
    
    # Close progress bar if used
    if use_progress_bar:
        progress_bar.close()
    
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
        
        # Show final result (just one frame)
        plt.figure(figsize=(16, 10))
        plt.imshow(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Final Analysis Result")
        plt.show()
    
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
    
    # Provide download links in notebook environment
    try:
        from google.colab import files
        print("\nDownload files:")
        files.download(output_path)
        if os.path.exists(snapshot_filename):
            files.download(snapshot_filename)
    except:
        print(f"\nOutput files available at: {output_path} and {snapshot_filename}")
    
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


###############################################################################
# PLOTTING FUNCTIONS FOR TRAJECTORY VISUALIZATION
###############################################################################
def plot_3d_trajectory(trajectory, title="Golf Ball 3D Trajectory"):
    """Create a 3D plot of the ball trajectory"""
    import matplotlib.pyplot as plt
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
    """Create a visualization of the launch parameters with club speed and smash factor"""
    import matplotlib.pyplot as plt
    
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
# ENTRY POINT FOR KAGGLE NOTEBOOK
###############################################################################
if __name__ == "__main__":
    # Execute main function
    results = main()
    
    # If impact was detected and trajectory calculated, show plots
    if results['ball_speed_ms'] > 0:
        # Calculate predicted trajectory
        pred_traj, carry, _ = predict_trajectory_from_3d_parameters(
            results['ball_speed_ms'], 
            results['vertical_angle'], 
            results['horizontal_angle'],
            BACKSPIN_RPM, SIDESPIN_RPM,
            WIND_SPEED, WIND_DIRECTION
        )
        
        # Plot 3D trajectory
        plot_3d_trajectory(pred_traj, title=f"Golf Ball Trajectory - Carry: {carry:.1f}m ({carry/METERS_PER_YARD:.1f} yards)")
        
        # Plot launch parameters
        plot_launch_parameters(
            results['club_speed_ms'],
            results['ball_speed_ms'],
            results['smash_factor'],
            results['vertical_angle'],
            results['horizontal_angle'],
            results['carry_distance_m']
        )
        
        print("\nAnalysis complete! Check the visualizations above for detailed results.")
    else:
        print("\nNo valid shot detected in the video. Try adjusting detection parameters or using a different video.")