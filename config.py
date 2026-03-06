"""
config.py
---------

Central configuration file for the YOLO real-time object detection system.

Why this file exists
--------------------
Instead of hardcoding values throughout the application, we store
important settings in one place. This makes the system easier to:

1. Maintain
2. Debug
3. Tune for performance
4. Reuse in other projects

If we want to switch cameras, lower the confidence threshold,
change the model size, or turn debugging on/off, we only need
to modify this file.
"""

# ==========================================================
# CAMERA CONFIGURATION
# ==========================================================

# Index of the webcam device.
# 0 = default camera
# 1 = second camera
# 2 = third camera, etc.
CAMERA_INDEX = 0

# Optional frame dimensions.
# These can help standardize video input size.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

# YOLO model file to load.
# Smaller models are faster, larger models are more accurate.
#
# Common choices:
# yolov8n.pt = nano (fastest, smallest)
# yolov8s.pt = small
# yolov8m.pt = medium
# yolov8l.pt = large
# yolov8x.pt = extra large (slowest, most accurate)
MODEL_NAME = "yolov8m.pt"

# Minimum confidence score required before a detection is accepted.
# Example:
# If a box is predicted with confidence 0.25 and threshold is 0.40,
# that detection will be ignored.
CONFIDENCE_THRESHOLD = 0.4

# IOU threshold used during Non-Maximum Suppression (NMS).
# This helps remove overlapping duplicate boxes.
IOU_THRESHOLD = 0.5


# ==========================================================
# DISPLAY CONFIGURATION
# ==========================================================

# Window title shown by OpenCV
WINDOW_NAME = "YOLO Object Detection"

# Whether to display FPS information on screen
SHOW_FPS = True

# Text drawing size
FONT_SCALE = 0.7

# Thickness of box outlines
BOX_THICKNESS = 2


# ==========================================================
# PERFORMANCE / INFERENCE SETTINGS
# ==========================================================

# If True, the system may use GPU acceleration if available
USE_GPU = True

# Inference image size
# Larger values can improve accuracy but reduce speed.
IMAGE_SIZE = 640


# ==========================================================
# DEBUG SETTINGS
# ==========================================================

# If True, print debug information to terminal
DEBUG = True

# If True, print detected objects every frame
PRINT_DETECTIONS = False


# ==========================================================
# LOGGING SETTINGS
# ==========================================================

# Folder where logs are stored
LOG_DIR = "logs"

# CSV file where detection data is written
LOG_FILE = "logs/detection_log.csv"