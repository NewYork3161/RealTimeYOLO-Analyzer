"""
camera.py
---------

Camera interface module for the YOLO object detection system.

Purpose
-------
This file isolates all camera-related behavior from the rest of the
application. The main program should not have to deal directly with
low-level webcam setup details.

What this module does
---------------------
1. Opens the webcam
2. Configures frame size
3. Captures frames
4. Releases the camera cleanly on exit

Why this matters
----------------
Separating camera logic into its own module improves:

- readability
- maintainability
- modularity
- scalability

This is a common software engineering pattern known as
"separation of concerns."
"""

import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, DEBUG


class Camera:
    """
    Camera class

    This class provides a clean abstraction over OpenCV's
    VideoCapture object.

    Instead of working with cv2.VideoCapture directly inside main.py,
    the rest of the application can simply ask this class for frames.
    """

    def __init__(self):
        """
        Constructor

        This runs automatically when we create a Camera object.

        It attempts to open the webcam and apply optional
        frame width and height settings.
        """

        # Create the webcam capture object
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        # Validate that the camera opened correctly
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        # Apply requested frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if DEBUG:
            print("Webcam initialized successfully.")
            print(f"Requested resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    def get_frame(self):
        """
        Capture and return one frame from the webcam.

        Returns
        -------
        frame : numpy.ndarray | None
            The captured image frame if successful.
            Returns None if frame capture fails.
        """

        ret, frame = self.cap.read()

        # ret is True if OpenCV successfully grabbed a frame
        if not ret:
            print("Warning: Failed to capture frame.")
            return None

        return frame

    def release(self):
        """
        Release the webcam resource.

        This is important because camera devices should always
        be freed when the application exits.
        """

        if self.cap.isOpened():
            self.cap.release()

        if DEBUG:
            print("Webcam released successfully.")