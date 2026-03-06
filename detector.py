"""
detector.py
-----------

AI inference engine for the YOLO real-time object detection system.

Purpose
-------
This module isolates all machine learning functionality from the rest
of the application.

Responsibilities
----------------
This file is responsible for:

1. Loading the YOLO model
2. Running inference on frames
3. Extracting detection information
4. Rendering bounding boxes and labels

This file intentionally does NOT handle:
- webcam setup
- logging
- analytics
- chart generation
- overall application control

That design follows the Single Responsibility Principle (SRP),
which says each module should have one clear reason to change.
"""

import cv2
from ultralytics import YOLO
from config import (
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DEBUG,
    PRINT_DETECTIONS,
)


class ObjectDetector:
    """
    ObjectDetector

    This class acts as the AI engine of the system.

    It provides a clean interface for:
    - loading the model
    - running detection
    - counting detections
    - listing detected object names
    - drawing visual results
    """

    def __init__(self):
        """
        Load the pretrained YOLO model into memory.

        This may take a moment the first time the program starts.
        """

        if DEBUG:
            print("Loading YOLO model...")

        # Load pretrained YOLO model from local file
        self.model = YOLO(MODEL_NAME)

        # YOLO stores a mapping of class IDs to class names
        # Example:
        # 0 -> person
        # 1 -> bicycle
        # 2 -> car
        self.class_names = self.model.names

        if DEBUG:
            print("YOLO model loaded successfully.")
            print(f"Model supports {len(self.class_names)} object classes.")

    def detect(self, frame):
        """
        Run YOLO inference on one frame.

        Parameters
        ----------
        frame : numpy.ndarray
            The image captured from the webcam.

        Returns
        -------
        results : list
            YOLO result objects for the input frame.

        What YOLO does internally
        -------------------------
        1. Preprocess the frame
        2. Run neural network inference
        3. Post-process predictions
        4. Return detections with boxes, classes, and confidence scores
        """

        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD
        )

        if PRINT_DETECTIONS:
            objects = self.list_detected_objects(results)
            print("Detected objects:", objects)

        return results

    def draw(self, results):
        """
        Draw bounding boxes and labels onto the frame.

        Parameters
        ----------
        results : list
            YOLO result objects.

        Returns
        -------
        annotated_frame : numpy.ndarray
            Frame with visual detection overlays.
        """

        annotated_frame = results[0].plot()
        return annotated_frame

    def count_detections(self, results):
        """
        Count the number of detected objects in the frame.

        Returns
        -------
        int
            Total number of detections.
        """

        boxes = results[0].boxes

        if boxes is None:
            return 0

        return len(boxes)

    def list_detected_objects(self, results):
        """
        Extract human-readable object names from YOLO results.

        Example output:
            ["person", "chair", "cell phone"]

        Returns
        -------
        list[str]
            List of detected object names.
        """

        objects = []
        boxes = results[0].boxes

        if boxes is None:
            return objects

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            objects.append(class_name)

        return objects