"""
main.py
-------

Main entry point for the YOLO object detection pipeline.

System Pipeline
---------------
Camera -> Frame Capture -> YOLO Detection -> Logging -> Display

What this file does
-------------------
This file orchestrates the full real-time workflow.

High-level process
------------------
1. Initialize logging
2. Load the YOLO detector
3. Start the webcam
4. Capture frames continuously
5. Run object detection on each frame
6. Measure inference time
7. Log detection results to CSV
8. Draw detection boxes on the frame
9. Display the annotated frame
10. Exit cleanly when the user quits

Why this file is important
--------------------------
This file acts as the "controller" of the application.

It does not perform detection logic itself.
It does not manage analytics itself.
It coordinates the modules that do those jobs.

That modular architecture is important in real production systems.
"""

import cv2
import time

from camera import Camera
from detector import ObjectDetector
from logger import log_detection, initialize_log
from config import WINDOW_NAME, DEBUG


def main():
    """
    Main driver function for the real-time detection system.
    """

    # ------------------------------------------------------
    # Step 1: Initialize the logging system
    # ------------------------------------------------------
    # This ensures the CSV file exists before any detections
    # are recorded.
    initialize_log()

    # ------------------------------------------------------
    # Step 2: Load the YOLO detector
    # ------------------------------------------------------
    detector = ObjectDetector()

    # ------------------------------------------------------
    # Step 3: Start the camera module
    # ------------------------------------------------------
    try:
        camera = Camera()
    except RuntimeError as error:
        print(error)
        return

    # Create the OpenCV display window
    cv2.namedWindow(WINDOW_NAME)

    if DEBUG:
        print("Starting real-time detection loop. Press 'q' to exit.")

    # ------------------------------------------------------
    # Step 4: Real-time processing loop
    # ------------------------------------------------------
    while True:
        # Capture one frame from the webcam
        frame = camera.get_frame()

        # If frame capture fails, stop the loop
        if frame is None:
            break

        # --------------------------------------------------
        # Step 5: Measure inference time
        # --------------------------------------------------
        start_time = time.time()

        # Run YOLO object detection
        results = detector.detect(frame)

        end_time = time.time()
        inference_time = end_time - start_time

        # --------------------------------------------------
        # Step 6: Extract object data
        # --------------------------------------------------
        objects = detector.list_detected_objects(results)
        count = len(objects)

        # --------------------------------------------------
        # Step 7: Log the detection results
        # --------------------------------------------------
        log_detection(count, objects, inference_time)

        # --------------------------------------------------
        # Step 8: Draw bounding boxes and labels
        # --------------------------------------------------
        annotated_frame = detector.draw(results)

        # --------------------------------------------------
        # Step 9: Display annotated video
        # --------------------------------------------------
        cv2.imshow(WINDOW_NAME, annotated_frame)

        # --------------------------------------------------
        # Step 10: Exit conditions
        # --------------------------------------------------
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # If window is manually closed, quit
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ------------------------------------------------------
    # Step 11: Cleanup resources
    # ------------------------------------------------------
    camera.release()
    cv2.destroyAllWindows()

    if DEBUG:
        print("Application exited cleanly.")


if __name__ == "__main__":
    main()