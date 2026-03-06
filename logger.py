"""
logger.py
---------

Logging module for the YOLO object detection system.

Purpose
-------
This file stores structured detection data into CSV format so that
the system can later analyze performance and detection behavior.

Why logging matters
-------------------
Logging turns the real-time detection system into an analytics system.

Instead of only showing detections on screen, we also keep records
that can later be used to answer questions like:

- How fast was inference running?
- How many objects were detected per frame?
- What types of objects appeared most often?
- Did performance change over time?

This is important in portfolio projects because it shows an employer
that the system is not just "doing AI" but also collecting useful data.
"""

import csv
import os
from datetime import datetime
from config import LOG_DIR, LOG_FILE


def initialize_log():
    """
    Create the log directory and CSV file if they do not already exist.

    This function writes the CSV header row only once.
    """

    # Create logs folder if needed
    os.makedirs(LOG_DIR, exist_ok=True)

    # Only create the file/header if the file does not already exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Header row describes each column in the dataset
            writer.writerow([
                "timestamp",
                "object_count",
                "objects_detected",
                "inference_time"
            ])


def log_detection(object_count, objects, inference_time):
    """
    Append one detection record to the CSV file.

    Parameters
    ----------
    object_count : int
        Number of objects detected in the frame.

    objects : list[str]
        Human-readable names of objects found in the frame.

    inference_time : float
        Time in seconds required for YOLO inference.
    """

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            datetime.now(),          # exact time the frame was processed
            object_count,            # total detections
            ";".join(objects),       # store list as one semicolon-separated string
            inference_time           # speed of inference for this frame
        ])