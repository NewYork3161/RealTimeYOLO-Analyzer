"""
charts.py
---------

Creates visual charts from YOLO detection log data.

Purpose
-------
This module turns raw logged data into visual insights.

Why this matters
----------------
Charts make it easier to understand system behavior over time.

For example, we can visualize:
- whether inference speed is stable
- whether the system slows down under certain conditions
- how many objects tend to appear per frame

This makes the project stronger as a portfolio piece because it shows
analysis and reporting, not just detection.
"""

import pandas as pd
import matplotlib.pyplot as plt
from config import LOG_FILE


def generate_performance_chart():
    """
    Plot inference time per frame.

    X-axis:
        frame index

    Y-axis:
        inference time in seconds
    """

    df = pd.read_csv(LOG_FILE)

    plt.figure()
    plt.plot(df["inference_time"])
    plt.title("YOLO Inference Time per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True)
    plt.show()


def generate_object_count_chart():
    """
    Plot number of detected objects per frame.

    X-axis:
        frame index

    Y-axis:
        object count
    """

    df = pd.read_csv(LOG_FILE)

    plt.figure()
    plt.plot(df["object_count"])
    plt.title("Objects Detected per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Object Count")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    generate_performance_chart()
    generate_object_count_chart()