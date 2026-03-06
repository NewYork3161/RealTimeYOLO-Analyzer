"""
analytics.py
------------

Loads detection logs and performs analysis using pandas.

Purpose
-------
This module transforms raw CSV logs into useful performance summaries.

Why pandas is used
------------------
Pandas is one of the most widely used Python libraries for data analysis.

By using pandas here, this project demonstrates that the system is not
just a real-time AI demo, but also an analytics pipeline.

This is valuable for portfolio work because many real production systems
combine:

- computer vision
- structured logging
- analytics
- reporting
"""

import pandas as pd
from config import LOG_FILE


def load_data():
    """
    Load the detection log CSV into a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
        Structured table containing detection records.
    """

    df = pd.read_csv(LOG_FILE)
    return df


def summary_statistics():
    """
    Print summary statistics about the detection session.

    Metrics included
    ----------------
    - Total frames logged
    - Average objects per frame
    - Average inference time
    """

    df = load_data()

    print("\nDetection Summary\n")
    print("Total Frames Logged:", len(df))
    print("Average Objects Per Frame:", df["object_count"].mean())
    print("Average Inference Time:", df["inference_time"].mean())


if __name__ == "__main__":
    summary_statistics()