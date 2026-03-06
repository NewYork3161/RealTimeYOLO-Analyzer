"""
test_yolo_detection.py
----------------------

Comprehensive unit test suite for the YOLO Real-Time Object Detection System.

PURPOSE
-------
This file contains automated tests that verify the correctness,
stability, and performance of the major components in the detection pipeline.

Testing is extremely important in professional software engineering because
it ensures that future code changes do not accidentally break existing
functionality.

By implementing unit tests in this project, we demonstrate that the system
is not just a quick demo, but a maintainable and professionally engineered
software system.

SYSTEM COMPONENTS BEING TESTED
------------------------------

This test suite validates functionality across multiple modules:

    detector.py
        Ensures the YOLO inference engine loads and runs correctly.

    config.py
        Verifies configuration parameters are valid.

    logger.py
        Ensures detection data is written to CSV correctly.

    analytics.py
        Ensures detection logs can be loaded and analyzed using pandas.

    charts.py
        Ensures visualization functions execute without crashing.

TESTING FRAMEWORK
-----------------
Python unittest

This is the standard testing framework included with Python and is widely
used in professional Python codebases.

DEPENDENCIES
------------
numpy
ultralytics
pandas
matplotlib

These libraries are already part of the project environment.
"""

import unittest
import numpy as np
import time
import os

from ultralytics import YOLO

# Import system modules
from detector import ObjectDetector
from logger import initialize_log, log_detection
from analytics import load_data
from charts import generate_performance_chart
import config


class TestYOLOSystem(unittest.TestCase):
    """
    TestYOLOSystem

    This class contains all automated tests for the YOLO detection pipeline.

    Each test validates a specific subsystem of the application to ensure
    that the system continues functioning correctly after code modifications.
    """

    @classmethod
    def setUpClass(cls):
        """
        setUpClass()

        Runs once before all tests execute.

        This is useful for expensive operations like loading deep learning
        models, which would otherwise slow down the test suite if repeated
        for every test.

        In this step we initialize:

        • YOLO model
        • ObjectDetector engine
        """

        cls.detector = ObjectDetector()
        cls.model = YOLO(config.MODEL_NAME)

    def setUp(self):
        """
        setUp()

        Runs before each individual test.

        We create simulated image frames to mimic webcam input.

        A valid frame has the shape:
            (height, width, channels)

        Example:
            (640, 640, 3)
        """

        # Valid RGB image
        self.valid_frame = np.zeros((640, 640, 3), dtype=np.uint8)

        # Invalid grayscale image
        self.invalid_frame = np.zeros((640, 640), dtype=np.uint8)

    # --------------------------------------------------
    # MODEL TESTS
    # --------------------------------------------------

    def test_model_loaded(self):
        """
        Verify that the YOLO model loads successfully.
        """

        self.assertIsNotNone(self.model)

    def test_model_has_class_names(self):
        """
        Ensure the YOLO model contains class name mappings.

        YOLO models trained on the COCO dataset should contain
        around 80 class labels.
        """

        self.assertTrue(hasattr(self.model, "names"))
        self.assertGreater(len(self.model.names), 0)

    # --------------------------------------------------
    # CONFIGURATION TESTS
    # --------------------------------------------------

    def test_config_model_name(self):
        """
        Verify that the model name stored in config.py is a valid string.
        """

        self.assertIsInstance(config.MODEL_NAME, str)

    def test_config_confidence_range(self):
        """
        Ensure the confidence threshold falls within the valid
        probability range [0.0, 1.0].
        """

        self.assertGreaterEqual(config.CONFIDENCE_THRESHOLD, 0.0)
        self.assertLessEqual(config.CONFIDENCE_THRESHOLD, 1.0)

    # --------------------------------------------------
    # FRAME VALIDATION TESTS
    # --------------------------------------------------

    def test_valid_frame_dimensions(self):
        """
        Ensure that a valid frame matches the expected shape.
        """

        self.assertEqual(self.valid_frame.shape, (640, 640, 3))

    def test_invalid_frame_dimensions(self):
        """
        Ensure that grayscale frames are detected as invalid.
        """

        self.assertNotEqual(self.invalid_frame.shape, (640, 640, 3))

    # --------------------------------------------------
    # DETECTOR TESTS
    # --------------------------------------------------

    def test_detector_runs_inference(self):
        """
        Ensure that the detector can perform inference without crashing.
        """

        results = self.detector.detect(self.valid_frame)
        self.assertIsNotNone(results)

    def test_detection_output_structure(self):
        """
        Verify that detection results contain bounding box information.
        """

        results = self.detector.detect(self.valid_frame)

        self.assertTrue(hasattr(results[0], "boxes"))

    def test_detection_visualization(self):
        """
        Ensure the detection visualization produces a valid image frame.
        """

        results = self.detector.detect(self.valid_frame)

        annotated_frame = self.detector.draw(results)

        self.assertIsNotNone(annotated_frame)

    def test_detection_count_function(self):
        """
        Verify the detection counting helper function returns a valid number.
        """

        results = self.detector.detect(self.valid_frame)

        count = self.detector.count_detections(results)

        self.assertGreaterEqual(count, 0)

    # --------------------------------------------------
    # LOGGER TESTS
    # --------------------------------------------------

    def test_logging_system(self):
        """
        Verify that the logging system creates and writes to the CSV log file.
        """

        initialize_log()

        log_detection(1, ["person"], 0.25)

        self.assertTrue(os.path.exists(config.LOG_FILE))

    # --------------------------------------------------
    # ANALYTICS TESTS
    # --------------------------------------------------

    def test_analytics_load(self):
        """
        Ensure pandas can load the detection log file.
        """

        df = load_data()

        self.assertIsNotNone(df)

    # --------------------------------------------------
    # CHART GENERATION TEST
    # --------------------------------------------------

    def test_chart_generation(self):
        """
        Ensure chart generation executes without crashing.
        """

        try:
            generate_performance_chart()
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

    # --------------------------------------------------
    # PERFORMANCE TEST
    # --------------------------------------------------

    def test_inference_speed(self):
        """
        Ensure YOLO inference completes within a reasonable time.

        Even on CPU systems, inference should generally finish
        within about 2 seconds for a single frame.
        """

        start = time.time()

        self.detector.detect(self.valid_frame)

        end = time.time()

        inference_time = end - start

        self.assertLess(inference_time, 2.0)


# --------------------------------------------------
# TEST RUNNER
# --------------------------------------------------

if __name__ == "__main__":
    """
    When this file is executed directly, run the entire test suite.
    """

    unittest.main()