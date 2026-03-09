"""
tests/test_detector.py
Unit tests for the HelmetDetector and helper functions.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Make sure parent dir is on path when running tests directly
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector import HelmetDetector, Detection


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

class TestComputeIoU:
    def test_identical_boxes(self):
        box = (0, 0, 100, 100)
        assert HelmetDetector._compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0, 0, 50, 50)
        b = (100, 100, 200, 200)
        assert HelmetDetector._compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        # Intersection: 50×50 = 2500. Union: 10000+10000-2500 = 17500
        expected = 2500 / 17500
        assert HelmetDetector._compute_iou(a, b) == pytest.approx(expected, rel=1e-5)

    def test_contained_box(self):
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        # Intersection = inner area = 2500, union = outer area = 10000
        expected = 2500 / 10000
        assert HelmetDetector._compute_iou(outer, inner) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Head region estimation
# ---------------------------------------------------------------------------

class TestHeadRegion:
    def setup_method(self):
        # Patch YOLO to avoid loading weights during unit tests
        with patch("detector.YOLO"):
            self.detector = HelmetDetector.__new__(HelmetDetector)
            self.detector.head_ratio = 0.25

    def test_head_bbox_top_quarter(self):
        person_bbox = (0, 0, 100, 200)
        head = self.detector._estimate_head_region(person_bbox)
        assert head == (0, 0, 100, 50)   # top 25% of height = 50px

    def test_head_bbox_non_zero_y(self):
        person_bbox = (10, 40, 110, 240)  # height = 200
        head = self.detector._estimate_head_region(person_bbox)
        # head_y2 = 40 + 0.25*200 = 40+50 = 90
        assert head == (10, 40, 110, 90)


# ---------------------------------------------------------------------------
# Heuristic helmet check (no YOLO needed)
# ---------------------------------------------------------------------------

class TestHeuristicHelmetCheck:
    def setup_method(self):
        with patch("detector.YOLO"):
            self.detector = HelmetDetector.__new__(HelmetDetector)

    def test_yellow_helmet_detected(self):
        """A frame with a yellow region in the head ROI should return compliant."""
        import cv2
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Fill top region with yellow (BGR: 0, 255, 255)
        frame[0:50, 0:200] = (0, 255, 255)
        head_bbox = (0, 0, 200, 50)
        compliant, conf = self.detector._heuristic_helmet_check(frame, head_bbox)
        assert compliant is True
        assert conf > 0.0

    def test_black_frame_not_compliant(self):
        """All-black frame should not trigger helmet detection."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        head_bbox = (0, 0, 200, 50)
        compliant, conf = self.detector._heuristic_helmet_check(frame, head_bbox)
        assert compliant is False

    def test_empty_roi(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Invalid ROI (x1 >= x2)
        compliant, conf = self.detector._heuristic_helmet_check(frame, (50, 50, 50, 100))
        assert compliant is False
        assert conf == 0.0


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_default_fields(self):
        d = Detection(bbox=(0, 0, 100, 200), confidence=0.9, compliant=True)
        assert d.track_id is None
        assert d.helmet_conf == 0.0
        assert d.head_bbox is None

    def test_non_compliant(self):
        d = Detection(bbox=(0, 0, 50, 100), confidence=0.75, compliant=False)
        assert d.compliant is False
