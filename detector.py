"""
detector.py
Core detection module for Helmet Safety Compliance System.
Handles person detection, helmet association, and compliance classification.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detected person with compliance status."""
    bbox: tuple          
    confidence: float
    compliant: bool
    helmet_conf: float = 0.0
    track_id: Optional[int] = None
    head_bbox: Optional[tuple] = None


class HelmetDetector:
    """
    Detects persons and helmets using YOLOv8, then associates
    each person with a helmet based on spatial overlap logic.
    
    Args:
        model_path: Path to YOLO weights (.pt file).
        conf_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        head_ratio: Fraction of person bbox height to consider as head region.
    """

    PERSON_CLASS_ID = 0

    HELMET_CLASS_ID = 1
    NO_HELMET_CLASS_ID = 2

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        head_ratio: float = 0.25,
        use_custom_model: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.head_ratio = head_ratio
        self.use_custom_model = use_custom_model

        logger.info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.overrides["conf"] = conf_threshold
        self.model.overrides["iou"] = iou_threshold
        self.model.overrides["verbose"] = False
        logger.info("Model loaded successfully.")


    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single frame and return a list of Detection objects.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            List of Detection instances.
        """
        results = self.model(frame, stream=False)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return []

        if self.use_custom_model:
            return self._parse_custom_model(boxes, frame)
        else:
            return self._parse_base_model(boxes, frame)


    def _parse_base_model(self, boxes, frame: np.ndarray) -> list[Detection]:
        """
        Handles base YOLO model (person-only or COCO).
        Helmet detection falls back to head-region colour/shape heuristic.
        """
        persons = []
        helmets = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = tuple(map(int, box.xyxy[0].tolist()))

            if cls_id == self.PERSON_CLASS_ID:
                persons.append((xyxy, conf))

        detections = []
        for bbox, conf in persons:
            head_bbox = self._estimate_head_region(bbox)
            compliant, helmet_conf = self._heuristic_helmet_check(frame, head_bbox)
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=conf,
                    compliant=compliant,
                    helmet_conf=helmet_conf,
                    head_bbox=head_bbox,
                )
            )
        return detections

    def _parse_custom_model(self, boxes, frame: np.ndarray) -> list[Detection]:
        """
        Handles custom-trained model with explicit helmet / no-helmet classes.
        Associates each person with the best overlapping helmet detection.
        """
        persons = []
        helmet_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = tuple(map(int, box.xyxy[0].tolist()))

            if cls_id == self.PERSON_CLASS_ID:
                persons.append((xyxy, conf))
            elif cls_id == self.HELMET_CLASS_ID:
                helmet_boxes.append((xyxy, conf))

        detections = []
        for p_bbox, p_conf in persons:
            head_bbox = self._estimate_head_region(p_bbox)
            best_iou = 0.0
            best_conf = 0.0
            for h_bbox, h_conf in helmet_boxes:
                iou = self._compute_iou(head_bbox, h_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = h_conf

            compliant = best_iou >= 0.10
            detections.append(
                Detection(
                    bbox=p_bbox,
                    confidence=p_conf,
                    compliant=compliant,
                    helmet_conf=best_conf,
                    head_bbox=head_bbox,
                )
            )
        return detections

    def _estimate_head_region(self, person_bbox: tuple) -> tuple:
        """
        Returns bounding box for the estimated head region
        (top `head_ratio` fraction of the person bbox).
        """
        x1, y1, x2, y2 = person_bbox
        h = y2 - y1
        head_y2 = y1 + int(h * self.head_ratio)
        return (x1, y1, x2, head_y2)

    def _heuristic_helmet_check(
        self, frame: np.ndarray, head_bbox: tuple
    ) -> tuple[bool, float]:
        """
        Lightweight colour-based heuristic to estimate helmet presence
        when a dedicated helmet detection model isn't available.
        Returns (is_compliant, confidence_score).

        The heuristic looks for hard-hat colours (yellow, white, orange, red,
        blue) in the HSV space within the head region.
        """
        x1, y1, x2, y2 = head_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        masks = [
            # Yellow
            cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255])),
            # White
            cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Orange
            cv2.inRange(hsv, np.array([10, 100, 100]), np.array([20, 255, 255])),
            # Red (two ranges)
            cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])),
            # Blue
            cv2.inRange(hsv, np.array([100, 100, 50]), np.array([130, 255, 255])),
        ]

        combined = np.zeros_like(masks[0])
        for m in masks:
            combined = cv2.bitwise_or(combined, m)

        total_pixels = roi.shape[0] * roi.shape[1]
        helmet_pixels = cv2.countNonZero(combined)
        ratio = helmet_pixels / total_pixels if total_pixels > 0 else 0.0

        # Require at least 15% of head ROI to be helmet colour
        compliant = ratio >= 0.15
        confidence = min(ratio * 3.0, 1.0)  # scale to [0,1]
        return compliant, round(confidence, 3)

    @staticmethod
    def _compute_iou(box_a: tuple, box_b: tuple) -> float:
        """Compute intersection-over-union between two (x1,y1,x2,y2) boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
