"""
visualizer.py
Handles all OpenCV drawing — bounding boxes, labels, HUD overlay.
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Optional


# Colour palette (BGR)
COLOUR_COMPLIANT     = (57, 255, 20)     # neon green
COLOUR_NON_COMPLIANT = (0, 60, 255)      # vivid red
COLOUR_HEAD_BOX      = (0, 200, 255)     # amber
COLOUR_HUD_BG        = (20, 20, 20)
COLOUR_WHITE         = (255, 255, 255)
COLOUR_WARN          = (0, 140, 255)     # orange


class Visualizer:
    """
    Draws detection overlays and compliance HUD on video frames.

    Args:
        show_head_box: Draw the estimated head-region box.
        show_confidence: Display detection confidence score.
        show_track_id: Display track ID next to each person.
        font_scale: OpenCV font scale multiplier.
    """

    def __init__(
        self,
        show_head_box: bool = False,
        show_confidence: bool = True,
        show_track_id: bool = True,
        font_scale: float = 0.55,
    ):
        self.show_head_box = show_head_box
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(self, frame: np.ndarray, tracked_detections: list) -> np.ndarray:
        """
        Draw bounding boxes and labels for all tracked detections.

        Args:
            frame: BGR frame to draw on (will be modified in-place).
            tracked_detections: list of (Detection, track_id) tuples.

        Returns:
            Annotated frame.
        """
        for det, tid in tracked_detections:
            colour = COLOUR_COMPLIANT if det.compliant else COLOUR_NON_COMPLIANT
            x1, y1, x2, y2 = det.bbox

            # Person bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Optional head region box
            if self.show_head_box and det.head_bbox:
                hx1, hy1, hx2, hy2 = det.head_bbox
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), COLOUR_HEAD_BOX, 1)

            # Label background + text
            label_parts = []
            if self.show_track_id:
                label_parts.append(f"ID:{tid}")
            label_parts.append("✓ Helmet" if det.compliant else "✗ No Helmet")
            if self.show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            label = "  ".join(label_parts)

            (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
            lx, ly = x1, max(y1 - 6, th + 4)
            cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), colour, -1)
            cv2.putText(
                frame, label, (lx + 2, ly - 2),
                self.font, self.font_scale, COLOUR_WHITE, 1, cv2.LINE_AA,
            )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_id: int,
        total: int,
        compliant: int,
        non_compliant: int,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draws a semi-transparent heads-up display (HUD) in the top-left corner.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # HUD background
        cv2.rectangle(overlay, (8, 8), (280, 140), COLOUR_HUD_BG, -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        lines = [
            (f"Frame: {frame_id:>6}", COLOUR_WHITE),
            (f"FPS  : {fps:>6.1f}",  COLOUR_WHITE),
            (f"Total: {total:>6}",    COLOUR_WHITE),
            (f"OK   : {compliant:>6}", COLOUR_COMPLIANT),
            (f"VIOL : {non_compliant:>6}", COLOUR_NON_COMPLIANT if non_compliant > 0 else COLOUR_WHITE),
        ]
        for i, (text, colour) in enumerate(lines):
            cv2.putText(
                frame, text, (16, 32 + i * 22),
                self.font, 0.52, colour, 1, cv2.LINE_AA,
            )

        # Violation warning banner
        if non_compliant > 0:
            msg = f"  ⚠  {non_compliant} VIOLATION{'S' if non_compliant > 1 else ''} DETECTED  ⚠  "
            (bw, bh), _ = cv2.getTextSize(msg, self.font, 0.65, 2)
            bx = (w - bw) // 2
            by = h - 20
            cv2.rectangle(frame, (bx - 8, by - bh - 8), (bx + bw + 8, by + 8), (0, 0, 180), -1)
            cv2.putText(
                frame, msg, (bx, by),
                self.font, 0.65, COLOUR_WHITE, 2, cv2.LINE_AA,
            )

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (w - 220, h - 12), self.font, 0.42, (180, 180, 180), 1)

        return frame

    def make_violation_snapshot(
        self,
        frame: np.ndarray,
        tracked_detections: list,
        frame_id: int,
        save_path: str,
    ):
        """
        Saves a cropped snapshot of each non-compliant person.

        Args:
            frame: The current annotated frame.
            tracked_detections: list of (Detection, track_id).
            frame_id: Current frame index.
            save_path: Directory where snapshots are saved.
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        saved = []
        for det, tid in tracked_detections:
            if det.compliant:
                continue
            x1, y1, x2, y2 = det.bbox
            # Pad slightly for context
            pad = 10
            h, w = frame.shape[:2]
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)

            crop = frame[cy1:cy2, cx1:cx2]
            fname = os.path.join(save_path, f"violation_f{frame_id:06d}_t{tid}.jpg")
            cv2.imwrite(fname, crop)
            saved.append((fname, tid))

        return saved
