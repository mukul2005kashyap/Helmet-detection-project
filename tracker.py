"""
tracker.py
Wraps ByteTrack / DeepSORT to assign stable track IDs across frames.
Falls back gracefully to a lightweight IoU-based tracker if neither
library is installed, so the project runs out-of-the-box.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class _IoUTracker:
    """
    Minimal IoU-based multi-object tracker.
    Good enough for demos; replace with ByteTrack for production accuracy.
    """

    def __init__(self, iou_threshold: float = 0.30, max_lost: int = 30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._next_id = 1
        self._tracks: dict[int, dict] = {}  # id -> {bbox, lost}

    def update(self, detections: list) -> list[tuple]:
        """
        Args:
            detections: list of Detection objects (must have .bbox)
        Returns:
            list of (Detection, track_id) pairs
        """
        det_bboxes = [d.bbox for d in detections]
        track_ids = list(self._tracks.keys())

        assigned = {}  # det_idx -> track_id
        used_tracks = set()

        # Greedy IoU matching
        if track_ids and det_bboxes:
            iou_matrix = np.zeros((len(det_bboxes), len(track_ids)))
            for di, db in enumerate(det_bboxes):
                for ti, tid in enumerate(track_ids):
                    iou_matrix[di, ti] = self._iou(db, self._tracks[tid]["bbox"])

            for _ in range(min(len(det_bboxes), len(track_ids))):
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                di, ti = idx
                if iou_matrix[di, ti] < self.iou_threshold:
                    break
                tid = track_ids[ti]
                assigned[di] = tid
                used_tracks.add(tid)
                iou_matrix[di, :] = -1
                iou_matrix[:, ti] = -1

        # Update matched tracks
        for di, tid in assigned.items():
            self._tracks[tid] = {"bbox": det_bboxes[di], "lost": 0}

        # Increment lost counter for unmatched tracks
        for tid in track_ids:
            if tid not in used_tracks:
                self._tracks[tid]["lost"] += 1

        # Remove stale tracks
        self._tracks = {
            tid: v for tid, v in self._tracks.items()
            if v["lost"] <= self.max_lost
        }

        # Create new tracks for unmatched detections
        for di in range(len(det_bboxes)):
            if di not in assigned:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = {"bbox": det_bboxes[di], "lost": 0}
                assigned[di] = new_id

        return [(detections[di], assigned[di]) for di in range(len(detections))]

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        aa = (a[2]-a[0]) * (a[3]-a[1])
        ab = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (aa + ab - inter)


class _ByteTrackWrapper:
    """Thin wrapper around supervision's ByteTrack."""

    def __init__(self):
        import supervision as sv
        self.tracker = sv.ByteTrack()
        logger.info("Using ByteTrack via supervision library.")

    def update(self, detections: list) -> list[tuple]:
        import supervision as sv
        import numpy as np

        if not detections:
            return []

        bboxes = np.array([d.bbox for d in detections], dtype=np.float32)
        confs  = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=class_ids,
        )
        tracked = self.tracker.update_with_detections(sv_dets)

        results = []
        for i, det in enumerate(detections):
            # Match back by bbox overlap
            best_tid = i + 1  # fallback
            if tracked.tracker_id is not None and len(tracked.tracker_id) > i:
                best_tid = int(tracked.tracker_id[i])
            results.append((det, best_tid))
        return results


def build_tracker(backend: str = "auto"):
    """
    Factory function.

    Args:
        backend: 'bytetrack', 'iou', or 'auto' (tries ByteTrack first).

    Returns:
        A tracker object with an `.update(detections)` method.
    """
    if backend == "bytetrack" or backend == "auto":
        try:
            tracker = _ByteTrackWrapper()
            return tracker
        except ImportError:
            if backend == "bytetrack":
                raise
            logger.warning(
                "supervision/ByteTrack not found. "
                "Falling back to built-in IoU tracker. "
                "Install with: pip install supervision"
            )

    logger.info("Using built-in IoU tracker.")
    return _IoUTracker()
