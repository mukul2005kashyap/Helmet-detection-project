"""
analytics.py
Tracks compliance statistics across frames and writes CSV logs.
"""

import csv
import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FrameStats:
    frame_id: int
    timestamp: float
    total_persons: int
    compliant: int
    non_compliant: int

    @property
    def compliance_rate(self) -> float:
        if self.total_persons == 0:
            return 1.0
        return self.compliant / self.total_persons


@dataclass
class TrackRecord:
    """Maintains per-track compliance history."""
    track_id: int
    first_seen: float
    last_seen: float = 0.0
    compliant_frames: int = 0
    non_compliant_frames: int = 0
    violation_saved: bool = False

    @property
    def total_frames(self) -> int:
        return self.compliant_frames + self.non_compliant_frames

    @property
    def compliance_rate(self) -> float:
        if self.total_frames == 0:
            return 1.0
        return self.compliant_frames / self.total_frames


class ComplianceAnalytics:
    """
    Aggregates frame-level and track-level compliance data.
    Writes per-frame CSV logs and per-track summary logs.

    Args:
        output_dir: Directory where CSV files are saved.
        fps: Video FPS (used for timestamp conversion).
    """

    def __init__(self, output_dir: str = "outputs/logs", fps: float = 30.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        self._frame_stats: list[FrameStats] = []
        self._tracks: dict[int, TrackRecord] = {}
        self._start_time = datetime.now()

        self.total_frames_processed = 0
        self.total_violations_detected = 0

        self._frame_csv_path = self.output_dir / "frame_compliance_log.csv"
        self._track_csv_path = self.output_dir / "track_summary_log.csv"
        self._violation_csv_path = self.output_dir / "violation_log.csv"

        self._init_csv_files()


    def update(self, frame_id: int, tracked_detections: list) -> FrameStats:
        """
        Call once per frame after tracking.

        Args:
            frame_id: Frame index (0-based).
            tracked_detections: list of (Detection, track_id) tuples.

        Returns:
            FrameStats for this frame.
        """
        timestamp = frame_id / self.fps
        compliant_count = sum(1 for d, _ in tracked_detections if d.compliant)
        non_compliant_count = len(tracked_detections) - compliant_count

        stats = FrameStats(
            frame_id=frame_id,
            timestamp=round(timestamp, 3),
            total_persons=len(tracked_detections),
            compliant=compliant_count,
            non_compliant=non_compliant_count,
        )
        self._frame_stats.append(stats)
        self.total_frames_processed += 1
        self.total_violations_detected += non_compliant_count

        for det, tid in tracked_detections:
            if tid not in self._tracks:
                self._tracks[tid] = TrackRecord(
                    track_id=tid,
                    first_seen=timestamp,
                )
            rec = self._tracks[tid]
            rec.last_seen = timestamp
            if det.compliant:
                rec.compliant_frames += 1
            else:
                rec.non_compliant_frames += 1

        self._append_frame_row(stats)
        return stats

    def log_violation(self, frame_id: int, track_id: int, snapshot_path: str):
        """Record a saved violation snapshot."""
        timestamp = frame_id / self.fps
        with open(self._violation_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                round(timestamp, 3),
                track_id,
                snapshot_path,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])
        if track_id in self._tracks:
            self._tracks[track_id].violation_saved = True

    def save_summary(self):
        """Write final track-level summary CSV."""
        with open(self._track_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "track_id", "first_seen_s", "last_seen_s",
                "compliant_frames", "non_compliant_frames",
                "total_frames", "compliance_rate_%", "violation_saved",
            ])
            for tid, rec in sorted(self._tracks.items()):
                writer.writerow([
                    rec.track_id,
                    rec.first_seen,
                    rec.last_seen,
                    rec.compliant_frames,
                    rec.non_compliant_frames,
                    rec.total_frames,
                    round(rec.compliance_rate * 100, 1),
                    rec.violation_saved,
                ])
        logger.info(f"Track summary saved → {self._track_csv_path}")

    def print_summary(self):
        """Print a final summary to stdout."""
        total_persons = sum(s.total_persons for s in self._frame_stats)
        total_compliant = sum(s.compliant for s in self._frame_stats)
        total_non_compliant = sum(s.non_compliant for s in self._frame_stats)
        avg_rate = (
            (total_compliant / total_persons * 100) if total_persons > 0 else 100.0
        )

        print("\n" + "="*55)
        print("  HELMET COMPLIANCE DETECTION — FINAL REPORT")
        print("="*55)
        print(f"  Frames processed       : {self.total_frames_processed}")
        print(f"  Unique tracks          : {len(self._tracks)}")
        print(f"  Total person-frames    : {total_persons}")
        print(f"  Compliant              : {total_compliant}")
        print(f"  Non-compliant          : {total_non_compliant}")
        print(f"  Average compliance rate: {avg_rate:.1f}%")
        print(f"  Frame log              : {self._frame_csv_path}")
        print(f"  Track summary          : {self._track_csv_path}")
        print(f"  Violation log          : {self._violation_csv_path}")
        print("="*55 + "\n")

    @property
    def tracks(self) -> dict[int, TrackRecord]:
        return self._tracks

    def _init_csv_files(self):
        with open(self._frame_csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "frame_id", "timestamp_s", "total_persons",
                "compliant", "non_compliant", "compliance_rate_%",
            ])

        with open(self._violation_csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "frame_id", "timestamp_s", "track_id",
                "snapshot_path", "recorded_at",
            ])

    def _append_frame_row(self, stats: FrameStats):
        with open(self._frame_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                stats.frame_id,
                stats.timestamp,
                stats.total_persons,
                stats.compliant,
                stats.non_compliant,
                round(stats.compliance_rate * 100, 1),
            ])
