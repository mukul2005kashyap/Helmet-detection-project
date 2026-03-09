"""
tests/test_analytics.py
Unit tests for the ComplianceAnalytics module.
"""

import os
import sys
import tempfile
import csv
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector import Detection
from analytics import ComplianceAnalytics, FrameStats


def _make_tracked(compliant_flags: list[bool]) -> list:
    """Helper: build a list of (Detection, track_id) tuples."""
    results = []
    for i, flag in enumerate(compliant_flags):
        d = Detection(bbox=(i*10, 0, i*10+10, 50), confidence=0.8, compliant=flag)
        results.append((d, i + 1))
    return results


class TestComplianceAnalytics:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.analytics = ComplianceAnalytics(output_dir=self.tmp, fps=30.0)

    def test_update_returns_correct_stats(self):
        tracked = _make_tracked([True, False, True])
        stats = self.analytics.update(frame_id=0, tracked_detections=tracked)
        assert stats.total_persons == 3
        assert stats.compliant == 2
        assert stats.non_compliant == 1

    def test_compliance_rate(self):
        tracked = _make_tracked([True, True, False, False])
        stats = self.analytics.update(0, tracked)
        assert stats.compliance_rate == pytest.approx(0.5)

    def test_empty_frame(self):
        stats = self.analytics.update(0, [])
        assert stats.total_persons == 0
        assert stats.compliance_rate == 1.0  # 100% when no one present

    def test_frame_csv_written(self):
        self.analytics.update(0, _make_tracked([True, False]))
        csv_path = os.path.join(self.tmp, "frame_compliance_log.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 data row

    def test_track_record_accumulated(self):
        for fid in range(5):
            self.analytics.update(fid, _make_tracked([True, False]))
        assert len(self.analytics.tracks) == 2
        assert self.analytics.tracks[1].compliant_frames == 5
        assert self.analytics.tracks[2].non_compliant_frames == 5

    def test_save_summary(self):
        self.analytics.update(0, _make_tracked([True, False]))
        self.analytics.save_summary()
        summary_path = os.path.join(self.tmp, "track_summary_log.csv")
        assert os.path.exists(summary_path)

    def test_violation_log(self):
        self.analytics.log_violation(frame_id=10, track_id=2, snapshot_path="/tmp/snap.jpg")
        viol_path = os.path.join(self.tmp, "violation_log.csv")
        with open(viol_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 row
        assert rows[1][2] == "2"  # track_id
