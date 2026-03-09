"""
config/settings.py
Centralised configuration for the Helmet Safety Detection system.
All tuneable parameters live here — no magic numbers scattered across files.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # ---- Model ----
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.40
    iou_threshold: float = 0.45
    head_ratio: float = 0.25        # top fraction of person bbox treated as head
    use_custom_model: bool = False  # True if weights include helmet/no-helmet classes

    # ---- Tracker ----
    tracker_backend: str = "auto"   # "auto" | "bytetrack" | "iou"
    iou_tracker_threshold: float = 0.30
    max_lost_frames: int = 30       # frames before a track is dropped

    # ---- Output ----
    output_dir: str = "outputs"
    snapshot_interval: int = 30     # save violation snapshot every N frames
    save_video: bool = False
    show_display: bool = True
    show_head_box: bool = False
    show_confidence: bool = True
    show_track_id: bool = True

    # ---- Alert (optional) ----
    alert_enabled: bool = False
    alert_cooldown_s: float = 5.0   # seconds between repeated alerts

    # ---- API (optional bonus) ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---- Restricted zones (optional bonus) ----
    # Each zone is a list of (x, y) polygon points in pixel coordinates.
    restricted_zones: list = field(default_factory=list)

    def snapshot_dir(self) -> Path:
        return Path(self.output_dir) / "snapshots"

    def video_dir(self) -> Path:
        return Path(self.output_dir) / "videos"

    def log_dir(self) -> Path:
        return Path(self.output_dir) / "logs"
