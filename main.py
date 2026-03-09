import argparse
import logging
import os
import sys
import time
from pathlib import Path
import cv2

from detector import HelmetDetector
from tracker import build_tracker
from analytics import ComplianceAnalytics
from visualizer import Visualizer
from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-Time Helmet Safety Compliance Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", default="0", help="Video path or camera index")
    p.add_argument("--model", default="yolov8n.pt", help="Path to YOLO weights")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--tracker", default="auto")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--snapshot-interval", type=int, default=30)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--custom-model", action="store_true")
    p.add_argument("--show-head-box", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p


def run(args: argparse.Namespace):
    cfg = Settings(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output_dir,
    )

    detector = HelmetDetector(
        model_path=cfg.model_path,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        use_custom_model=args.custom_model,
    )

    tracker = build_tracker(backend=args.tracker)
    visualizer = Visualizer(show_head_box=args.show_head_box)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Source: {source} | {width}×{height} @ {fps:.1f} FPS | {total_frames if total_frames > 0 else '?'} frames")

    log_dir = os.path.join(args.output_dir, "logs")
    analytics = ComplianceAnalytics(output_dir=log_dir, fps=fps)

    writer = None
    if args.save_video:
        video_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        out_path = os.path.join(video_dir, "output_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        logger.info(f"Saving annotated video → {out_path}")

    snapshot_dir = os.path.join(args.output_dir, "snapshots")

    frame_id = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0

    logger.info("Starting detection loop. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of stream.")
            break

        if args.max_frames and frame_id >= args.max_frames:
            logger.info(f"Reached max-frames limit ({args.max_frames}).")
            break

        detections = detector.detect(frame)

        tracked = tracker.update(detections)

        stats = analytics.update(frame_id, tracked)

        if frame_id % args.snapshot_interval == 0 and stats.non_compliant > 0:
            saved = visualizer.make_violation_snapshot(frame, tracked, frame_id, snapshot_dir)
            for snap_path, tid in saved:
                analytics.log_violation(frame_id, tid, snap_path)

        frame = visualizer.draw_detections(frame, tracked)

        now = time.perf_counter()
        display_fps = 0.9 * display_fps + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
        fps_timer = now

        frame = visualizer.draw_hud(
            frame,
            frame_id,
            total=stats.total_persons,
            compliant=stats.compliant,
            non_compliant=stats.non_compliant,
            fps=display_fps,
        )

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Helmet Safety Compliance Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                logger.info("User quit.")
                break

        frame_id += 1

        if frame_id % 100 == 0:
            logger.info(
                f"Frame {frame_id:>6} | "
                f"Persons: {stats.total_persons} | "
                f"OK: {stats.compliant} | "
                f"Violations: {stats.non_compliant} | "
                f"FPS: {display_fps:.1f}"
            )

    cap.release()

    if writer:
        writer.release()

    cv2.destroyAllWindows()

    analytics.save_summary()
    analytics.print_summary()

    logger.info("Done.")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)