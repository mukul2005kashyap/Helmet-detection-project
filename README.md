# 🪖 Real-Time Helmet Safety Compliance Detection System

A computer vision pipeline that detects helmet compliance in industrial and warehouse environments using YOLOv8, OpenCV, and ByteTrack.

---

## 📌 Table of Contents
1. [Architecture](#architecture)
2. [Approach](#approach)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Outputs](#outputs)
7. [Bonus Features](#bonus-features)
8. [Limitations](#limitations)

---

## Architecture

```
Video Source (file / webcam)
        │
        ▼
┌──────────────────┐
│  HelmetDetector  │  YOLOv8 → detects persons (+ helmets if custom model)
│  (detector.py)   │  → estimates head region
│                  │  → classifies Compliant / Non-Compliant
└────────┬─────────┘
         │  list[Detection]
         ▼
┌──────────────────┐
│  Tracker         │  ByteTrack (or fallback IoU tracker)
│  (tracker.py)    │  → assigns stable track IDs across frames
└────────┬─────────┘
         │  list[(Detection, track_id)]
         ▼
┌──────────────────┐        ┌────────────────────────┐
│  Analytics       │───────▶│  CSV Logs              │
│  (analytics.py)  │        │  frame_compliance_log  │
│                  │        │  track_summary_log     │
│                  │        │  violation_log         │
└────────┬─────────┘        └────────────────────────┘
         │
         ▼
┌──────────────────┐        ┌──────────────────┐
│  Visualizer      │───────▶│  Output Video    │
│  (visualizer.py) │        │  Violation Snaps │
└──────────────────┘        └──────────────────┘
```

---

## Approach

### Detection

Two modes are supported:

**Base model** (`yolov8n.pt` / any COCO-trained model):
- Detects persons using class ID 0.
- Estimates head region as the top 25% of each person bounding box.
- Applies an HSV colour-range heuristic to check for hard-hat colours (yellow, white, orange, red, blue) in the head region.
- Simple, zero-shot — no training required.

**Custom model** (fine-tuned on helmet dataset):
- Detects `person`, `helmet`, and optionally `no_helmet` classes.
- Spatially associates each helmet detection with the nearest person's head region via IoU matching.
- Significantly more accurate — recommended for production.

### Helmet Association Logic

```
For each detected person:
  1. Estimate head bbox = top 25% of person bbox
  2. Compute IoU between head bbox and all detected helmet bboxes
  3. If best IoU ≥ 0.10  →  Compliant
     Else                →  Non-Compliant
```

When using the base model without helmet class, the colour heuristic fires:
```
  Convert head ROI → HSV
  Check % pixels matching hard-hat colour ranges
  If ratio ≥ 15%  →  Compliant (confidence ∝ ratio)
```

### Tracking

ByteTrack (via `supervision`) is used by default for robust multi-object tracking across frames. The system falls back to a built-in IoU tracker if `supervision` is not installed.

Tracks are maintained with a `max_lost` parameter — a track is dropped after 30 consecutive frames without a matching detection.

---

## Project Structure

```
helmet-safety-detection/
├── main.py              # Entry point — CLI
├── detector.py          # YOLOv8 detection + helmet association
├── tracker.py           # ByteTrack / IoU tracker wrapper
├── analytics.py         # Per-frame + per-track compliance stats, CSV logging
├── visualizer.py        # OpenCV drawing — boxes, labels, HUD, snapshots
├── api.py               # (Bonus) FastAPI REST endpoint
├── dashboard.py         # (Bonus) Streamlit analytics dashboard
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py      # Centralised configuration
├── models/              # Place custom .pt weights here
├── tests/
│   ├── test_detector.py
│   └── test_analytics.py
└── outputs/             # Auto-created at runtime
    ├── logs/
    │   ├── frame_compliance_log.csv
    │   ├── track_summary_log.csv
    │   └── violation_log.csv
    ├── snapshots/        # Violation crop images
    └── videos/           # Annotated output video
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/<your-username>/helmet-safety-detection.git
cd helmet-safety-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Download a custom helmet detection model

Pre-trained helmet datasets (e.g., Safety Helmet Detection on Roboflow, SHWD dataset) can be used to fine-tune YOLOv8:

```bash
# Fine-tune on a custom dataset (example)
yolo detect train data=helmet_dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Place the resulting `best.pt` inside the `models/` directory.

---

## Usage

### Process a video file

```bash
python main.py --source path/to/video.mp4
```

### Use webcam

```bash
python main.py --source 0
```

### Use custom model & save output video

```bash
python main.py \
  --source video.mp4 \
  --model models/best.pt \
  --custom-model \
  --save-video \
  --conf 0.45
```

### Headless mode (no display, useful on servers)

```bash
python main.py --source video.mp4 --no-display --save-video
```

### All CLI options

```
--source             Video file path or camera index (default: 0)
--model              YOLO weights path (default: yolov8n.pt)
--conf               Detection confidence threshold (default: 0.40)
--iou                NMS IoU threshold (default: 0.45)
--tracker            Tracker backend: auto | bytetrack | iou (default: auto)
--save-video         Save annotated output MP4
--no-display         Disable live preview window
--snapshot-interval  Save violation snapshot every N frames (default: 30)
--output-dir         Root output directory (default: outputs)
--custom-model       Use model with explicit helmet/no-helmet classes
--show-head-box      Draw the estimated head region box
--max-frames         Limit processing to N frames (0 = unlimited)
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/videos/output_annotated.mp4` | Annotated video with bounding boxes and HUD |
| `outputs/logs/frame_compliance_log.csv` | Per-frame: total, compliant, non-compliant, rate |
| `outputs/logs/track_summary_log.csv` | Per-track: compliance rate, duration, violation flag |
| `outputs/logs/violation_log.csv` | Timestamped log of every saved snapshot |
| `outputs/snapshots/violation_*.jpg` | Cropped images of non-compliant workers |

### CSV Schema — `frame_compliance_log.csv`

| Column | Type | Description |
|--------|------|-------------|
| `frame_id` | int | Frame index (0-based) |
| `timestamp_s` | float | Timestamp in seconds |
| `total_persons` | int | Persons detected |
| `compliant` | int | Persons wearing helmets |
| `non_compliant` | int | Persons without helmets |
| `compliance_rate_%` | float | `compliant / total × 100` |

---

## Bonus Features

### REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Upload an image and receive JSON compliance results:

```bash
curl -X POST http://localhost:8000/detect \
     -F "file=@frame.jpg" | python -m json.tool
```

Example response:
```json
{
  "total_persons": 3,
  "compliant": 2,
  "non_compliant": 1,
  "compliance_rate": 0.667,
  "persons": [
    {"person_id": 0, "bbox": [120, 80, 240, 420], "confidence": 0.87, "compliant": true, "helmet_confidence": 0.72},
    ...
  ],
  "processing_time_ms": 34.2
}
```

### Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Visualises compliance timeline, per-track bar chart, and violation snapshot gallery from saved logs.

---

## Limitations

- **Base model accuracy**: The HSV colour heuristic for helmet detection is a rough approximation and will misclassify similarly coloured hair or backgrounds. A custom fine-tuned model is strongly recommended for real deployment.

- **Occlusion**: When workers are heavily occluded or overlapping, both detection and tracking accuracy degrade.

- **Head region assumption**: Estimating the head as the top 25% of the bounding box fails for crouching, bending, or partially visible persons.

- **Lighting sensitivity**: The colour heuristic is sensitive to lighting changes (very dark or overexposed environments).

- **Small objects**: YOLOv8n (nano) struggles with small or distant subjects — switch to `yolov8s` or `yolov8m` for better accuracy at the cost of speed.

- **CPU speed**: Running on CPU at 1080p will typically achieve 3–8 FPS with YOLOv8n. Reduce resolution or use a GPU for real-time performance.

---

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## License

MIT License — free to use and modify.
