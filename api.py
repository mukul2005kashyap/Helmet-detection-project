import io
import logging
import time
from typing import Optional

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install FastAPI: pip install fastapi uvicorn python-multipart")

from detector import HelmetDetector
from config.settings import Settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Helmet Safety Compliance API",
    description="Real-time helmet compliance detection via REST",
    version="1.0.0",
)

_settings = Settings()
_detector: Optional[HelmetDetector] = None

def get_detector() -> HelmetDetector:
    global _detector
    if _detector is None:
        _detector = HelmetDetector(
            model_path=_settings.model_path,
            conf_threshold=_settings.conf_threshold,
            iou_threshold=_settings.iou_threshold,
        )
    return _detector

class PersonResult(BaseModel):
    person_id: int
    bbox: list[int]
    confidence: float
    compliant: bool
    helmet_confidence: float

class DetectionResponse(BaseModel):
    total_persons: int
    compliant: int
    non_compliant: int
    compliance_rate: float
    persons: list[PersonResult]
    processing_time_ms: float

@app.get("/health")
def health():
    return {"status": "ok", "model": _settings.model_path}

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    t0 = time.perf_counter()
    detections = get_detector().detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    compliant_count     = sum(1 for d in detections if d.compliant)
    non_compliant_count = len(detections) - compliant_count
    rate = compliant_count / len(detections) if detections else 1.0

    persons = [
        PersonResult(
            person_id=i,
            bbox=list(d.bbox),
            confidence=round(d.confidence, 3),
            compliant=d.compliant,
            helmet_confidence=round(d.helmet_conf, 3),
        )
        for i, d in enumerate(detections)
    ]

    return DetectionResponse(
        total_persons=len(detections),
        compliant=compliant_count,
        non_compliant=non_compliant_count,
        compliance_rate=round(rate, 3),
        persons=persons,
        processing_time_ms=round(elapsed_ms, 2),
    )