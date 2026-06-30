"""
FastAPI surface for registration (hardened).

Endpoints:
  GET  /                       serves the selfie capture web page
  POST /register               multipart: section_id, roll_no, name, files[]
  POST /sections/{id}/reload   rebuild a section's FAISS index from DB
  GET  /health

Hardening vs the prototype:
  - lifespan handler (not deprecated on_event)
  - file count cap + per-file size cap
  - images downscaled before processing (memory + speed)
  - structured logging
  - clear 4xx errors
"""

import os
import re
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ._logging import get_logger
from .store import get_store
from .index_manager import IndexManager
from .service import RegistrationService
from .embedder import default_embedder
from .quality import check_face_crop_quality, pick_primary_face

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)

log = get_logger("api")

# Limits
MAX_FILES = int(os.getenv("ATTEND_MAX_FILES", "12"))
MAX_FILE_BYTES = int(os.getenv("ATTEND_MAX_FILE_BYTES", str(8 * 1024 * 1024)))  # 8MB
MAX_DIM = int(os.getenv("ATTEND_MAX_DIM", "1280"))  # downscale longest side

_state = {}

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    index = IndexManager(store)
    service = RegistrationService(store, index, embedder=default_embedder)
    _state["store"] = store
    _state["index"] = index
    _state["service"] = service
    log.info(
        "startup complete | model_pack=%s | store=%s | providers=%s",
        default_embedder.pack_name,
        type(store).__name__,
        getattr(default_embedder, "providers", None),
    )
    yield
    store.close()
    log.info("shutdown complete")


app = FastAPI(title="Attendance — Registration", lifespan=lifespan)


def _decode_and_downscale(file_bytes: bytes) -> np.ndarray:
    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_FILE_BYTES} bytes")
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode an uploaded image")
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > MAX_DIM:
        scale = MAX_DIM / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model_pack": default_embedder.pack_name}


@app.post("/preview")
async def preview_shot(file: UploadFile = File(...)):
    """
    Run SCRFD on one captured frame and return detection + quality metrics.
    Called immediately after each pose capture so the user sees per-shot
    feedback (border colour + score badge) before submitting.
    Detection only — no ArcFace embedding, so it's fast (~50-150 ms on CPU).
    """
    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        return {"detected": False, "reason": "file too large"}

    try:
        img = _decode_and_downscale(raw)
    except Exception:
        return {"detected": False, "reason": "could not decode image"}

    # Run SCRFD in a threadpool so we don't block the async event loop.
    def _run():
        faces = default_embedder.detect(img)
        face, why = pick_primary_face(faces)
        if face is None:
            return {"detected": False, "reason": why}

        q  = check_face_crop_quality(img, face)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]

        return {
            "detected":   True,
            "det_score":  round(float(getattr(face, "det_score", 0.0)), 3),
            "passed":     q.passed,
            "reason":     q.reason,
            "blur":       round(q.blur_variance, 1),
            "brightness": round(q.brightness, 1),
            # Bounding box as fractions of image size (0-1) so the
            # frontend can scale it to any display size.
            "bbox_pct": [
                round(max(0.0, x1 / w), 4),
                round(max(0.0, y1 / h), 4),
                round(min(1.0, x2 / w), 4),
                round(min(1.0, y2 / h), 4),
            ],
        }

    return await run_in_threadpool(_run)


@app.get("/branches")
def list_branches():
    """Return all branches so the registration form can populate its dropdown."""
    return {"branches": _state["store"].list_branches()}


@app.get("/sections/lookup")
def lookup_section(branch: str, year: int, section: str):
    """Resolve a human-readable (branch, year, section) tuple to a section UUID."""
    section_id = _state["store"].lookup_section(branch, year, section)
    if not section_id:
        raise HTTPException(
            404,
            f"Section not found: branch={branch} year={year} section={section}. "
            "Ask your admin to run the seed script."
        )
    return {"section_id": section_id}


@app.get("/")
def index_page():
    path = os.path.join(WEB_DIR, "register.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(404, "web page not found")


@app.post("/register")
async def register(
    section_id: str = Form(...),
    roll_no: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    # ── Input validation ────────────────────────────────────────────────
    section_id = section_id.strip()
    roll_no    = roll_no.strip()
    name       = name.strip()

    if not _UUID_RE.match(section_id.lower()):
        raise HTTPException(400, "section_id must be a valid UUID")
    if not (1 <= len(roll_no) <= 50):
        raise HTTPException(400, "roll_no must be 1–50 characters")
    if not (2 <= len(name) <= 100):
        raise HTTPException(400, "name must be 2–100 characters")

    if len(files) < 5:
        raise HTTPException(400, "upload at least 5 photos")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"too many photos (max {MAX_FILES})")

    images = []
    for f in files:
        images.append(_decode_and_downscale(await f.read()))

    log.info("register request | section=%s roll=%s photos=%d",
             section_id, roll_no, len(images))

    report = _state["service"].register(
        section_id=section_id, roll_no=roll_no, name=name, images_bgr=images
    )

    log.info("register result  | roll=%s success=%s accepted=%d reason=%s",
             roll_no, report.success, report.accepted_count, report.reason)

    return {
        "student_id": report.student_id,
        "success": report.success,
        "reason": report.reason,
        "accepted_count": report.accepted_count,
        "section_index_size": report.section_index_size,
        "photos": [
            {
                "index": p.index,
                "accepted": p.accepted,
                "reason": p.reason,
                "det_score": round(p.det_score, 3),
                "blur": round(p.blur, 1),
                "brightness": round(p.brightness, 1),
            }
            for p in report.photo_reports
        ],
    }


@app.post("/sections/{section_id}/reload")
def reload_section(section_id: str):
    size = _state["index"].reload_section(section_id)
    log.info("manual index reload | section=%s size=%d", section_id, size)
    return {"section_id": section_id, "index_size": size}
