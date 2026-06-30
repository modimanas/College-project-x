"""FastAPI surface for the detection service.

What this service does
----------------------
- Receives start/stop commands from the backend (timetable owner).
- Runs room-level workers that connect to RTSP feeds and compute attendance.
- Exposes results for backend to poll.

What it does NOT do
-------------------
- Timetable logic (which section should run now).
- Writing attendance to the main DB.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ._logging import get_logger
from .config import MODEL, RUNTIME
from .schemas import (
    SessionResultResponse,
    WindowRunRequest,
    WindowStatusResponse,
)
from .window_manager import WindowManager

log = get_logger("api")


_windows = WindowManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "startup | model_pack=%s providers=%s det_size=%s detect_every_n=%s",
        MODEL.pack_name,
        MODEL.providers,
        MODEL.det_size,
        RUNTIME.detect_every_n_frames,
    )
    yield
    # Best-effort stop sessions on shutdown.
    log.info("shutdown")


app = FastAPI(title="Attendance — Detection", lifespan=lifespan)


@app.get("/health")
def health():
    """Health check endpoint for monitoring."""

    return {
        "status": "ok",
        "model_pack": MODEL.pack_name,
        "providers": MODEL.providers,
        "det_size": MODEL.det_size,
    }


@app.post("/window/run")
def window_run(req: WindowRunRequest):
    """Run one attendance window for a room (30–45 seconds)."""

    seconds = req.duration_seconds or RUNTIME.window_seconds
    _windows.run_window(
        room_id=req.room_id,
        section_id=req.section_id,
        window_id=req.window_id,
        cameras=req.cameras,
        duration_seconds=seconds,
    )
    return {"status": "running", "room_id": req.room_id, "window_id": req.window_id, "seconds": seconds}


@app.get("/window/status", response_model=WindowStatusResponse)
def window_status(room_id: str):
    """Return whether the room currently has a running window."""

    active, window_id, ends_at = _windows.room_window_status(room_id)
    return WindowStatusResponse(room_id=room_id, active=active, window_id=window_id, ends_at=ends_at)


@app.get("/session/result", response_model=SessionResultResponse)
def session_result(room_id: str):
    """Return current attendance results for a room."""

    state = _windows.room_state(room_id)
    if state is None:
        raise HTTPException(404, f"no room state for room_id={room_id}")

    active, _, _ = _windows.room_window_status(room_id)

    present = []
    for agg in state.students.values():
        if len(agg.windows_seen) < RUNTIME.min_windows_present:
            continue
        # roll_no/name are not returned yet (we only have student_id in index).
        present.append(
            {
                "student_id": agg.student_id,
                "roll_no": None,
                "name": None,
                "best_score": agg.best_confidence,
                "first_seen_ts": agg.first_seen_ts,
                "last_seen_ts": agg.last_seen_ts,
                "cameras_seen": sorted(agg.cameras_seen),
            }
        )

    return SessionResultResponse(
        room_id=state.room_id,
        section_id=state.section_id,
        active=active,
        present=present,
        unknown_tracks=state.unknown_tracks,
        last_update_ts=state.last_update_ts,
    )
