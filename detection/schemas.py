"""Pydantic schemas for the detection service API.

What this module does
---------------------
- Defines request/response models for session start/stop and results.
- Keeps API types stable so backend integration is predictable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CameraSpec(BaseModel):
    """One camera feed in a room."""

    camera_id: str = Field(..., examples=["cam1"], description="Stable camera name within a room")
    rtsp_url: str = Field(..., examples=["rtsp://user:pass@10.0.0.5:554/stream1"])


class SessionStartRequest(BaseModel):
    """Start (or restart) attendance detection for a room."""

    room_id: str = Field(..., examples=["CSE-LAB-101"])
    section_id: str = Field(..., examples=["11111111-1111-1111-1111-111111111111"])
    cameras: list[CameraSpec] = Field(..., min_length=1, max_length=3)
    start_ts: datetime | None = None
    end_ts: datetime | None = None


class WindowRunRequest(BaseModel):
    """Run one attendance window for a room.

    Backend should call this 3 times per lecture:
    - window_id=1 (start)
    - window_id=2 (middle)
    - window_id=3 (end)
    """

    room_id: str
    section_id: str
    window_id: int = Field(..., ge=1, le=3)
    cameras: list[CameraSpec] = Field(..., min_length=1, max_length=3)
    duration_seconds: int | None = Field(default=None, ge=5, le=120)


class SessionStartResponse(BaseModel):
    status: Literal["started"]
    room_id: str
    section_id: str


class SessionStopRequest(BaseModel):
    room_id: str


class SessionStopResponse(BaseModel):
    status: Literal["stopped"]
    room_id: str


class PresentStudent(BaseModel):
    """A student that has met the 'present' rule for the current session."""

    student_id: str
    roll_no: str | None = None
    name: str | None = None
    best_score: float
    first_seen_ts: datetime
    last_seen_ts: datetime
    cameras_seen: list[str]


class SessionResultResponse(BaseModel):
    room_id: str
    section_id: str
    active: bool
    present: list[PresentStudent]
    unknown_tracks: int = 0
    last_update_ts: datetime | None = None


class WindowStatusResponse(BaseModel):
    room_id: str
    active: bool
    window_id: int | None = None
    ends_at: datetime | None = None

