"""Lecture window manager.

What this module does
---------------------
- Runs short attendance windows (30–45s) on demand.
- Keeps a per-room, per-lecture accumulator of:
  - windows_seen per student
  - best_confidence
  - total_seen_time

Why it exists
-------------
Your system is for attendance (not surveillance). So we only process video
during a few short windows per lecture.

Backend responsibility
----------------------
Backend triggers windows at the correct time using the timetable mapping.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ._logging import get_logger
from .schemas import CameraSpec
from .worker import WindowWorker

log = get_logger("windows")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StudentAggregate:
    """Aggregated attendance evidence across windows."""

    student_id: str
    windows_seen: set[int] = field(default_factory=set)
    best_confidence: float = 0.0
    total_seen_time: float = 0.0
    first_seen_ts: datetime | None = None
    last_seen_ts: datetime | None = None
    cameras_seen: set[str] = field(default_factory=set)

    def update(self, *, window_id: int, confidence: float, seen_seconds: float, ts: datetime, camera_id: str) -> None:
        self.windows_seen.add(window_id)
        self.best_confidence = max(self.best_confidence, confidence)
        self.total_seen_time += max(0.0, seen_seconds)
        self.cameras_seen.add(camera_id)
        if self.first_seen_ts is None:
            self.first_seen_ts = ts
        self.last_seen_ts = ts


@dataclass
class RoomState:
    room_id: str
    section_id: str
    active_window_id: int | None = None
    window_ends_at: datetime | None = None
    last_update_ts: datetime | None = None
    unknown_tracks: int = 0
    students: dict[str, StudentAggregate] = field(default_factory=dict)


class WindowManager:
    """Runs attendance windows and stores per-room evidence."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rooms: dict[str, RoomState] = {}
        self._workers: dict[str, WindowWorker] = {}

    def run_window(
        self,
        *,
        room_id: str,
        section_id: str,
        window_id: int,
        cameras: list[CameraSpec],
        duration_seconds: int,
    ) -> None:
        """Start a window worker for a room.

        If a previous window is running for the room, it is stopped.
        """

        if window_id not in (1, 2, 3):
            raise ValueError("window_id must be 1, 2, or 3")

        now = _utc_now()
        ends_at = now + timedelta(seconds=duration_seconds)

        with self._lock:
            # Replace any running worker.
            old = self._workers.pop(room_id, None)
            if old is not None:
                old.stop()

            state = self._rooms.get(room_id)
            if state is None or state.section_id != section_id:
                state = RoomState(room_id=room_id, section_id=section_id)
                self._rooms[room_id] = state

            state.active_window_id = window_id
            state.window_ends_at = ends_at

            worker = WindowWorker(
                room_id=room_id,
                section_id=section_id,
                window_id=window_id,
                cameras=cameras,
                on_match=self._on_match,
                on_unknown=self._on_unknown,
                on_heartbeat=self._on_heartbeat,
            )
            self._workers[room_id] = worker

        worker.start(duration_seconds=duration_seconds)
        log.info("window started | room=%s section=%s window=%d seconds=%d", room_id, section_id, window_id, duration_seconds)

    def _on_match(self, *, room_id: str, section_id: str, window_id: int, student_id: str, confidence: float, seen_seconds: float, ts: datetime, camera_id: str) -> None:
        """Callback invoked by workers when a student is confidently recognized."""

        with self._lock:
            state = self._rooms.get(room_id)
            if state is None or state.section_id != section_id:
                return
            agg = state.students.get(student_id)
            if agg is None:
                agg = StudentAggregate(student_id=student_id)
                state.students[student_id] = agg
            agg.update(window_id=window_id, confidence=confidence, seen_seconds=seen_seconds, ts=ts, camera_id=camera_id)
            state.last_update_ts = ts

    def _on_unknown(self, *, room_id: str, section_id: str, ts: datetime) -> None:
        """Callback invoked by workers when a track does not match any student."""

        with self._lock:
            state = self._rooms.get(room_id)
            if state is None or state.section_id != section_id:
                return
            state.unknown_tracks += 1
            state.last_update_ts = ts

    def _on_heartbeat(self, *, room_id: str, section_id: str, window_id: int, ts: datetime) -> None:
        """Callback invoked periodically so the API can show liveness."""

        with self._lock:
            state = self._rooms.get(room_id)
            if state is None or state.section_id != section_id:
                return
            state.last_update_ts = ts

    def room_state(self, room_id: str) -> RoomState | None:
        """Return current room state for API read."""

        with self._lock:
            return self._rooms.get(room_id)

    def room_window_status(self, room_id: str) -> tuple[bool, int | None, datetime | None]:
        """Return whether a window is currently running for room_id."""

        with self._lock:
            state = self._rooms.get(room_id)
            worker = self._workers.get(room_id)
            active = worker is not None and worker.is_active()
            win = state.active_window_id if state else None
            ends = state.window_ends_at if state else None
            return active, win, ends

