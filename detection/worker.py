"""Attendance window worker.

This module implements your requested design:

- Attendance runs only during short windows (30–45s).
- Each window uses per-camera tracking (track IDs local to that camera+window).
- No cross-camera tracking. We merge only by ``student_id``.

Production note
---------------
We keep the tracker interface small so you can swap in a real ByteTrack
implementation later. The default tracker is IoU-based.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from registration.embedder import Embedder

from ._logging import get_logger
from .config import MODEL, RUNTIME
from .matcher import SectionMatcher
from .rtsp import RtspReader
from .schemas import CameraSpec
from .tracker import IoUTracker

log = get_logger("worker")


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""

    return datetime.now(timezone.utc)


OnMatch = Callable[
    [
        # keyword-only in practice via call sites
    ],
    None,
]


class WindowWorker:
    """Runs attendance processing for one room during one window."""

    def __init__(
        self,
        *,
        room_id: str,
        section_id: str,
        window_id: int,
        cameras: list[CameraSpec],
        on_match: Callable[..., None],
        on_unknown: Callable[..., None],
        on_heartbeat: Callable[..., None],
    ):
        self.room_id = room_id
        self.section_id = section_id
        self.window_id = window_id
        self.cameras = cameras

        self._on_match = on_match
        self._on_unknown = on_unknown
        self._on_heartbeat = on_heartbeat

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Each window owns its matcher+embedder lifecycle.
        self._embedder = Embedder(MODEL)
        self._matcher = SectionMatcher()

        # Tracking state (local to each camera, reset every window).
        self._trackers = {
            c.camera_id: IoUTracker(iou_threshold=RUNTIME.track_iou_threshold)
            for c in cameras
        }

    def is_active(self) -> bool:
        """True if the worker thread is running."""

        return self._thread is not None and self._thread.is_alive() and (not self._stop.is_set())

    def start(self, *, duration_seconds: int) -> None:
        """Start the worker for a fixed duration."""

        if self.is_active():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            kwargs={"duration_seconds": duration_seconds},
            name=f"window-{self.room_id}-{self.window_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker early."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self._matcher.close()
        except Exception:
            pass

    def _run(self, *, duration_seconds: int) -> None:
        """Run the attendance loop for a fixed window duration."""

        start_ts = time.time()
        end_ts = start_ts + duration_seconds

        # Load section index once per window.
        self._matcher.load_section(self.section_id)

        # Reset trackers at window start.
        for tr in self._trackers.values():
            tr.reset()

        readers = [RtspReader(rtsp_url=c.rtsp_url, camera_id=c.camera_id) for c in self.cameras]
        iters = {
            c.camera_id: iter(r.frames(stop_flag=self._stop.is_set))
            for c, r in zip(self.cameras, readers)
        }
        frame_counters = {c.camera_id: 0 for c in self.cameras}

        try:
            while (not self._stop.is_set()) and (time.time() < end_ts):
                now_dt = _utc_now()
                self._on_heartbeat(room_id=self.room_id, section_id=self.section_id, window_id=self.window_id, ts=now_dt)

                progressed = False
                for cam in self.cameras:
                    if self._stop.is_set() or time.time() >= end_ts:
                        break

                    cid = cam.camera_id
                    try:
                        frame = next(iters[cid])
                    except StopIteration:
                        continue

                    progressed = True
                    frame_counters[cid] += 1
                    if frame_counters[cid] % RUNTIME.detect_every_n_frames != 0:
                        continue

                    self._process_frame(camera_id=cid, image_bgr=frame.image_bgr, ts=frame.ts)

                if not progressed:
                    time.sleep(0.05)
        finally:
            for r in readers:
                r.close()
            self._matcher.close()

    def _process_frame(self, *, camera_id: str, image_bgr: np.ndarray, ts: float) -> None:
        """Process one frame:

        - Detect faces.
        - Update camera-local tracker.
        - Embed/match on a cooldown per track.
        - When a track has enough confident seen time, emit a match event.
        """

        try:
            faces = self._embedder.detect(image_bgr)
        except Exception as exc:
            log.warning("detect failed | room=%s cam=%s err=%s", self.room_id, camera_id, exc)
            return

        dets = []
        for f in faces:
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            dets.append(np.asarray(bbox, dtype=np.float32))

        tracks = self._trackers[camera_id].update(detections_xyxy=dets, ts=ts)

        for tr in tracks:
            # Recognition throttling per track.
            if (ts - tr.last_embed_ts) < RUNTIME.embed_cooldown_s:
                continue

            # Find the face object corresponding to this track bbox (simple IoU best-match).
            face = None
            best_iou = 0.0
            for f in faces:
                bbox = np.asarray(getattr(f, "bbox", None), dtype=np.float32)
                if bbox.shape[0] != 4:
                    continue
                # Compute IoU vs track bbox.
                x1 = max(float(bbox[0]), float(tr.bbox[0]))
                y1 = max(float(bbox[1]), float(tr.bbox[1]))
                x2 = min(float(bbox[2]), float(tr.bbox[2]))
                y2 = min(float(bbox[3]), float(tr.bbox[3]))
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if inter <= 0:
                    continue
                area_a = max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))
                area_b = max(0.0, float(tr.bbox[2] - tr.bbox[0])) * max(0.0, float(tr.bbox[3] - tr.bbox[1]))
                denom = area_a + area_b - inter
                if denom <= 0:
                    continue
                iou = inter / denom
                if iou > best_iou:
                    best_iou = iou
                    face = f

            if face is None:
                continue

            tr.last_embed_ts = ts

            try:
                emb = self._embedder.embed_face(face, image_bgr=image_bgr)
            except Exception:
                continue

            m = self._matcher.match_one(self.section_id, emb.vector, threshold=RUNTIME.match_threshold)
            now_dt = _utc_now()

            if m.student_id is None:
                self._on_unknown(room_id=self.room_id, section_id=self.section_id, ts=now_dt)
                continue

            tr.last_student_id = m.student_id
            tr.last_score = m.score

            # Only count for the window if the track has been visible long enough.
            if tr.seen_seconds >= RUNTIME.min_seen_seconds_in_window:
                self._on_match(
                    room_id=self.room_id,
                    section_id=self.section_id,
                    window_id=self.window_id,
                    student_id=m.student_id,
                    confidence=m.score,
                    seen_seconds=tr.seen_seconds,
                    ts=now_dt,
                    camera_id=camera_id,
                )

