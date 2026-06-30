"""Lightweight per-camera tracker (ByteTrack-style scaffold).

What this module does
---------------------
- Maintains short-lived tracks within a single camera.
- Assigns a stable ``track_id`` to detections across frames.

Why not full ByteTrack here?
----------------------------
ByteTrack is a specific published algorithm with multiple implementations.
For this repo we provide a clean *interface* and a simple IoU tracker that:
  - is dependency-free
  - is CPU-friendly
  - is easy to replace with a real ByteTrack implementation later

This keeps your production API stable while you upgrade the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""

    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


@dataclass
class Track:
    """One tracked face within a single camera."""

    track_id: int
    bbox: np.ndarray
    last_ts: float

    # Recognition throttling
    last_embed_ts: float = 0.0
    last_student_id: str | None = None
    last_score: float = 0.0
    # "Seen" accumulation per window (seconds)
    seen_seconds: float = 0.0
    _last_seen_update_ts: float = 0.0

    def touch(self, *, bbox: np.ndarray, ts: float) -> None:
        """Update track location + last seen time."""

        self.bbox = bbox
        self.last_ts = ts

    def add_seen_time(self, ts: float) -> None:
        """Accumulate approximate visible time for this track."""

        if self._last_seen_update_ts <= 0.0:
            self._last_seen_update_ts = ts
            return
        dt = max(0.0, ts - self._last_seen_update_ts)
        # Cap dt to avoid huge jumps if frames stall.
        self.seen_seconds += min(dt, 1.0)
        self._last_seen_update_ts = ts


class IoUTracker:
    """A simple IoU tracker suitable for short attendance windows."""

    def __init__(self, *, iou_threshold: float = 0.3, max_age_s: float = 2.0):
        self._iou_threshold = iou_threshold
        self._max_age_s = max_age_s
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def reset(self) -> None:
        """Drop all tracks (start of a new attendance window)."""

        self._tracks.clear()
        self._next_id = 1

    def update(self, *, detections_xyxy: list[np.ndarray], ts: float) -> list[Track]:
        """Associate detections to tracks and return updated tracks.

        Args:
            detections_xyxy: List of face boxes for this frame.
            ts: Unix timestamp.
        """

        # Retire old tracks.
        dead = [tid for tid, tr in self._tracks.items() if (ts - tr.last_ts) > self._max_age_s]
        for tid in dead:
            self._tracks.pop(tid, None)

        assigned_tracks: set[int] = set()
        updated: list[Track] = []

        for det in detections_xyxy:
            best_tid = None
            best_iou = 0.0
            for tid, tr in self._tracks.items():
                if tid in assigned_tracks:
                    continue
                iou = _iou_xyxy(det, tr.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self._iou_threshold:
                tr = self._tracks[best_tid]
                tr.touch(bbox=det, ts=ts)
                tr.add_seen_time(ts)
                assigned_tracks.add(best_tid)
                updated.append(tr)
            else:
                tid = self._next_id
                self._next_id += 1
                tr = Track(track_id=tid, bbox=det, last_ts=ts)
                tr.add_seen_time(ts)
                self._tracks[tid] = tr
                assigned_tracks.add(tid)
                updated.append(tr)

        return updated

