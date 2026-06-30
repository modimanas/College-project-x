"""RTSP capture helpers.

What this module does
---------------------
- Opens an RTSP stream via OpenCV.
- Provides a safe frame iterator with reconnect/backoff.

Why it exists
-------------
RTSP feeds are flaky in production (network jitter, camera resets). We want
robust behavior: keep trying, log errors, and never crash the whole service.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from ._logging import get_logger

log = get_logger("rtsp")


@dataclass(frozen=True)
class Frame:
    """A decoded video frame with metadata."""

    ts: float
    image_bgr: "cv2.typing.MatLike"


class RtspReader:
    """A resilient RTSP reader.

    What this class does
    --------------------
    - Keeps a VideoCapture open.
    - Reconnects when reads fail.
    - Yields decoded frames with a monotonic timestamp.
    """

    def __init__(self, *, rtsp_url: str, camera_id: str, reconnect_backoff_s: float = 1.0):
        self._url = rtsp_url
        self._camera_id = camera_id
        self._reconnect_backoff_s = max(0.2, reconnect_backoff_s)
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        """Open the RTSP stream and return the capture handle."""

        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"could not open RTSP stream for camera_id={self._camera_id}")
        return cap

    def close(self) -> None:
        """Close the underlying VideoCapture if open."""

        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def frames(self, *, stop_flag) -> "iter[Frame]":
        """Yield frames until stop_flag() becomes True.

        Args:
            stop_flag: A callable returning True when this reader should stop.
        """

        while not stop_flag():
            if self._cap is None:
                try:
                    self._cap = self._open()
                    log.info("rtsp opened | camera=%s", self._camera_id)
                except Exception as exc:
                    log.warning("rtsp open failed | camera=%s err=%s", self._camera_id, exc)
                    time.sleep(self._reconnect_backoff_s)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.warning("rtsp read failed | camera=%s (reconnecting)", self._camera_id)
                self.close()
                time.sleep(self._reconnect_backoff_s)
                continue

            yield Frame(ts=time.time(), image_bgr=frame)

