"""
Auto-capture kiosk.

Runs a live webcam stream and AUTO-SNAPS a frame the moment it passes the
quality gate for the current requested angle. The student never presses a
button — the screen coaches them ("look straight", "turn slightly left",
"tilt chin down") and the system captures when the shot is good.

This produces the 5-8 multi-angle photos the registration pipeline wants,
including the all-important TOP-ANGLE shot that matches a board-mounted
classroom camera.

Designed to run on a laptop with a normal webcam for development. On the
kiosk it can drive a fullscreen UI; here it uses simple OpenCV windows.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .quality import check_face_crop_quality, pick_primary_face
from .embedder import default_embedder


# The angle sequence we coach the student through. The labels are hints
# stored alongside embeddings; the TOP variants matter most for board cams.
CAPTURE_SEQUENCE = [
    ("front",        "Look straight at the camera"),
    ("slight_left",  "Turn your head slightly LEFT"),
    ("slight_right", "Turn your head slightly RIGHT"),
    ("top",          "Lift the camera / tilt chin DOWN (board angle)"),
    ("top_left",     "Board angle, slight LEFT"),
    ("top_right",    "Board angle, slight RIGHT"),
    ("natural",      "Sit normally, natural expression"),
    ("down_gaze",    "Look slightly DOWN as if at a notebook"),
]


@dataclass
class CaptureResult:
    images_bgr: list          # captured frames (np.ndarray)
    angle_hints: list         # parallel labels
    completed: bool


def _stable_frames_required(fps_hint: int = 15) -> int:
    """Require the quality to hold for ~0.4s so we don't snap a lucky frame."""
    return max(3, int(fps_hint * 0.4))


def run_kiosk_capture(camera_index: int = 0,
                      target_count: int = 8,
                      hold_seconds: float = 0.4,
                      embedder=default_embedder) -> CaptureResult:
    """
    Blocking capture loop. Returns when target_count good shots are gathered
    or the operator presses ESC.

    NOTE: requires a display. For a headless kiosk, swap the cv2.imshow calls
    for your UI framework but keep the auto-snap logic intact.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")

    images, hints = [], []
    seq = CAPTURE_SEQUENCE[:target_count]
    step = 0
    good_streak = 0
    last_capture_t = 0.0
    cooldown = 1.0  # seconds between captures so angles actually change

    try:
        while step < len(seq):
            ok, frame = cap.read()
            if not ok:
                continue

            angle_key, prompt = seq[step]
            faces = embedder.detect(frame)
            face, why = pick_primary_face(faces)

            status = why
            ready = False
            if face is not None:
                q = check_face_crop_quality(frame, face)
                status = q.reason
                ready = q.passed

            if ready and (time.time() - last_capture_t) > cooldown:
                good_streak += 1
            else:
                good_streak = 0

            # Auto-snap when quality has been stable long enough.
            if good_streak >= _stable_frames_required():
                images.append(frame.copy())
                hints.append(angle_key)
                last_capture_t = time.time()
                good_streak = 0
                step += 1

            # ── Operator/student feedback overlay ──
            disp = frame.copy()
            cv2.putText(disp, f"[{step+1}/{len(seq)}] {prompt}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if ready else (0, 165, 255), 2)
            cv2.putText(disp, status, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if face is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(disp, (x1, y1), (x2, y2),
                              (0, 255, 0) if ready else (0, 0, 255), 2)
            cv2.imshow("Registration kiosk (ESC to abort)", disp)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return CaptureResult(images_bgr=images, angle_hints=hints,
                         completed=(step >= len(seq)))
