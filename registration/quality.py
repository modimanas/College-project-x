"""
Quality gates for registration photos.

Philosophy: it is FAR better to reject a marginal photo at registration
than to store a weak embedding that causes false rejects forever after.
We are strict here on purpose. At DETECTION time we are lenient (multi-frame
accumulation handles bad angles); at REGISTRATION time we are strict.

Each gate returns (passed: bool, reason: str). The first failure short-circuits.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import QUALITY


@dataclass
class QualityResult:
    passed: bool
    reason: str = "ok"
    # diagnostics (useful for the kiosk UI to coach the user)
    blur_variance: float = 0.0
    brightness: float = 0.0
    face_pixels: int = 0
    det_score: float = 0.0
    tilt_degrees: float = 0.0


def _laplacian_variance(gray: np.ndarray) -> float:
    """Higher variance = sharper image. Low variance = blurry."""
    val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return val if np.isfinite(val) else 0.0


def _mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _eye_tilt_degrees(kps: np.ndarray) -> float:
    """
    Head roll from the eye line.
    kps order (insightface): [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    if kps is None or kps.shape[0] < 2:
        return 0.0                       # no landmarks → skip tilt gate
    left_eye, right_eye = kps[0], kps[1]
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    if abs(dx) < 1e-4:
        return 90.0                      # vertical eye line → extreme tilt
    return abs(math.degrees(math.atan2(dy, dx)))


def check_face_crop_quality(image_bgr: np.ndarray, face) -> QualityResult:
    """
    Run all gates against a single detected `face` (an insightface Face object
    with .bbox, .kps, .det_score) found in `image_bgr`.

    `face` is passed in (not detected here) so detection runs once upstream.
    """
    x1, y1, x2, y2 = face.bbox.astype(int)
    bbox_w = max(0, x2 - x1)
    bbox_h = max(0, y2 - y1)
    face_short_side = min(bbox_w, bbox_h)

    res = QualityResult(passed=False)
    res.face_pixels = int(face_short_side)
    res.det_score = float(getattr(face, "det_score", 0.0))

    # Gate 1: detection confidence
    if res.det_score < QUALITY.min_det_score:
        res.reason = f"low detection confidence ({res.det_score:.2f})"
        return res

    # Gate 2: face large enough to embed reliably
    if face_short_side < QUALITY.min_face_pixels:
        res.reason = f"face too small ({face_short_side}px)"
        return res

    # Crop the face region (clamped to image bounds) for pixel-level checks
    h, w = image_bgr.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        res.reason = "empty face crop"
        return res

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Gate 3: blur
    res.blur_variance = _laplacian_variance(gray)
    if res.blur_variance < QUALITY.min_blur_variance:
        res.reason = f"image too blurry (var {res.blur_variance:.0f})"
        return res

    # Gate 4: brightness
    res.brightness = _mean_brightness(gray)
    if res.brightness < QUALITY.min_brightness:
        res.reason = f"too dark ({res.brightness:.0f})"
        return res
    if res.brightness > QUALITY.max_brightness:
        res.reason = f"too bright / overexposed ({res.brightness:.0f})"
        return res

    # Gate 5: head tilt (landmark reliability)
    if face.kps is not None:
        res.tilt_degrees = _eye_tilt_degrees(face.kps)
        if res.tilt_degrees > QUALITY.max_tilt_degrees:
            res.reason = f"head tilted too much ({res.tilt_degrees:.0f} deg)"
            return res

    res.passed = True
    res.reason = "ok"
    return res


def pick_primary_face(faces: list):
    """
    A registration photo MUST contain exactly one usable subject.
    If multiple faces are present we reject (could be a group photo /
    someone in the background) rather than guessing.

    Returns (face | None, reason).
    """
    if len(faces) == 0:
        return None, "no face detected"
    if len(faces) > 1:
        # Allow it only if one face clearly dominates (others tiny / background).
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        biggest = faces_sorted[0]
        second = faces_sorted[1]
        a1 = (biggest.bbox[2] - biggest.bbox[0]) * (biggest.bbox[3] - biggest.bbox[1])
        a2 = (second.bbox[2] - second.bbox[0]) * (second.bbox[3] - second.bbox[1])
        if a2 > 0.25 * a1:
            return None, f"multiple faces detected ({len(faces)})"
        return biggest, "ok"
    return faces[0], "ok"
