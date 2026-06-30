"""Central configuration for detection.

What this module does
---------------------
- Picks a model pack + execution providers (GPU if available, else CPU).
- Defines runtime knobs for frame sampling and attendance decision rules.

Why it exists
-------------
We want *one* codepath that can run on both GPU and CPU machines.
The backend should be able to deploy the same image everywhere and let the
service auto-select the best acceleration it can find.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Model pack configuration used by insightface FaceAnalysis."""

    pack_name: str
    providers: list[str] = field(default_factory=list)
    ctx_id: int = 0
    det_size: tuple[int, int] = (640, 640)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime parameters that control compute vs. accuracy."""

    # Lecture windows: attendance runs only in these short bursts.
    # Window duration should cover the room once (students looking up).
    window_seconds: int = 45

    # Per lecture, you typically run 3 windows: start/mid/end.
    # Present rule: a student must appear in at least this many windows.
    min_windows_present: int = 2

    # Decoder runs continuously within a window; detection runs every N frames.
    detect_every_n_frames: int = 10

    # Per-track embedding cooldown (seconds). Avoid embedding every frame.
    embed_cooldown_s: float = 1.5

    # Recognition acceptance.
    match_threshold: float = 0.50

    # To count a student as "seen" in a window, require at least this many
    # seconds of tracked visibility with confident recognition.
    min_seen_seconds_in_window: float = 1.5

    # Tracking: IoU threshold for associating detections to an existing track.
    track_iou_threshold: float = 0.30


VALID_MODEL_PACKS = {"buffalo_l", "antelopev2"}


def _available_providers() -> list[str]:
    """Return onnxruntime providers if importable; otherwise CPU-only."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _cuda_available() -> bool:
    """True if CUDAExecutionProvider is available."""
    return "CUDAExecutionProvider" in _available_providers()


def _mps_available() -> bool:
    """True if CoreML/MPS provider is available (macOS acceleration)."""
    providers = _available_providers()
    return ("CoreMLExecutionProvider" in providers) or ("MPSExecutionProvider" in providers)


def resolve_model_config() -> ModelConfig:
    """Resolve model pack + providers.

    What this function does
    -----------------------
    - Prefers GPU acceleration when available.
    - Falls back to CPU automatically.
    - Allows overrides via environment variables.

    Env vars
    --------
    - ``ATTEND_MODEL_PACK``: Force ``buffalo_l`` or ``antelopev2``.
    - ``ATTEND_FORCE_CPU``: If ``1``, disable accelerators.
    - ``ATTEND_DET_SIZE``: Optional ``640x640`` style override.
    """

    forced_pack = os.getenv("ATTEND_MODEL_PACK", "").strip()
    force_cpu = os.getenv("ATTEND_FORCE_CPU", "0") == "1"

    if forced_pack and forced_pack not in VALID_MODEL_PACKS:
        raise ValueError(
            f"ATTEND_MODEL_PACK='{forced_pack}' is not valid. "
            f"Choose one of: {sorted(VALID_MODEL_PACKS)}"
        )

    use_accel = (not force_cpu) and (_cuda_available() or _mps_available())
    pack = forced_pack or ("antelopev2" if use_accel else "buffalo_l")

    if (not force_cpu) and _cuda_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    elif (not force_cpu) and _mps_available():
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        ctx_id = -1
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    det_size = (640, 640)
    det_size_env = os.getenv("ATTEND_DET_SIZE", "").strip().lower()
    if det_size_env:
        try:
            w_s, h_s = det_size_env.split("x", 1)
            det_size = (int(w_s), int(h_s))
        except Exception as exc:
            raise ValueError("ATTEND_DET_SIZE must look like '640x640'") from exc

    return ModelConfig(pack_name=pack, providers=providers, ctx_id=ctx_id, det_size=det_size)


def resolve_runtime_config() -> RuntimeConfig:
    """Resolve non-model runtime configuration from env vars."""

    window_seconds = int(os.getenv("ATTEND_WINDOW_SECONDS", "45"))
    min_windows_present = int(os.getenv("ATTEND_MIN_WINDOWS_PRESENT", "2"))
    detect_every_n_frames = int(os.getenv("ATTEND_DET_EVERY_N", "10"))
    embed_cooldown_s = float(os.getenv("ATTEND_EMBED_COOLDOWN_S", "1.5"))
    match_threshold = float(os.getenv("ATTEND_MATCH_THRESHOLD", "0.5"))
    min_seen_seconds_in_window = float(os.getenv("ATTEND_MIN_SEEN_SECONDS_IN_WINDOW", "1.5"))
    track_iou_threshold = float(os.getenv("ATTEND_TRACK_IOU", "0.3"))
    return RuntimeConfig(
        window_seconds=max(5, window_seconds),
        min_windows_present=max(1, min(3, min_windows_present)),
        detect_every_n_frames=max(1, detect_every_n_frames),
        embed_cooldown_s=max(0.0, embed_cooldown_s),
        match_threshold=match_threshold,
        min_seen_seconds_in_window=max(0.0, min_seen_seconds_in_window),
        track_iou_threshold=min(0.9, max(0.05, track_iou_threshold)),
    )


MODEL = resolve_model_config()
RUNTIME = resolve_runtime_config()
