"""
Central configuration for the registration module.

Key design decision: model pack auto-selection.
- buffalo_l  (ResNet50 backbone) -> fast, used for CPU dev/testing
- antelopev2 (ResNet100 backbone) -> max accuracy, used for GPU production

The selection is automatic based on whether a CUDA provider is available,
but can be force-overridden with the ATTEND_MODEL_PACK env var.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QualityThresholds:
    """Gates a registration photo must pass to be accepted."""
    min_face_pixels: int = 80          # face bbox shorter side must exceed this
    min_blur_variance: float = 100.0   # Laplacian variance; below = too blurry
    min_brightness: float = 50.0       # mean luma; below = too dark
    max_brightness: float = 205.0      # mean luma; above = blown out / too bright
    min_det_score: float = 0.80        # SCRFD detection confidence
    max_tilt_degrees: float = 30.0     # head roll (eye line angle) limit


@dataclass(frozen=True)
class RegistrationPolicy:
    """How many photos / embeddings we require per student."""
    min_valid_embeddings: int = 5      # below this -> registration rejected
    target_embeddings: int = 8         # sweet spot we aim to capture
    max_embeddings: int = 10           # hard cap stored per student
    # consistency: every embedding should be at least this similar to the
    # median of the set, else it's likely a different person's photo.
    intra_consistency_min: float = 0.40


@dataclass(frozen=True)
class ModelConfig:
    pack_name: str                     # "buffalo_l" or "antelopev2"
    providers: list = field(default_factory=list)
    ctx_id: int = 0                    # 0 = first GPU, -1 = CPU
    det_size: tuple = (640, 640)       # SCRFD input resolution


VALID_MODEL_PACKS = {"buffalo_l", "antelopev2"}


def _cuda_available() -> bool:
    """Detect a usable CUDA execution provider via onnxruntime."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        # onnxruntime-gpu exposes CUDAExecutionProvider; CPU-only builds won't.
        return "CUDAExecutionProvider" in available
    except Exception as exc:
        import logging
        logging.getLogger("attendance.config").debug(
            "CUDA probe failed (treating as unavailable): %s", exc)
        return False


def _mps_available() -> bool:
    """Detect Apple's Metal (MPS/CoreML) provider when present."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        return ("CoreMLExecutionProvider" in available) or ("MPSExecutionProvider" in available)
    except Exception:
        return False


def resolve_model_config() -> ModelConfig:
    """
    Decide which model pack + providers to use.

    Override with env vars:
      ATTEND_MODEL_PACK = buffalo_l | antelopev2
      ATTEND_FORCE_CPU  = 1   (force CPU even if GPU present)
    """
    force_cpu = os.getenv("ATTEND_FORCE_CPU", "0") == "1"
    # Prefer CUDA when present; otherwise fall back to other accelerators.
    use_gpu = (not force_cpu) and (_cuda_available() or _mps_available())

    forced_pack = os.getenv("ATTEND_MODEL_PACK", "").strip()
    if forced_pack:
        if forced_pack not in VALID_MODEL_PACKS:
            raise ValueError(
                f"ATTEND_MODEL_PACK='{forced_pack}' is not valid. "
                f"Choose one of: {sorted(VALID_MODEL_PACKS)}"
            )
        pack = forced_pack
    else:
        pack = "antelopev2" if use_gpu else "buffalo_l"

    if use_gpu and _cuda_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    elif use_gpu and _mps_available():
        # On macOS, CoreML/MPS can accelerate inference; keep CPU as fallback.
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        ctx_id = -1
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    return ModelConfig(pack_name=pack, providers=providers, ctx_id=ctx_id)


# Module-level singletons (import these elsewhere)
QUALITY = QualityThresholds()
POLICY = RegistrationPolicy()
MODEL = resolve_model_config()


if __name__ == "__main__":
    print("Resolved model config:")
    print(f"  pack      : {MODEL.pack_name}")
    print(f"  providers : {MODEL.providers}")
    print(f"  ctx_id    : {MODEL.ctx_id}")
