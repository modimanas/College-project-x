"""
The embedder — the heart of the system.

Wraps insightface's FaceAnalysis app, which internally does:
  1. SCRFD face detection (+ 5 landmarks)
  2. 5-point similarity-transform alignment (norm_crop to 112x112)
  3. ArcFace/antelopev2 recognition -> 512-d embedding

We never align manually; app.get() does it. We DO L2-normalize every
embedding ourselves so cosine similarity == inner product downstream
(this is what FAISS IndexFlatIP expects).

Model pack is chosen by config.resolve_model_config():
  GPU -> antelopev2 (R100, max accuracy)
  CPU -> buffalo_l  (R50, fast, for dev)
"""

import threading
from dataclasses import dataclass

import numpy as np

from .config import MODEL


@dataclass
class FaceEmbedding:
    vector: np.ndarray        # (512,) float32, L2-normalized
    det_score: float
    bbox: np.ndarray          # (4,) int
    kps: np.ndarray           # (5, 2) landmarks — the 5-point mask points
    aligned_crop: np.ndarray  # (112,112,3) BGR — the aligned face "mask"


class Embedder:
    """
    Lazy-loaded singleton-ish wrapper. Construct once at startup and reuse;
    loading the ONNX models takes a few seconds and ~300MB RAM.
    """

    def __init__(self, model_config=MODEL):
        self._cfg = model_config
        self._app = None          # loaded on first use
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        # Fast path — no lock needed once loaded.
        if self._app is not None:
            return
        # Slow path — serialize concurrent first-time loaders (double-checked).
        with self._load_lock:
            if self._app is not None:
                return
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name=self._cfg.pack_name,
                providers=self._cfg.providers,
            )
            app.prepare(ctx_id=self._cfg.ctx_id, det_size=self._cfg.det_size)
            self._app = app

    @property
    def pack_name(self) -> str:
        return self._cfg.pack_name

    @property
    def providers(self) -> list:
        return list(self._cfg.providers)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)

    def detect(self, image_bgr: np.ndarray) -> list:
        """
        Return raw insightface Face objects (with .bbox, .kps, .det_score,
        .normed_embedding). Detection + alignment + recognition all run here.
        """
        self._ensure_loaded()
        return self._app.get(image_bgr)

    def align_mask(self, image_bgr: np.ndarray, face) -> np.ndarray:
        """
        Produce the 5-point aligned face "mask" — the 112x112 crop the
        recognition model actually consumes. Uses insightface's norm_crop,
        which applies a similarity transform so the 5 landmarks land on the
        ArcFace template positions. This is the SAME alignment used internally
        for recognition; we surface it so it can be stored/re-embedded later.
        """
        from insightface.utils import face_align
        return face_align.norm_crop(image_bgr, landmark=face.kps, image_size=112)

    def embed_face(self, face, image_bgr: np.ndarray = None) -> FaceEmbedding:
        """
        Convert an insightface Face object into our normalized FaceEmbedding,
        including the 5-point landmarks and the aligned 112x112 mask crop.

        image_bgr is required to compute the aligned mask (norm_crop needs the
        source pixels). If omitted, aligned_crop is an empty array.
        """
        raw = getattr(face, "embedding", None)
        if raw is None:
            raise ValueError("face has no embedding; was recognition enabled?")
        vec = self._l2_normalize(np.asarray(raw, dtype=np.float32))

        kps = np.asarray(face.kps, dtype=np.float32)
        if image_bgr is not None:
            aligned = self.align_mask(image_bgr, face)
        else:
            aligned = np.zeros((112, 112, 3), dtype=np.uint8)

        return FaceEmbedding(
            vector=vec,
            det_score=float(getattr(face, "det_score", 0.0)),
            bbox=face.bbox.astype(int),
            kps=kps,
            aligned_crop=aligned,
        )

    def re_embed_aligned(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Re-generate an embedding directly from a STORED aligned mask crop —
        no detection/alignment needed. This is the model-migration path:
        swap the recognition model, run every stored crop through it.
        Returns an L2-normalized 512-d vector.
        """
        self._ensure_loaded()
        rec = self._app.models.get("recognition")
        if rec is None:
            raise RuntimeError("recognition model not loaded")
        feat = rec.get_feat(aligned_crop).flatten()
        return self._l2_normalize(np.asarray(feat, dtype=np.float32))

    def embed_image_single(self, image_bgr: np.ndarray):
        """
        Convenience for registration: detect, return all faces so the caller
        can run quality gates and pick the primary one.
        """
        return self.detect(image_bgr)


# A process-wide default instance other modules can import.
default_embedder = Embedder()
