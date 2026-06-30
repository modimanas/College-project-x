"""Registration module for the real-time attendance system."""

from .config import MODEL, QUALITY, POLICY, resolve_model_config
from .embedder import Embedder, default_embedder, FaceEmbedding
from .quality import check_face_crop_quality, pick_primary_face, QualityResult
from .validator import validate_embedding_set, ConsistencyResult
from .store import get_store, MemoryStore, PostgresStore, StoredStudent, decode_mask
from .index_manager import IndexManager, SectionIndex
from .service import RegistrationService, RegistrationReport

__all__ = [
    "MODEL", "QUALITY", "POLICY", "resolve_model_config",
    "Embedder", "default_embedder", "FaceEmbedding",
    "check_face_crop_quality", "pick_primary_face", "QualityResult",
    "validate_embedding_set", "ConsistencyResult",
    "get_store", "MemoryStore", "PostgresStore", "StoredStudent", "decode_mask",
    "IndexManager", "SectionIndex",
    "RegistrationService", "RegistrationReport",
]
