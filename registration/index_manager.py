"""
Per-section FAISS index manager (the cluster matching core).

Design: ONE flat inner-product index per section. Because embeddings are
L2-normalized, inner product == cosine similarity. Each section holds only
its own students' embeddings (~500 vectors for 100 students x 5), so a flat
scan is microseconds — no ANN approximation needed at this scale.

A student has multiple embeddings; all of them go into the index, each
mapped back to the student_id. At match time we take the single highest
similarity across all of a student's vectors (handled by the caller: the
top-1 FAISS hit already corresponds to the best-matching angle).

Hot-reload: when a new student registers, we rebuild only that section's
index from the store. Rebuilding 500 vectors is instant, so we don't bother
with incremental adds (keeps the code correct and simple).
"""

import threading

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class _NumpyIndex:
    """Fallback if FAISS isn't installed — identical interface, slower."""

    def __init__(self, dim: int):
        self.dim = dim
        self.mat = np.zeros((0, dim), dtype=np.float32)

    def add(self, vecs: np.ndarray):
        self.mat = vecs.astype(np.float32)

    def search(self, query: np.ndarray, k: int):
        if self.mat.shape[0] == 0:
            n = query.shape[0]
            return (np.full((n, k), -1.0, np.float32),
                    np.full((n, k), -1, np.int64))
        sims = query @ self.mat.T               # (Nq, M) cosine sims
        idx = np.argsort(sims, axis=1)[:, ::-1][:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype(np.float32), idx.astype(np.int64)


class SectionIndex:
    def __init__(self, section_id: str, dim: int = 512):
        self.section_id = section_id
        self.dim = dim
        self.student_ids: list = []   # row index -> student_id
        self._index = self._new_index(dim)

    @staticmethod
    def _new_index(dim: int):
        if _HAS_FAISS:
            return faiss.IndexFlatIP(dim)
        return _NumpyIndex(dim)

    def rebuild(self, vectors: np.ndarray, student_ids: list):
        self._index = self._new_index(self.dim)
        self.student_ids = list(student_ids)
        if vectors.shape[0] > 0:
            v = np.ascontiguousarray(vectors, dtype=np.float32)
            self._index.add(v)

    def search(self, queries: np.ndarray, k: int = 1):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        return self._index.search(q, k)

    @property
    def size(self) -> int:
        return len(self.student_ids)


class IndexManager:
    """
    Thread-safe holder of all section indexes. Workers query it; the
    registration flow calls reload_section() after each new student.
    """

    def __init__(self, store, dim: int = 512):
        self._store = store
        self._dim = dim
        self._sections = {}                 # section_id -> SectionIndex
        self._lock = threading.RLock()

    def reload_section(self, section_id: str) -> int:
        vectors, ids = self._store.get_section_embeddings(section_id)
        with self._lock:
            idx = self._sections.get(section_id)
            if idx is None:
                idx = SectionIndex(section_id, self._dim)
                self._sections[section_id] = idx
            idx.rebuild(vectors, ids)
            return idx.size

    def reload_all(self, section_ids: list):
        for sid in section_ids:
            self.reload_section(sid)

    def section_size(self, section_id: str) -> int:
        with self._lock:
            idx = self._sections.get(section_id)
            return idx.size if idx else 0

    def match(self, section_id: str, query_vectors: np.ndarray,
              threshold: float = 0.5):
        """
        query_vectors: (Nq, 512) L2-normalized.
        Returns list of (student_id | None, score) per query, top-1, gated
        by threshold. Because each student has several vectors in the index,
        the top-1 hit is already that student's best-matching angle.
        """
        with self._lock:
            idx = self._sections.get(section_id)
            if idx is None or idx.size == 0:
                return [(None, 0.0)] * query_vectors.shape[0]
            scores, rows = idx.search(query_vectors, k=1)
            out = []
            for i in range(query_vectors.shape[0]):
                score = float(scores[i][0])
                row = int(rows[i][0])
                if row < 0 or score < threshold:
                    out.append((None, score))
                else:
                    out.append((idx.student_ids[row], score))
            return out
