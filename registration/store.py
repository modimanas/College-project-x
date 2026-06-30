"""
Storage layer.

Primary backend: PostgreSQL + pgvector (production).
Fallback backend: in-memory dict (so you can run/test the full pipeline
on your laptop with zero infra). Selected by presence of ATTEND_DB_DSN.

We store EVERY embedding as its own row (never an average) — matching is
done by max-similarity across a student's rows downstream.
"""

import os
import json
from dataclasses import dataclass, field

import numpy as np

# Fixed section UUID used by MemoryStore (no DB, no foreign-key enforcement).
DEMO_SECTION_ID = "11111111-1111-1111-1111-111111111111"

# Branch list returned by MemoryStore so the registration UI works without a DB.
_DEMO_BRANCHES = [
    {"id": "00000001-0000-0000-0000-000000000001", "code": "CSIT",     "name": "Computer Science & Information Technology"},
    {"id": "00000001-0000-0000-0000-000000000002", "code": "CSE",      "name": "Computer Science & Engineering"},
    {"id": "00000001-0000-0000-0000-000000000003", "code": "CS",       "name": "Computer Science"},
    {"id": "00000001-0000-0000-0000-000000000004", "code": "CSE-AIML", "name": "CSE (AI & Machine Learning)"},
    {"id": "00000001-0000-0000-0000-000000000005", "code": "CSE-AI",   "name": "CSE (Artificial Intelligence)"},
    {"id": "00000001-0000-0000-0000-000000000006", "code": "IT",       "name": "Information Technology"},
    {"id": "00000001-0000-0000-0000-000000000007", "code": "CSE-DS",   "name": "CSE (Data Science)"},
    {"id": "00000001-0000-0000-0000-000000000008", "code": "CSE-CS",   "name": "CSE (Cyber Security)"},
    {"id": "00000001-0000-0000-0000-000000000009", "code": "ECE",      "name": "Electronics & Communication Engineering"},
    {"id": "00000001-0000-0000-0000-000000000010", "code": "EEE",      "name": "Electrical & Electronics Engineering"},
    {"id": "00000001-0000-0000-0000-000000000011", "code": "ELCE",     "name": "Electronics & Computer Engineering"},
    {"id": "00000001-0000-0000-0000-000000000012", "code": "ECE-VLSI", "name": "ECE (VLSI Design)"},
    {"id": "00000001-0000-0000-0000-000000000013", "code": "ME",       "name": "Mechanical Engineering"},
    {"id": "00000001-0000-0000-0000-000000000014", "code": "AM-AI",    "name": "Applied Mechanics & AI"},
    {"id": "00000001-0000-0000-0000-000000000015", "code": "ASH",      "name": "Applied Science & Humanities"},
    {"id": "00000001-0000-0000-0000-000000000016", "code": "BPHARM",   "name": "Bachelor of Pharmacy"},
]

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _encode_mask(mask: np.ndarray) -> bytes:
    """Encode a 112x112 aligned crop as JPEG bytes for compact storage."""
    if not _HAS_CV2:
        return np.ascontiguousarray(mask).tobytes()
    ok, buf = cv2.imencode(".jpg", mask, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else b""


def decode_mask(blob: bytes) -> np.ndarray:
    """Decode stored JPEG bytes back to a 112x112x3 BGR array (for re-embed)."""
    arr = np.frombuffer(blob, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@dataclass
class StoredStudent:
    student_id: str
    section_id: str
    roll_no: str
    name: str
    model_pack: str
    embeddings: list = field(default_factory=list)   # list[np.ndarray] (512,)
    det_scores: list = field(default_factory=list)
    landmarks: list = field(default_factory=list)    # list[np.ndarray] (5,2)
    masks: list = field(default_factory=list)         # list[np.ndarray] (112,112,3) aligned crops


class BaseStore:
    def save_student(self, s: StoredStudent) -> str: ...
    def get_section_embeddings(self, section_id: str): ...
    def list_branches(self) -> list: ...
    def lookup_section(self, branch_code: str, year: int, section_name: str) -> "str | None": ...
    def close(self): ...


# ── In-memory backend (dev / no DB) ──────────────────────────────────────

class MemoryStore(BaseStore):
    def __init__(self):
        self._students = {}  # student_id -> StoredStudent

    def save_student(self, s: StoredStudent) -> str:
        self._students[s.student_id] = s
        return s.student_id

    def get_section_embeddings(self, section_id: str):
        """Returns (vectors: np.ndarray (M,512), student_ids: list[str])."""
        vecs, ids = [], []
        for s in self._students.values():
            if s.section_id != section_id:
                continue
            for v in s.embeddings:
                vecs.append(v)
                ids.append(s.student_id)
        if not vecs:
            return np.zeros((0, 512), dtype=np.float32), []
        return np.stack(vecs).astype(np.float32), ids

    def list_branches(self):
        return _DEMO_BRANCHES

    def lookup_section(self, branch_code, year, section_name):
        # In-memory mode: no real hierarchy — every lookup resolves to the demo UUID.
        return DEMO_SECTION_ID

    def close(self):
        pass


# ── PostgreSQL + pgvector backend (production) ───────────────────────────

class PostgresStore(BaseStore):
    def __init__(self, dsn: str):
        import psycopg
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self._conn = psycopg.connect(dsn, autocommit=False)
        register_vector(self._conn)

    def save_student(self, s: StoredStudent) -> str:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO students (id, section_id, roll_no, name,
                                          model_pack, registered)
                    VALUES (%s, %s, %s, %s, %s, TRUE)
                    ON CONFLICT (section_id, roll_no) DO UPDATE
                      SET name = EXCLUDED.name,
                          model_pack = EXCLUDED.model_pack,
                          registered = TRUE
                    RETURNING id
                    """,
                    (s.student_id, s.section_id, s.roll_no, s.name, s.model_pack),
                )
                student_id = str(cur.fetchone()[0])

                # Replace any prior embeddings for a clean re-registration.
                cur.execute(
                    "DELETE FROM student_embeddings WHERE student_id = %s",
                    (student_id,),
                )
                for idx, (vec, score) in enumerate(
                    zip(s.embeddings, s.det_scores or [None] * len(s.embeddings))
                ):
                    lm = s.landmarks[idx] if idx < len(s.landmarks) else None
                    mask = s.masks[idx] if idx < len(s.masks) else None

                    landmarks_json = (
                        json.dumps(np.asarray(lm, dtype=float).tolist())
                        if lm is not None else None
                    )
                    mask_bytes = _encode_mask(mask) if mask is not None else None

                    cur.execute(
                        """
                        INSERT INTO student_embeddings
                            (student_id, section_id, embedding, det_score,
                             landmarks, aligned_mask, source)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s, 'registration')
                        """,
                        (student_id, s.section_id,
                         np.asarray(vec, dtype=np.float32), score,
                         landmarks_json, mask_bytes),
                    )
            self._conn.commit()
            return student_id
        except Exception:
            self._conn.rollback()
            raise

    def get_section_embeddings(self, section_id: str):
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_id, embedding
                FROM student_embeddings
                WHERE section_id = %s
                """,
                (section_id,),
            )
            rows = cur.fetchall()
        if not rows:
            return np.zeros((0, 512), dtype=np.float32), []
        ids = [str(r[0]) for r in rows]
        vecs = np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])
        return vecs, ids

    def list_branches(self):
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, code, name FROM branches ORDER BY code")
            return [{"id": str(r[0]), "code": r[1], "name": r[2]} for r in cur.fetchall()]

    def lookup_section(self, branch_code: str, year: int, section_name: str):
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id FROM sections s
                JOIN branches b ON b.id = s.branch_id
                WHERE b.code = %s AND s.year = %s AND s.name = %s
                """,
                (branch_code, year, section_name),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None

    def close(self):
        self._conn.close()


# ── Factory ───────────────────────────────────────────────────────────────

def get_store() -> BaseStore:
    dsn = os.getenv("ATTEND_DB_DSN")
    if dsn:
        return PostgresStore(dsn)
    return MemoryStore()
