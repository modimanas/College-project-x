"""Section-scoped embedding matcher.

What this module does
---------------------
- Wraps the existing registration index system for detection-time matching.
- Provides a small, stable API: ``load_section`` + ``match_one``.

Why it exists
-------------
As an ML engineer you want your detection worker to focus on:
  RTSP -> detect -> track -> embed -> match -> decide
and not worry about DB schemas. The matcher hides those details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from registration.index_manager import IndexManager
from registration.store import get_store

from ._logging import get_logger

log = get_logger("matcher")


@dataclass(frozen=True)
class Match:
    """One match result."""

    student_id: str | None
    score: float


class SectionMatcher:
    """Loads a section index and performs fast cosine-similarity search."""

    def __init__(self):
        self._store = get_store()
        self._index = IndexManager(self._store)
        self._loaded_section_id: str | None = None

    @property
    def loaded_section_id(self) -> str | None:
        return self._loaded_section_id

    def close(self) -> None:
        """Close the underlying store connection."""

        self._store.close()

    def load_section(self, section_id: str) -> int:
        """Load/reload a section index into memory.

        Returns:
            Number of vectors in the loaded section index.
        """

        size = self._index.reload_section(section_id)
        self._loaded_section_id = section_id
        log.info("section loaded | section=%s vectors=%d", section_id, size)
        return size

    def match_one(self, section_id: str, embedding_512: np.ndarray, *, threshold: float) -> Match:
        """Match one embedding against a given section.

        Args:
            section_id: Section to search within.
            embedding_512: L2-normalized (512,) float32 embedding.
            threshold: Minimum cosine similarity to accept.
        """

        if embedding_512.ndim != 1:
            raise ValueError("embedding_512 must be a 1D vector")
        if embedding_512.shape[0] != 512:
            raise ValueError("embedding_512 must have length 512")

        sid, score = self._index.match(section_id, np.expand_dims(embedding_512, axis=0), threshold=threshold)[0]
        return Match(student_id=sid, score=float(score))

