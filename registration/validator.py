"""
Set-level validation across a student's captured embeddings.

A single bad photo (wrong person uploaded by mistake, or a background
face that slipped through) produces an embedding that is dissimilar to
the rest. We detect those outliers BEFORE storing, so a student's record
is never silently corrupted.

Method: compute the element-wise median embedding of the set (robust to
outliers, unlike the mean), then check each embedding's cosine similarity
to that median. Anything below POLICY.intra_consistency_min is flagged.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import POLICY


@dataclass
class ConsistencyResult:
    accepted_indices: list = field(default_factory=list)
    rejected_indices: list = field(default_factory=list)
    similarities: list = field(default_factory=list)  # per-input sim to median
    enough: bool = False                               # >= min_valid after pruning
    reason: str = "ok"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def validate_embedding_set(embeddings: list) -> ConsistencyResult:
    """
    embeddings: list of (512,) L2-normalized float32 vectors for ONE student.
    Returns which to keep / drop and whether enough remain to register.
    """
    result = ConsistencyResult()

    if len(embeddings) == 0:
        result.reason = "no embeddings provided"
        return result

    mat = np.stack(embeddings).astype(np.float32)  # (N, 512)

    # Robust center: element-wise median, then renormalize to unit length.
    median_vec = _normalize(np.median(mat, axis=0))

    # Cosine sim of each embedding to the median (both unit vectors -> dot).
    sims = mat @ median_vec  # (N,)
    result.similarities = [float(s) for s in sims]

    for i, s in enumerate(sims):
        if s >= POLICY.intra_consistency_min:
            result.accepted_indices.append(i)
        else:
            result.rejected_indices.append(i)

    result.enough = len(result.accepted_indices) >= POLICY.min_valid_embeddings

    if not result.enough:
        result.reason = (
            f"only {len(result.accepted_indices)} consistent embeddings "
            f"(need {POLICY.min_valid_embeddings}); "
            f"{len(result.rejected_indices)} outliers dropped"
        )
    elif result.rejected_indices:
        result.reason = (
            f"accepted {len(result.accepted_indices)}, "
            f"dropped {len(result.rejected_indices)} outlier(s)"
        )

    return result


def cap_to_max(embeddings: list, sims_to_median: list) -> list:
    """
    If we captured more than max_embeddings, keep the highest-quality ones
    (closest to the median = most representative). Returns kept indices.
    """
    if len(embeddings) <= POLICY.max_embeddings:
        return list(range(len(embeddings)))
    order = np.argsort(sims_to_median)[::-1]  # highest sim first
    return sorted(order[: POLICY.max_embeddings].tolist())
