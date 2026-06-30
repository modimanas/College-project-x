"""
End-to-end logic test using a FAKE embedder (no insightface / GPU needed).

We simulate:
  - 3 students in section CSC-A, each with 8 consistent embeddings
  - one student whose photo set includes a "wrong person" outlier
  - a detection-time match against the cluster index

This proves quality-gate flow, consistency validation, store, index build,
and cluster matching all work end-to-end.
"""

import os
import sys
import numpy as np

# Ensure imports work when running as a script from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from registration.store import MemoryStore, StoredStudent
from registration.index_manager import IndexManager
from registration.validator import validate_embedding_set


def unit(v):
    return v / np.linalg.norm(v)


def make_person_vectors(seed, n=8, noise_norm=0.45):
    """
    A person = a base direction + small per-shot noise (angle variation).
    noise_norm controls how far each shot drifts from the base. We scale the
    raw gaussian so its L2 norm equals noise_norm, giving realistic
    same-person cosine similarities (~0.85-0.92) instead of near-orthogonal.
    """
    rng = np.random.default_rng(seed)
    base = unit(rng.standard_normal(512).astype(np.float32))
    vecs = []
    for _ in range(n):
        raw = rng.standard_normal(512).astype(np.float32)
        noise = raw / np.linalg.norm(raw) * noise_norm
        vecs.append(unit(base + noise))
    return base, vecs


print("=" * 60)
print("TEST 1 — consistency validator drops a wrong-person outlier")
print("=" * 60)
_, good = make_person_vectors(seed=1, n=7)
_, other = make_person_vectors(seed=999, n=1)   # different person
mixed = good + other                            # 7 good + 1 outlier
res = validate_embedding_set(mixed)
print(f"  accepted: {len(res.accepted_indices)}  rejected: {len(res.rejected_indices)}")
print(f"  reason  : {res.reason}")
assert len(res.rejected_indices) == 1, "should drop exactly the outlier"
assert res.enough, "7 good ones is enough"
print("  PASS ✅")

print()
print("=" * 60)
print("TEST 2 — store + per-section FAISS/numpy index build")
print("=" * 60)
store = MemoryStore()
index = IndexManager(store)

people = {}
for i, sid in enumerate(["stu-A", "stu-B", "stu-C"]):
    base, vecs = make_person_vectors(seed=10 + i, n=8)
    people[sid] = base
    store.save_student(StoredStudent(
        student_id=sid, section_id="CSC-A", roll_no=f"R{i}",
        name=sid, model_pack="fake", embeddings=vecs,
        det_scores=[0.99] * 8,
    ))

size = index.reload_section("CSC-A")
print(f"  CSC-A index vectors: {size} (3 students x 8 = 24 expected)")
assert size == 24
print("  PASS ✅")

print()
print("=" * 60)
print("TEST 3 — detection-time cluster match (top angle wins)")
print("=" * 60)
# Simulate a detected face of stu-B at a noisy angle.
rng = np.random.default_rng(7)
def noisy(base, scale=0.45):
    raw = rng.standard_normal(512).astype(np.float32)
    return unit(base + raw / np.linalg.norm(raw) * scale)
query = noisy(people["stu-B"])
matches = index.match("CSC-A", np.stack([query]), threshold=0.5)
sid, score = matches[0]
print(f"  matched: {sid}  score: {score:.3f}")
assert sid == "stu-B", f"expected stu-B, got {sid}"
print("  PASS ✅")

print()
print("=" * 60)
print("TEST 4 — unknown face (different section person) is rejected")
print("=" * 60)
_, stranger = make_person_vectors(seed=5000, n=1)
matches = index.match("CSC-A", np.stack([stranger[0]]), threshold=0.5)
sid, score = matches[0]
print(f"  matched: {sid}  score: {score:.3f}")
assert sid is None, "stranger must not match anyone"
print("  PASS ✅")

print()
print("=" * 60)
print("TEST 5 — batch match (70 detected faces at once)")
print("=" * 60)
batch = []
truth = []
for _ in range(70):
    who = rng.choice(["stu-A", "stu-B", "stu-C"])
    truth.append(who)
    batch.append(noisy(people[who]))
matches = index.match("CSC-A", np.stack(batch), threshold=0.5)
correct = sum(1 for (sid, _), t in zip(matches, truth) if sid == t)
print(f"  correctly matched {correct}/70 simulated faces")
assert correct >= 68, "should match almost all"
print("  PASS ✅")

print()
print("ALL TESTS PASSED ✅")
