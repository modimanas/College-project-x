"""
Registration orchestrator.

This is the single entry point the API / kiosk / batch importer calls.
It runs the full pipeline for one student:

    images -> detect -> quality gate -> embed -> consistency check
           -> store -> reload that section's FAISS index

It returns a structured report so the caller (kiosk UI / API) can tell the
operator exactly which photos were accepted, which were dropped, and why.
"""

import uuid
from dataclasses import dataclass, field

import numpy as np

from .config import POLICY
from ._logging import get_logger
from .quality import check_face_crop_quality, pick_primary_face
from .embedder import default_embedder
from .validator import validate_embedding_set, cap_to_max

log = get_logger("service")


@dataclass
class PhotoReport:
    index: int
    accepted: bool
    reason: str
    det_score: float = 0.0
    blur: float = 0.0
    brightness: float = 0.0


@dataclass
class RegistrationReport:
    student_id: str
    success: bool
    reason: str
    accepted_count: int = 0
    photo_reports: list = field(default_factory=list)
    section_index_size: int = 0


class RegistrationService:
    def __init__(self, store, index_manager, embedder=default_embedder):
        self._store = store
        self._index = index_manager
        self._embedder = embedder

    def register(self, *, section_id: str, roll_no: str, name: str,
                 images_bgr: list, defer_index_reload: bool = False) -> RegistrationReport:
        """
        images_bgr: list of np.ndarray (H,W,3) BGR images for ONE student.
        defer_index_reload: if True, skip the FAISS rebuild (used by batch
            import; caller rebuilds the section once at the end).
        """
        student_id = str(uuid.uuid4())
        report = RegistrationReport(student_id=student_id, success=False,
                                    reason="processing")

        if not images_bgr:
            report.reason = "no images provided"
            return report

        good_vectors = []
        good_scores = []
        good_landmarks = []
        good_masks = []

        for i, img in enumerate(images_bgr):
            pr = PhotoReport(index=i, accepted=False, reason="")

            faces = self._embedder.detect(img)
            face, why = pick_primary_face(faces)
            if face is None:
                pr.reason = why
                report.photo_reports.append(pr)
                continue

            q = check_face_crop_quality(img, face)
            pr.det_score = q.det_score
            pr.blur = q.blur_variance
            pr.brightness = q.brightness
            if not q.passed:
                pr.reason = q.reason
                report.photo_reports.append(pr)
                continue

            emb = self._embedder.embed_face(face, image_bgr=img)
            good_vectors.append(emb.vector)
            good_scores.append(emb.det_score)
            good_landmarks.append(emb.kps)
            good_masks.append(emb.aligned_crop)
            pr.accepted = True
            pr.reason = "ok"
            report.photo_reports.append(pr)

        # Need a minimum number of clean photos before consistency check.
        if len(good_vectors) < POLICY.min_valid_embeddings:
            report.reason = (
                f"only {len(good_vectors)} usable photo(s); "
                f"need {POLICY.min_valid_embeddings}"
            )
            return report

        # Consistency: drop any embedding that doesn't look like the same person.
        consistency = validate_embedding_set(good_vectors)
        if consistency.rejected_indices:
            log.warning(
                "consistency outliers | roll=%s section=%s dropped=%s sims=%s",
                roll_no, section_id,
                consistency.rejected_indices,
                [f"{consistency.similarities[i]:.3f}" for i in consistency.rejected_indices],
            )
        if not consistency.enough:
            report.reason = consistency.reason
            return report

        kept_vectors = [good_vectors[i] for i in consistency.accepted_indices]
        kept_scores = [good_scores[i] for i in consistency.accepted_indices]
        kept_landmarks = [good_landmarks[i] for i in consistency.accepted_indices]
        kept_masks = [good_masks[i] for i in consistency.accepted_indices]
        kept_sims = [consistency.similarities[i] for i in consistency.accepted_indices]

        # Cap to max_embeddings, keeping the most representative.
        keep_idx = cap_to_max(kept_vectors, kept_sims)
        final_vectors = [kept_vectors[i] for i in keep_idx]
        final_scores = [kept_scores[i] for i in keep_idx]
        final_landmarks = [kept_landmarks[i] for i in keep_idx]
        final_masks = [kept_masks[i] for i in keep_idx]

        # Persist + hot-reload the section index.
        from .store import StoredStudent
        stored = StoredStudent(
            student_id=student_id,
            section_id=section_id,
            roll_no=roll_no,
            name=name,
            model_pack=self._embedder.pack_name,
            embeddings=final_vectors,
            det_scores=final_scores,
            landmarks=final_landmarks,
            masks=final_masks,
        )
        saved_id = self._store.save_student(stored)
        report.student_id = saved_id

        if defer_index_reload:
            index_size = self._index.section_size(section_id)
        else:
            index_size = self._index.reload_section(section_id)

        report.success = True
        report.accepted_count = len(final_vectors)
        report.section_index_size = index_size
        report.reason = consistency.reason  # 'ok' or 'dropped N outliers'
        return report

    def register_batch(self, students: list) -> list:
        """
        Bulk-register many students, rebuilding each affected section's FAISS
        index ONCE at the end instead of per student.

        students: list of dicts with keys section_id, roll_no, name, images_bgr.
        Returns the list of RegistrationReport in the same order.
        """
        reports = []
        touched_sections = set()
        for s in students:
            r = self.register(
                section_id=s["section_id"], roll_no=s["roll_no"],
                name=s["name"], images_bgr=s["images_bgr"],
                defer_index_reload=True,
            )
            reports.append(r)
            if r.success:
                touched_sections.add(s["section_id"])

        for sid in touched_sections:
            size = self._index.reload_section(sid)
            log.info("batch index rebuild | section=%s size=%d", sid, size)
        return reports
