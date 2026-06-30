# Registration Module

Turns a student's photos into clean, validated, section-clustered face
embeddings stored for real-time attendance matching.

## Pipeline

```
images ─► detect (SCRFD) ─► quality gate ─► align+embed (ArcFace/antelopev2)
       ─► consistency check ─► store (pgvector) ─► reload section FAISS index
```

## Key design decisions

- **Store all 5–8 embeddings per student, never an average.** Matching takes
  the max similarity across a student's vectors, so the angle that matches the
  board-mounted camera (the top-angle shot) wins. Averaging would dilute that.
- **Cluster matching.** Each section has its own FAISS index (~500 vectors).
  A camera only ever searches its own section's students, never all 7200.
- **Strict at registration, lenient at detection.** Bad photos are rejected
  here so stored embeddings stay clean; bad angles at detection time are
  handled by multi-frame accumulation (in the detection module, not here).
- **Consistency check** catches a wrong-person photo before it corrupts a
  student record (outlier vs the set's median embedding).
- **Auto model selection.** GPU → `antelopev2` (R100, max accuracy);
  CPU → `buffalo_l` (R50, fast) for laptop development.

## Run it

### Dev (laptop, no GPU, no DB)
```bash
pip install -r registration/requirements.txt
export ATTEND_FORCE_CPU=1          # use buffalo_l on CPU
# no ATTEND_DB_DSN -> in-memory store
uvicorn registration.api:app --reload
```

### Production (GPU box + Postgres)
```bash
export ATTEND_DB_DSN="postgresql://user:pass@host:5432/attendance"
# GPU auto-detected -> antelopev2 selected automatically
psql "$ATTEND_DB_DSN" -f registration/schema.sql
uvicorn registration.api:app --host 0.0.0.0 --port 8000
```

### Kiosk auto-capture (needs a webcam + display)
```python
from registration.auto_capture import run_kiosk_capture
from registration.service import RegistrationService
from registration.store import get_store
from registration.index_manager import IndexManager

store = get_store(); index = IndexManager(store)
svc = RegistrationService(store, index)

cap = run_kiosk_capture(camera_index=0, target_count=8)
report = svc.register(section_id="<uuid>", roll_no="R101",
                      name="Asha", images_bgr=cap.images_bgr)
print(report.success, report.reason, report.accepted_count)
```

## Env vars

| var | meaning |
|---|---|
| `ATTEND_DB_DSN` | Postgres DSN. If unset → in-memory store (dev). |
| `ATTEND_FORCE_CPU` | `1` forces CPU + buffalo_l even if a GPU exists. |
| `ATTEND_MODEL_PACK` | force `buffalo_l` or `antelopev2`. |

## Files

| file | role |
|---|---|
| `config.py` | thresholds, policy, model auto-selection |
| `quality.py` | per-photo gates (blur, light, size, tilt, single-face) |
| `embedder.py` | insightface wrapper (detect + align + embed) |
| `validator.py` | set-level consistency (drop wrong-person outliers) |
| `store.py` | Postgres+pgvector store, in-memory fallback |
| `index_manager.py` | per-section FAISS index + hot reload |
| `service.py` | orchestrator: the `register()` entry point |
| `auto_capture.py` | webcam kiosk that auto-snaps good angles |
| `api.py` | FastAPI endpoints |
| `schema.sql` | hierarchy + embedding tables |

> Note: the pretrained insightface packs (`buffalo_l`, `antelopev2`) are
> released for non-commercial/research use. Confirm the license fits your
> deployment before going live.

## Phase 1 — selfie self-registration (current)

Students register from their own phone via a web page.

```bash
# 1. start the database
docker compose up -d db

# 2. point the app at it + run
export ATTEND_DB_DSN="postgresql://attend:attend_dev_pw@localhost:5432/attendance"
export ATTEND_FORCE_CPU=1          # remove on the GPU box -> antelopev2
pip install -r registration/requirements.txt
uvicorn registration.api:app --host 0.0.0.0 --port 8000

# 3. on the phone, open  http://<server-ip>:8000/
#    (camera access needs HTTPS in production — put it behind a TLS proxy,
#     or use localhost which browsers treat as secure)
```

### The angle caveat (important)
Selfie cameras shoot from below/eye-level; the classroom camera shoots from
above the board. To bridge this gap:
1. one capture step asks the student to **raise the phone high and look up**,
   mimicking the overhead angle;
2. the real fix is **auto-enrollment** (a later phase): the classroom camera
   adds high-confidence top-down embeddings during the first weeks, and
   accuracy climbs as the profile learns the true angle.

Treat selfie registration as the **bootstrap**, not the final state.

## Verified
- `schema.sql` runs clean on PostgreSQL 16 + pgvector.
- `PostgresStore` write → pgvector → FAISS rebuild → match: round-trip passes.
- `test_logic.py`: 5/5 logic tests pass (consistency, index, match, batch).

## Phase 1 UI — live face-mesh scanner (current)

The registration page (`web/register.html`) is a biometric-style scanner:

- **MediaPipe FaceLandmarker** runs in the browser → draws the live 478-point
  mesh overlay and estimates head pose every frame.
- Pose drives **auto-capture**: a ring fills while the target angle is held
  stable, then snaps. No buttons (a manual fallback exists if needed).
- 5 angles, **auto-calibrated** off the first frontal shot, so the
  left/right/up detection is immune to camera mirroring and per-phone quirks.
- The 5 captured frames upload to the server, where **antelopev2**
  (SCRFD 5-point + ArcFace R100) does the real embedding + 5-point mask.

> The browser mesh is the *experience and the trigger*. antelopev2 on the
> server is the *truth that gets stored*. They are complementary, not the same
> model — the mesh is MediaPipe, the stored mask/embedding is antelopev2.

### Run with antelopev2 (forced regardless of GPU)
```bash
docker compose up -d db
export ATTEND_DB_DSN="postgresql://attend:attend_dev_pw@localhost:5432/attendance"
export ATTEND_MODEL_PACK=antelopev2      # force the R100 accuracy pack
pip install -r registration/requirements.txt
uvicorn registration.api:app --host 0.0.0.0 --port 8000
# phone -> https://<server>/   (camera needs HTTPS; localhost is exempt)
```
`antelopev2` downloads automatically on first run (insightface model zoo).
On CPU it is slower than buffalo_l but accuracy is higher; on a GPU box it is
both fast and most accurate.
