# Run on your laptop — quickstart

## Prereqs
- Python 3.10+
- Docker Desktop (for the database)
- A webcam
- (Windows only) install via WSL2 or have Visual C++ Build Tools — insightface
  needs a compiler on native Windows. Mac/Linux/WSL2 = no issue.

## Steps
```bash
tar -xzf attendance_phase1.tar.gz
cd attendance

# 1. database
docker compose up -d db

# 2. python deps  (first install pulls torch/onnx — a few minutes)
pip install -r registration/requirements.txt

# 3. environment
export ATTEND_DB_DSN="postgresql://attend:attend_dev_pw@localhost:5432/attendance"
export ATTEND_MODEL_PACK=antelopev2
# Windows PowerShell:
#   $env:ATTEND_DB_DSN="postgresql://attend:attend_dev_pw@localhost:5432/attendance"
#   $env:ATTEND_MODEL_PACK="antelopev2"

# 4. create a section to enroll into -> copy the printed UUID
python -m registration.seed

# 5. run
uvicorn registration.api:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000/** , paste the Section ID from step 4, enroll.

## Detection service (RTSP attendance)

The detection service is triggered by the backend when a class starts. It
connects to RTSP cameras, matches faces against the section index, and
returns a present list.

```bash
uvicorn detection.api:app --host 0.0.0.0 --port 8010
curl http://localhost:8010/health
```

Run a window (call this 3 times per lecture: start/mid/end):

```bash
curl -X POST http://localhost:8010/window/run \
  -H 'content-type: application/json' \
  -d '{
    "room_id": "CSE-LAB-101",
    "section_id": "11111111-1111-1111-1111-111111111111",
    "window_id": 1,
    "duration_seconds": 45,
    "cameras": [
      {"camera_id": "cam1", "rtsp_url": "rtsp://user:pass@10.0.0.5/stream1"},
      {"camera_id": "cam2", "rtsp_url": "rtsp://user:pass@10.0.0.6/stream1"},
      {"camera_id": "cam3", "rtsp_url": "rtsp://user:pass@10.0.0.7/stream1"}
    ]
  }'
```

Fetch results:

```bash
curl "http://localhost:8010/session/result?room_id=CSE-LAB-101"
```

## First-run waits (normal)
- antelopev2 (~350MB) auto-downloads once.
- MediaPipe mesh loads from CDN when the page opens (needs internet).
- First detection is slow (warmup), then fast.

## No-Docker mode (data NOT saved)
Skip steps 1 and the ATTEND_DB_DSN line. Uses in-memory store.
Section id to use: 11111111-1111-1111-1111-111111111111

## If something breaks
- camera blank -> click Allow on the browser prompt; close Zoom/other apps
- insightface install fails on Windows -> use WSL2
- check the server is up:  curl http://localhost:8000/health
