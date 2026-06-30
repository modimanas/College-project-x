# Detection Module (Attendance Windows)

This folder contains the **attendance detection** service.

It is designed for **attendance**, not continuous surveillance.
So it processes video only during a few short **attendance windows** in a lecture.

---

## 1) What this service does (in one line)

Backend tells the service:

> “Room X is running Section Y right now. Use these 3 RTSP cameras. Run attendance for ~45 seconds.”

The service returns:

> “These students were present (seen in at least 2 out of 3 windows).”

---

## 2) Why we use attendance windows
  
A lecture can be 50 minutes. Processing all 50 minutes is:

- expensive (CPU/GPU)
- unnecessary for attendance
- looks like surveillance

Instead we run **3 short windows** per lecture:

- **Window 1**: start of lecture
- **Window 2**: middle of lecture
- **Window 3**: end of lecture

Each window runs for **30–45 seconds** (default is **45 seconds**).

---

## 3) Attendance rule (very simple)

For each student we store:

- `windows_seen`: which windows the student appeared in (example: `{1, 3}`)
- `best_confidence`: best matching score we ever got for that student
- `total_seen_time`: total tracked time across the windows (seconds)

Final rule:

- **PRESENT** if the student appears in **at least 2 out of 3** windows.
- Otherwise **ABSENT**.

Why this rule is good:

- A single wrong match (one moment) is less likely to happen twice.
- Students who are actually in class will show up in multiple moments.

---

## 4) What happens during a window (pipeline)

Each attendance window runs this pipeline:

1. **Read frames from RTSP** (camera video)
2. **Detect faces** in the frame (SCRFD via InsightFace)
3. **Track faces** inside each camera (so the same face is not treated as “new” every frame)
4. **Embed + recognize** faces (ArcFace via InsightFace)
5. **Match only within the active section** (FAISS index per section)
6. **Update the student’s window evidence** (`windows_seen`, etc.)

Important: the system does **not** do cross-camera tracking.

- Camera 1 has its own tracks
- Camera 2 has its own tracks
- Camera 3 has its own tracks

We merge results only using the final identity: `student_id`.

---

## 5) Tracking (what it means)

Tracking means:

> When a face is detected, we give it a temporary ID like `track_12` and keep
> following it across the next frames.

This helps because:

- We don’t re-embed every frame (saves CPU/GPU)
- We can measure “this face was visible for ~2 seconds”

In this repo we ship a lightweight tracker (`IoUTracker`) that is easy to
replace later with a full ByteTrack implementation.

---

## 6) How we reduce CPU usage

The expensive parts are:

- Face detection
- Embedding/recognition

So we reduce how often we run them:

### A) Detection every N frames

We do not detect on every frame.
We detect on every `ATTEND_DET_EVERY_N` frames.

Default:

- `ATTEND_DET_EVERY_N=10`

### B) Embedding cooldown per track

Even inside a track, we don’t embed every time.
We embed at most once every `ATTEND_EMBED_COOLDOWN_S` seconds.

Default:

- `ATTEND_EMBED_COOLDOWN_S=1.5`

### C) Matching only the active section

Matching is done only against the **active section’s** FAISS index.
This keeps search fast and reduces wrong matches.

---

## 7) GPU vs CPU (automatic)

The service automatically selects models:

- If GPU provider is available → uses `antelopev2`
- Otherwise → uses `buffalo_l`

Overrides:

- `ATTEND_MODEL_PACK=antelopev2` (force)
- `ATTEND_FORCE_CPU=1` (force CPU)

---

## 8) API usage (backend integration)

Run the service:

```bash
uvicorn detection.api:app --host 0.0.0.0 --port 8010
```

Health check:

```bash
curl http://localhost:8010/health
```

Run **Window 1** (start):

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

Backend repeats the same call for:

- `window_id=2` (middle)
- `window_id=3` (end)

Check if a window is running:

```bash
curl "http://localhost:8010/window/status?room_id=CSE-LAB-101"
```

Fetch attendance result:

```bash
curl "http://localhost:8010/session/result?room_id=CSE-LAB-101"
```

---

## 9) Where the code lives

- `detection/api.py` — HTTP API
- `detection/window_manager.py` — stores window evidence (`windows_seen`, etc.)
- `detection/worker.py` — runs one window: RTSP → detect → track → embed → match
- `detection/tracker.py` — tracker implementation (easy to swap later)
- `detection/matcher.py` — section FAISS matching (reuses registration index)
- `detection/config.py` — all runtime knobs (CPU/GPU + thresholds)

