import cv2
import threading
import time
import traceback
from queue import Queue, Empty

INFER_SCALE = 1.0  # detect on full-size frame (better accuracy, slower)
INFER_EVERY = 3 # process every 3rd frame for speed
SIMILARITY_THRESHOLD = 0.40  # Lower threshold for better detection

class VideoWorker:
    def __init__(self, video_path, embedder, matcher, tracker):
        self.video_path = video_path
        self.embedder   = embedder
        self.matcher    = matcher
        self.tracker    = tracker

        self._infer_q   = Queue(maxsize=5)
        self.raw_frame  = None   # latest raw frame, updated at grab rate
        self.detections = []     # [(x1,y1,x2,y2, label, color)] in original frame coords
        self.face_count = 0
        self.running    = True

        cap = cv2.VideoCapture(self.video_path)
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    def start_threads(self):
        threading.Thread(target=self._grab_frames,       daemon=True).start()
        threading.Thread(target=self._process_inference, daemon=True).start()

    def _grab_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        # Skip first few frames to avoid codec delay
        for _ in range(5):
            cap.read()
        
        idx = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.raw_frame = frame
            idx += 1

            # Push to inference queue (skip frames for speed)
            if idx % INFER_EVERY == 0:
                if self._infer_q.full():
                    try:
                        self._infer_q.get_nowait()
                    except Empty:
                        pass
                # Don't block on put - skip if queue is full
                try:
                    self._infer_q.put(frame, block=False)
                except:
                    pass  # Queue full, skip this frame

            # Sleep less to increase frame rate
            time.sleep(1 / (self._fps * 1.5))  # 1.5x faster grab rate
        cap.release()

    def _process_inference(self):
        while self.running:
            try:
                # Timeout prevents indefinite blocking
                frame = self._infer_q.get(timeout=0.1)
                oh, ow = frame.shape[:2]
                sh, sw = int(oh * INFER_SCALE), int(ow * INFER_SCALE)
                small  = cv2.resize(frame, (sw, sh))

                faces  = self.embedder.detect(small)
                
                detections = []
                
                # # DEBUG: Print detection info
                # if len(faces) > 0:
                #     print(f"[DETECTION] Found {len(faces)} face(s) in frame")

                for face in faces:
                    x1 = max(0,  int(face.bbox[0] / INFER_SCALE))
                    y1 = max(0,  int(face.bbox[1] / INFER_SCALE))
                    x2 = min(ow, int(face.bbox[2] / INFER_SCALE))
                    y2 = min(oh, int(face.bbox[3] / INFER_SCALE))

                    student, score = self.matcher.match(face.normed_embedding)

                    if score < SIMILARITY_THRESHOLD:
                        label = "Unknown"
                        color = (60, 60, 255)
                    else:
                        label = f"{student['name']}  {score:.2f}"
                        color = (50, 220, 50)
                        self.tracker.mark(student['roll_no'], student['name'])

                    detections.append((x1, y1, x2, y2, label, color))

                self.face_count = len(faces)
                self.detections = detections

            except Exception as e:
                # Silently handle queue timeout (normal)
                if "Empty" not in str(type(e).__name__):
                    pass  # Ignore queue.Empty timeouts
