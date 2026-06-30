import os
import sys
import time
import cv2
import numpy as np
import tkinter as tk

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from database_demo import SessionLocal
from recognition.matcher import Matcher
from recognition.load_students import load_students
from recognition.video_worker import VideoWorker
from recognition.attendance_tracker import AttendanceTracker

# ----------------------------------------------------
# Screen Resolution (macOS Safe)/Windows safe also full screen allow 
# ----------------------------------------------------
_root = tk.Tk()
SCREEN_W = _root.winfo_screenwidth()
SCREEN_H = _root.winfo_screenheight()
_root.destroy()

# ----------------------------------------------------
# Load Database First
# ----------------------------------------------------
session = SessionLocal()

students = load_students(session)

matcher = Matcher(students)

tracker = AttendanceTracker(SessionLocal)

print(f"✅ Loaded {len(students)} students.")

# ----------------------------------------------------
# Load Face Model AFTER DB
# ----------------------------------------------------
from facex.embedder import Embedder

embedder = Embedder(
    model_name="antelopev2",
    use_gpu=True
        # CPU mode - CUDA libraries missing
)
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ----------------------------------------------------
# Camera Workers
# ----------------------------------------------------
left = VideoWorker(
    os.path.join(BASE_DIR, "videos", "left.mp4"),
    embedder,
    matcher,
    tracker
)

middle = VideoWorker(
    os.path.join(BASE_DIR, "videos", "middle.mp4"),
    embedder,
    matcher,
    tracker
)

right = VideoWorker(
    os.path.join(BASE_DIR, "videos", "right.mp4"),
    embedder,
    matcher,
    tracker
)

left.start_threads()
middle.start_threads()
right.start_threads()

print("🎥 All camera workers started.")

# ----------------------------------------------------
# OpenCV
# ----------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX

GRID_WINDOW = "KIET Attendance"

LEFT_WINDOW = "Camera Left"

MIDDLE_WINDOW = "Camera Middle"

RIGHT_WINDOW = "Camera Right"

# ----------------------------------------------------
# Camera Size
# ----------------------------------------------------
CAM_W = 640
CAM_H = 360

HEADER_H = 45

SEPARATOR = 3

TOTAL_W = CAM_W * 3 + SEPARATOR * 2

separator = np.full(
    (CAM_H, SEPARATOR, 3),
    40,
    dtype=np.uint8
)

# ----------------------------------------------------
# Helper
# ----------------------------------------------------
def safe_frame(worker):

    frame = worker.raw_frame

    if frame is None:

        return np.zeros(
            (
                CAM_H,
                CAM_W,
                3
            ),
            dtype=np.uint8
        )

    return frame.copy()

# ----------------------------------------------------
# Keep Aspect Ratio
# ----------------------------------------------------
def resize_keep_ratio(frame, target_w, target_h):

    h, w = frame.shape[:2]

    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        frame,
        (new_w, new_h)
    )

    canvas = np.zeros(
        (
            target_h,
            target_w,
            3
        ),
        dtype=np.uint8
    )

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas[
        y:y+new_h,
        x:x+new_w
    ] = resized

    return canvas

# ----------------------------------------------------
# Draw Detections
# ----------------------------------------------------
def draw_detections(frame, detections, orig_w, orig_h):

    display_h, display_w = frame.shape[:2]

    sx = display_w / orig_w
    sy = display_h / orig_h
    
    # # DEBUG
    # if len(detections) > 0:
    #     print(f"[DRAW] Drawing {len(detections)} detections on {display_w}x{display_h} display")

    for x1, y1, x2, y2, label, color in detections:

        x1 = int(x1 * sx)
        y1 = int(y1 * sy)
        x2 = int(x2 * sx)
        y2 = int(y2 * sy)

        # Bounding Box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        scale = max(
            0.45,
            min(
                0.75,
                (x2 - x1) / 120
            )
        )

        (tw, th), baseline = cv2.getTextSize(
            label,
            FONT,
            scale,
            1
        )

        cv2.rectangle(
            frame,
            (x1, y1 - th - 10),
            (x1 + tw + 8, y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 5),
            FONT,
            scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )


# ----------------------------------------------------
# Camera Label
# ----------------------------------------------------
def camera_label(frame, text):

    cv2.rectangle(
        frame,
        (8, 8),
        (170, 38),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        text,
        (15, 30),
        FONT,
        0.6,
        (0, 220, 255),
        2,
        cv2.LINE_AA
    )


# ----------------------------------------------------
# Compose One Camera
# ----------------------------------------------------
def compose(worker, width, height):

    raw = safe_frame(worker)

    original_h, original_w = raw.shape[:2]

    display = resize_keep_ratio(
        raw,
        width,
        height
    )

    draw_detections(
        display,
        worker.detections,
        original_w,
        original_h
    )

    return display


# ----------------------------------------------------
# Header
# ----------------------------------------------------
def build_header(fps):

    header = np.full(
        (
            HEADER_H,
            TOTAL_W,
            3
        ),
        18,
        dtype=np.uint8
    )

    total_faces = (
        left.face_count
        + middle.face_count
        + right.face_count
    )

    cv2.putText(
        header,
        "KIET Face Attendance Demo",
        (15, 30),
        FONT,
        0.75,
        (0, 220, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        header,
        f"FPS : {fps:.1f}",
        (TOTAL_W - 320, 30),
        FONT,
        0.55,
        (0, 255, 120),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        header,
        f"Faces : {total_faces}",
        (TOTAL_W - 215, 30),
        FONT,
        0.55,
        (255, 210, 80),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        header,
        f"Present : {tracker.present_count}",
        (TOTAL_W - 105, 30),
        FONT,
        0.55,
        (80, 220, 80),
        2,
        cv2.LINE_AA
    )

    return header


# ----------------------------------------------------
# OpenCV Windows
# ----------------------------------------------------
cv2.namedWindow(
    GRID_WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    GRID_WINDOW,
    TOTAL_W,
    CAM_H + HEADER_H
)

# Dedicated camera windows
# (Much better than fullscreen on macOS)

cv2.namedWindow(
    LEFT_WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    MIDDLE_WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    RIGHT_WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    LEFT_WINDOW,
    1500,
    900
)

cv2.resizeWindow(
    MIDDLE_WINDOW,
    1500,
    900
)

cv2.resizeWindow(
    RIGHT_WINDOW,
    1500,
    900
)

# Hide them initially
cv2.moveWindow(LEFT_WINDOW, -3000, -3000)
cv2.moveWindow(MIDDLE_WINDOW, -3000, -3000)
cv2.moveWindow(RIGHT_WINDOW, -3000, -3000)
# ----------------------------------------------------
# Current View
# ----------------------------------------------------
# 0 = Grid
# 1 = Left Camera
# 2 = Middle Camera
# 3 = Right Camera

current_view = 0

WINDOW_W = 1600
WINDOW_H = 900
# ----------------------------------------------------
# FPS
# ----------------------------------------------------
fps = 0.0
frame_counter = 0
last_time = time.time()

print()
print("==========================================")
print("  KIET Attendance Demo")
print("==========================================")
print(" G  -> Grid View")
print(" 1  -> Left Camera")
print(" 2  -> Middle Camera")
print(" 3  -> Right Camera")
print("ESC -> Exit")
print("==========================================")

while True:

    # Don't wait for all frames - start rendering as soon as first frame is available
    # This dramatically speeds up initial detection appearance
    if current_view == 0 and left.raw_frame is None:
        time.sleep(0.01)
        continue

    frame_counter += 1

    now = time.time()

    if now - last_time >= 1.0:

        fps = frame_counter / (now - last_time)

        frame_counter = 0

        last_time = now

    # --------------------------------------------------
    # GRID
    # --------------------------------------------------

    if current_view == 0:

        f1 = compose(left, CAM_W, CAM_H)
        f2 = compose(middle, CAM_W, CAM_H)
        f3 = compose(right, CAM_W, CAM_H)

        camera_label(f1, "CAM 1  LEFT")
        camera_label(f2, "CAM 2  MIDDLE")
        camera_label(f3, "CAM 3  RIGHT")

        row = np.hstack(
            [
                f1,
                separator,
                f2,
                separator,
                f3
            ]
        )

        header = build_header(fps)

        grid = np.vstack(
            [
                header,
                row
            ]
        )

        cv2.imshow(
            GRID_WINDOW,
            grid
        )

    # --------------------------------------------------
    # LEFT
    # --------------------------------------------------

    elif current_view == 1:

        frame = compose(
            left,
            WINDOW_W,
            WINDOW_H
        )

        camera_label(
            frame,
            "LEFT CAMERA"
        )

        cv2.imshow(
            LEFT_WINDOW,
            frame
        )

    # --------------------------------------------------
    # MIDDLE
    # --------------------------------------------------

    elif current_view == 2:

        frame = compose(
            middle,
            WINDOW_W,
            WINDOW_H
        )

        camera_label(
            frame,
            "MIDDLE CAMERA"
        )

        cv2.imshow(
            MIDDLE_WINDOW,
            frame
        )

    # --------------------------------------------------
    # RIGHT
    # --------------------------------------------------

    elif current_view == 3:

        frame = compose(
            right,
            WINDOW_W,
            WINDOW_H
        )

        camera_label(
            frame,
            "RIGHT CAMERA"
        )

        cv2.imshow(
            RIGHT_WINDOW,
            frame
        )

    # --------------------------------------------------
    # Keyboard
    # --------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord("g") or key == ord("G"):

        current_view = 0

        try:
            cv2.destroyWindow(LEFT_WINDOW)
            cv2.destroyWindow(MIDDLE_WINDOW)
            cv2.destroyWindow(RIGHT_WINDOW)
        except:
            pass

        cv2.namedWindow(
            GRID_WINDOW,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            GRID_WINDOW,
            TOTAL_W,
            CAM_H + HEADER_H
        )

    elif key == ord("1"):

        current_view = 1

        try:
            cv2.destroyWindow(GRID_WINDOW)
            cv2.destroyWindow(MIDDLE_WINDOW)
            cv2.destroyWindow(RIGHT_WINDOW)
        except:
            pass

        cv2.namedWindow(
            LEFT_WINDOW,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            LEFT_WINDOW,
            WINDOW_W,
            WINDOW_H
        )

    elif key == ord("2"):

        current_view = 2

        try:
            cv2.destroyWindow(GRID_WINDOW)
            cv2.destroyWindow(LEFT_WINDOW)
            cv2.destroyWindow(RIGHT_WINDOW)
        except:
            pass

        cv2.namedWindow(
            MIDDLE_WINDOW,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            MIDDLE_WINDOW,
            WINDOW_W,
            WINDOW_H
        )

    elif key == ord("3"):

        current_view = 3

        try:
            cv2.destroyWindow(GRID_WINDOW)
            cv2.destroyWindow(LEFT_WINDOW)
            cv2.destroyWindow(MIDDLE_WINDOW)
        except:
            pass

        cv2.namedWindow(
            RIGHT_WINDOW,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            RIGHT_WINDOW,
            WINDOW_W,
            WINDOW_H
        )

# ----------------------------------------------------
# Shutdown
# ----------------------------------------------------

left.running = False
middle.running = False
right.running = False

print("🛑 Stopping camera workers...")


time.sleep(0.5)

cv2.destroyAllWindows()

session.close()

