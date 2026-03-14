"""
camera_monitor/detection.py
===========================
AI Mock Interview System — Camera Monitoring Module

Python  : 3.9+
Libs    : opencv-python >= 4.x
          mediapipe     >= 0.10   (uses new Tasks API — mp.solutions removed)
          ultralytics             (YOLOv8)

Install all dependencies in one line:
    pip3 install opencv-python mediapipe ultralytics

DETECTION FEATURES
------------------
1.  Face detection          — OpenCV Haar Cascade
2.  Multiple people         — "Warning: Multiple people detected"
3.  No face present         — "Please stay in front of the camera"
4.  Eye / gaze direction    — MediaPipe Face Landmarker (Tasks API)
                              "Please focus on the screen"
5.  Phone detection         — YOLOv8 nano
                              "Warning: Mobile phone detected"

NOTE ON MEDIAPIPE 0.10+
-----------------------
MediaPipe 0.10 removed the old mp.solutions API completely.
This file uses the new Tasks API:
    from mediapipe.tasks.python.vision import FaceLandmarker
The FaceLandmarker model file (~5 MB) is downloaded automatically
on first run and cached in camera_monitor/models/.

HOW TO RUN STANDALONE
---------------------
    python3 camera_monitor/detection.py

HOW TO INTEGRATE INTO STREAMLIT
--------------------------------
    from camera_monitor.detection import CameraMonitor

    monitor = CameraMonitor()
    monitor.open_camera()

    # Inside your Streamlit render loop:
    frame, alerts = monitor.process_frame()
    # frame  → st.image(frame, channels="BGR")
    # alerts → for msg in alerts: st.warning(msg)

    monitor.release()
"""

import os
import time
import urllib.request
import cv2
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS  (graceful fallback if a library is missing)
# ═══════════════════════════════════════════════════════════════════════

# ── MediaPipe 0.10+ Tasks API ─────────────────────────────────────────
# The old mp.solutions was removed in 0.10.  We now use FaceLandmarker
# from mediapipe.tasks which works with mediapipe 0.10.x and above.
try:
    import mediapipe as mp
    from mediapipe.tasks          import python        as mp_python
    from mediapipe.tasks.python   import vision        as mp_vision
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
    _MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    _MEDIAPIPE_AVAILABLE = False
    print("[WARNING] mediapipe not found or incompatible.")
    print("          Eye gaze detection will be disabled.")
    print("          Install:  pip3 install mediapipe")

# ── Ultralytics YOLO ─────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not found. Phone detection will be disabled.")
    print("          Install:  pip3 install ultralytics")


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# ── Camera window size and position ───────────────────────────────────
WINDOW_WIDTH   = 400
WINDOW_HEIGHT  = 300
WINDOW_POS_X   = 20     # pixels from left edge of screen
WINDOW_POS_Y   = 60     # pixels from top  edge of screen
WINDOW_TITLE   = "Interview Monitor  |  Q = quit"

# ── Face detection ────────────────────────────────────────────────────
MIN_FACE_SIZE  = (70, 70)

# ── Warning timers ────────────────────────────────────────────────────
NO_FACE_THRESHOLD_SEC   = 2.5   # secs with no face → "stay in front"
GAZE_AWAY_THRESHOLD_SEC = 2.5   # secs looking away → "focus on screen"

# ── Gaze sensitivity ─────────────────────────────────────────────────
# Iris ratio: 0.0 = far left, 0.5 = centre, 1.0 = far right
# Values outside [GAZE_MIN, GAZE_MAX] count as "looking away"
GAZE_MIN = 0.38   # tightened from 0.25 — iris must stay within this range
GAZE_MAX = 0.62   # tightened from 0.75 — to count as "looking at screen"

# ── YOLO phone detection ─────────────────────────────────────────────
PHONE_CLASS_ID       = 67     # COCO class index for "cell phone"

PHONE_CONF_THRESHOLD = 0.25

# ── MediaPipe FaceLandmarker model ───────────────────────────────────
# Downloaded once automatically, stored in camera_monitor/models/
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


# ═══════════════════════════════════════════════════════════════════════
#  COLOURS  (BGR)
# ═══════════════════════════════════════════════════════════════════════
GREEN    = (0,  210,   0)
RED      = (0,    0, 220)
YELLOW   = (0,  215, 255)
ORANGE   = (0,  140, 255)
WHITE    = (255, 255, 255)
BLACK    = (0,    0,   0)
DARK_RED = (0,    0, 139)


# ═══════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _load_face_cascade() -> cv2.CascadeClassifier:
    """
    Load OpenCV's built-in Haar Cascade for frontal face detection.
    The XML ships with every OpenCV install — no download ever needed.
    """
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Haar Cascade not found at: {path}\n"
            "Reinstall:  pip3 install --upgrade opencv-python"
        )
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Haar Cascade file is corrupt. Reinstall opencv-python.")
    print("[INFO] Face detector (Haar Cascade) loaded.")
    return cascade


def _download_landmarker_model():
    """
    Download the MediaPipe FaceLandmarker .task model file if it is not
    already present on disk.

    The file is ~5 MB and is saved to:
        camera_monitor/models/face_landmarker.task

    This runs only once — on every subsequent run the cached file is used.
    """
    if os.path.exists(_MODEL_PATH):
        return   # already downloaded

    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"[INFO] Downloading FaceLandmarker model (~5 MB) ...")
    print(f"       Saving to: {_MODEL_PATH}")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Model downloaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not download model: {e}")
        print("          Eye gaze detection will be disabled this session.")
        # Remove partial download if any
        if os.path.exists(_MODEL_PATH):
            os.remove(_MODEL_PATH)


def _load_face_landmarker():
    """
    Initialise the MediaPipe FaceLandmarker using the new Tasks API
    (required for mediapipe 0.10+).

    HOW THE TASKS API WORKS
    -----------------------
    Instead of mp.solutions (removed in 0.10), we use:
        FaceLandmarker.create_from_options(options)

    RunningMode.IMAGE means we feed it one frame at a time (synchronous),
    which is exactly what we need inside our frame-by-frame loop.

    output_face_blendshapes and output_facial_transformation_matrixes
    are turned off — we only need the 478 landmark points.

    Returns None if mediapipe is unavailable or the model file is missing.
    """
    if not _MEDIAPIPE_AVAILABLE:
        return None

    # Download model if needed
    _download_landmarker_model()

    if not os.path.exists(_MODEL_PATH):
        print("[WARNING] FaceLandmarker model unavailable. Gaze detection off.")
        return None

    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    landmarker = FaceLandmarker.create_from_options(options)
    print("[INFO] MediaPipe FaceLandmarker (eye gaze detector) loaded.")
    return landmarker


def _load_yolo():
    """
    Load YOLOv8 nano for phone detection.
    ~6 MB, downloaded automatically by ultralytics on first run.
    Returns None if ultralytics is not installed.
    """
    if not _YOLO_AVAILABLE:
        return None
    model = YOLO("yolov8n.pt")
    print("[INFO] YOLOv8 nano model loaded for phone detection.")
    return model


# ═══════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════

class CameraMonitor:
    """
    Manages the webcam feed and runs all monitoring checks each frame.

    STANDALONE:
        monitor = CameraMonitor()
        monitor.start()

    STREAMLIT:
        monitor = CameraMonitor()
        monitor.open_camera()
        while running:
            frame, alerts = monitor.process_frame()
            col.image(frame, channels="BGR")
            for msg in alerts:
                st.warning(msg)
        monitor.release()
    """

    def __init__(self):
        self.face_cascade  = _load_face_cascade()
        self.face_lm       = _load_face_landmarker()   # None if unavailable
        self.yolo_model    = _load_yolo()              # None if unavailable

        # Warning timers — store the timestamp when a condition first appeared
        self.no_face_since:   Optional[float] = None
        self.gaze_away_since: Optional[float] = None

        self.cap: Optional[cv2.VideoCapture] = None

    # ─────────────────────────────────────────────
    #  Camera open / release
    # ─────────────────────────────────────────────

    def open_camera(self, camera_index: int = 0) -> bool:
        """Open webcam. Returns True on success."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(
                "[ERROR] Could not open webcam.\n"
                "  -> macOS: System Settings -> Privacy & Security -> Camera\n"
                "  -> Make sure no other app is already using the camera."
            )
            return False
        print("[INFO] Webcam opened.")
        return True

    def release(self):
        """Release webcam and close all windows."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released.")

    # ─────────────────────────────────────────────
    #  Core method — call this every frame
    # ─────────────────────────────────────────────

    def process_frame(self) -> Tuple[Optional[object], List[str]]:
        """
        Capture one frame, run all detections, return annotated frame + alerts.

        Returns
        -------
        frame  : BGR numpy array with boxes/banners drawn on it
                 None if the camera dropped a frame.
        alerts : list[str] — zero or more active warning messages.
                 Also printed to the console automatically.
        """
        if self.cap is None or not self.cap.isOpened():
            return None, []

        success, frame = self.cap.read()
        if not success or frame is None:
            return None, []

        frame = cv2.flip(frame, 1)   # mirror

        # Save the original full-resolution frame for YOLO phone detection.
        # YOLO needs higher resolution to reliably spot small objects like phones.
        # We resize AFTER saving this reference.
        original_frame = frame.copy()
        orig_h, orig_w = original_frame.shape[:2]

        frame    = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        frame_h, frame_w = frame.shape[:2]
        alerts: List[str] = []

        # Prepare colour versions for each detector
        grey = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run detectors ─────────────────────────────────────────────
        num_faces = self._detect_faces(frame, grey)
        self._check_face_warnings(alerts, num_faces)

        if num_faces == 1 and self.face_lm is not None:
            self._detect_gaze(frame, rgb, alerts)

        if self.yolo_model is not None:
            # Pass original_frame (full res) + scale factors so _detect_phone
            # can draw correctly-sized boxes onto the small display frame
            scale_x = frame_w / orig_w
            scale_y = frame_h / orig_h
            self._detect_phone(frame, original_frame, alerts, scale_x, scale_y)

        # ── Draw overlays ─────────────────────────────────────────────
        self._draw_status_bar(frame, num_faces)
        self._draw_alert_banners(frame, alerts, frame_w, frame_h)

        for msg in alerts:
            print(f"  [ALERT] {msg}")

        return frame, alerts

    # ─────────────────────────────────────────────
    #  DETECTOR 1 — Face detection (Haar Cascade)
    # ─────────────────────────────────────────────

    def _detect_faces(self, frame, grey) -> int:
        """Detect faces, draw boxes, return count."""
        faces = self.face_cascade.detectMultiScale(
            grey,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        num_faces = len(faces) if (faces is not None and len(faces) > 0) else 0

        if num_faces > 0:
            for (x, y, w, h) in faces:
                colour = GREEN if num_faces == 1 else RED
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                label = "You" if num_faces == 1 else "Extra person"
                cv2.putText(
                    frame, label, (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA,
                )
        return num_faces

    # ─────────────────────────────────────────────
    #  DETECTOR 2 — Face presence warnings
    # ─────────────────────────────────────────────

    def _check_face_warnings(self, alerts: List[str], num_faces: int):
        """Generate warnings based on face count."""
        if num_faces > 1:
            alerts.append("Warning: Multiple people detected")
            self.no_face_since = None

        elif num_faces == 0:
            if self.no_face_since is None:
                self.no_face_since = time.time()
            if time.time() - self.no_face_since >= NO_FACE_THRESHOLD_SEC:
                alerts.append("Please stay in front of the camera")

        else:
            self.no_face_since = None   # one face — all good

    # ─────────────────────────────────────────────
    #  DETECTOR 3 — Eye gaze (MediaPipe Tasks API)
    # ─────────────────────────────────────────────

    def _detect_gaze(self, frame, rgb, alerts: List[str]):
        """
        Estimate gaze direction using MediaPipe FaceLandmarker (0.10+ API).

        HOW IT WORKS
        ─────────────
        The FaceLandmarker model returns 478 facial landmarks.
        Landmarks 468-477 are the iris points (left + right).

        Key indices used:
            468 = left  iris centre
            473 = right iris centre
             33 = left  eye outer corner
            133 = left  eye inner corner
            362 = right eye inner corner
            263 = right eye outer corner

        We compute an "iris ratio" for each eye:
            ratio = (iris_x - inner_corner_x) / eye_width
            0.0 = far left, 0.5 = centred, 1.0 = far right

        If both eyes are outside the safe zone [GAZE_MIN, GAZE_MAX]
        for more than GAZE_AWAY_THRESHOLD_SEC seconds, the warning fires.

        NEW TASKS API vs OLD SOLUTIONS API
        ------------------------------------
        Old (broken on 0.10+):
            results = face_mesh.process(rgb)
            lm = results.multi_face_landmarks[0].landmark

        New (correct for 0.10+):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)
            lm       = result.face_landmarks[0]   # list of NormalizedLandmark
        """
        # Wrap the numpy RGB array in a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection
        result = self.face_lm.detect(mp_image)

        # No face landmarks found — skip
        if not result.face_landmarks:
            return

        lm = result.face_landmarks[0]   # list of NormalizedLandmark objects
        frame_h, frame_w = frame.shape[:2]

        # Helper: normalised [0,1] landmark → pixel (x, y)
        def px(idx: int) -> Tuple[int, int]:
            return int(lm[idx].x * frame_w), int(lm[idx].y * frame_h)

        # ── Eye landmark pixel positions ──────────────────────────────
        l_iris  = px(468)    # left  iris centre
        r_iris  = px(473)    # right iris centre
        l_outer = px(33)     # left  eye outer corner
        l_inner = px(133)    # left  eye inner corner
        r_inner = px(362)    # right eye inner corner
        r_outer = px(263)    # right eye outer corner

        # ── Iris ratio ────────────────────────────────────────────────
        def iris_ratio(iris_px, left_corner, right_corner) -> float:
            eye_w = right_corner[0] - left_corner[0]
            if eye_w <= 0:
                return 0.5   # degenerate case — assume centred
            r = (iris_px[0] - left_corner[0]) / eye_w
            return max(0.0, min(1.0, r))

        left_ratio  = iris_ratio(l_iris, l_outer, l_inner)
        right_ratio = iris_ratio(r_iris, r_inner, r_outer)

        # ── Gaze decision ─────────────────────────────────────────────
        left_away  = left_ratio  < GAZE_MIN or left_ratio  > GAZE_MAX
        right_away = right_ratio < GAZE_MIN or right_ratio > GAZE_MAX

        
        looking_away = left_away or right_away

        # ── Draw semi-transparent iris dots ──────────────────────────
        dot_colour = (0, 200, 100) if not looking_away else (0, 200, 255)
        overlay = frame.copy()
        for iris_pt in (l_iris, r_iris):
            cv2.circle(overlay, iris_pt, 3, dot_colour, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # ── Timer-based warning ───────────────────────────────────────
        if looking_away:
            # Only log to terminal when gaze-away is first detected
            if self.gaze_away_since is None:
                self.gaze_away_since = time.time()
                print(f"  [GAZE] User looked away  "
                      f"(left={left_ratio:.2f}, right={right_ratio:.2f})")

            # Only append the alert once — the moment the threshold is crossed
            elapsed = time.time() - self.gaze_away_since
            if elapsed >= GAZE_AWAY_THRESHOLD_SEC:
                # _gaze_warned flag prevents repeating the print every frame
                if not getattr(self, "_gaze_warned", False):
                    print("  [ALERT] Please focus on the screen")
                    self._gaze_warned = True
                alerts.append("Please focus on the screen")
        else:
            # User looked back — reset everything
            if getattr(self, "_gaze_warned", False):
                print("  [GAZE] User looked back at screen")
            self.gaze_away_since = None
            self._gaze_warned    = False

    # ─────────────────────────────────────────────
    #  DETECTOR 4 — Phone detection (YOLOv8)
    # ─────────────────────────────────────────────

    def _detect_phone(self, frame, original_frame, alerts: List[str],
                      scale_x: float = 1.0, scale_y: float = 1.0):
        """
        Use YOLOv8 nano to detect mobile phones (COCO class 67).

        WHY WE USE original_frame
        --------------------------
        The display frame is resized to 400x300 which is too small for YOLO
        to reliably detect phones.  We run YOLO on the full-resolution frame
        captured from the webcam, then multiply the bounding box coordinates
        by scale_x / scale_y to draw them correctly on the small display frame.
        """
        # Run YOLO on the full-resolution original frame (BGR, as YOLO expects)
        results = self.yolo_model.predict(original_frame, verbose=False)
        phone_found = False

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != PHONE_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                if conf < PHONE_CONF_THRESHOLD:
                    continue
                phone_found = True

                # Scale bounding box from original resolution → display resolution
                x1 = int(box.xyxy[0][0] * scale_x)
                y1 = int(box.xyxy[0][1] * scale_y)
                x2 = int(box.xyxy[0][2] * scale_x)
                y2 = int(box.xyxy[0][3] * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 2)
                cv2.putText(
                    frame, f"Phone {conf:.0%}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, ORANGE, 1, cv2.LINE_AA,
                )

        if phone_found:
            alerts.append("Warning: Mobile phone detected")

    # ─────────────────────────────────────────────
    #  Overlay helpers
    # ─────────────────────────────────────────────

    def _draw_status_bar(self, frame, num_faces: int):
        """Top-left badge: face count coloured green / yellow / red."""
        colour = GREEN if num_faces == 1 else (RED if num_faces > 1 else YELLOW)
        cv2.rectangle(frame, (0, 0), (150, 28), BLACK, -1)
        cv2.putText(
            frame, f"Faces: {num_faces}", (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2, cv2.LINE_AA,
        )

    def _draw_alert_banners(self, frame, alerts: List[str],
                             frame_w: int, frame_h: int):
        """
        Solid warning banners stacked at the bottom of the frame.
        Dark-red background + white border + yellow shadowed text.
        """
        BANNER_H = 34
        GAP      = 5

        for i, text in enumerate(alerts):
            bottom = frame_h - 8  - i * (BANNER_H + GAP)
            top    = bottom - BANNER_H
            if top < 0:
                continue
            cv2.rectangle(frame, (0, top), (frame_w, bottom), DARK_RED, -1)
            cv2.rectangle(frame, (0, top), (frame_w, bottom), WHITE,    1)
            ty = top + int(BANNER_H * 0.68)
            cv2.putText(frame, text, (11, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, BLACK,  3, cv2.LINE_AA)
            cv2.putText(frame, text, (10, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, YELLOW, 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────
    #  Standalone blocking loop
    # ─────────────────────────────────────────────

    def start(self, camera_index: int = 0):
        """Run the monitoring loop until Q is pressed. Use for standalone testing."""
        if not self.open_camera(camera_index):
            return

        print("[INFO] Monitoring started. Press Q inside the window to stop.")
        print("-" * 55)

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.moveWindow(WINDOW_TITLE, WINDOW_POS_X, WINDOW_POS_Y)

        while True:
            frame, _ = self.process_frame()
            if frame is None:
                print("[ERROR] Lost camera feed.")
                break
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                print("[INFO] Q pressed — shutting down.")
                break

        self.release()


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    monitor = CameraMonitor()
    monitor.start()