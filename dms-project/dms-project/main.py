# ─────────────────────────────────────────────────────────────────
# main.py
# Entry point. Ties all modules together.
#
# On startup, prompts the user to choose calibration method:
#   [1] Live camera  — calibrates interactively and records the
#                      session to calib_videos/ for future reuse.
#   [2] Video file   — lists available videos in calib_videos/
#                      and lets the user pick one by number.
# ─────────────────────────────────────────────────────────────────

import cv2
import sys
import time
import os
from datetime import datetime

import config
from landmarks      import get_landmarks, get_all_metrics
from calibration    import calibrate_from_video, calibrate_from_camera
from detection      import Detector, DetectionResult
from display        import draw_warnings, draw_hud, draw_no_face, draw_ready_splash, draw_phone_boxes
from yolo_detection import PhoneDetector

# ── Folder where calibration videos are saved and read from ──────
CALIB_VIDEO_DIR = "calib_videos"

# Supported video extensions to scan
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


# ─────────────────────────────────────────────────────────────────
# RecordingCapture — wrapper that records frames as they are read
# ─────────────────────────────────────────────────────────────────

class RecordingCapture:
    """
    Thin wrapper around cv2.VideoCapture that writes every frame
    to a VideoWriter as it is read.

    calibration.py calls cap.read() — this wrapper intercepts those
    calls to record them without touching calibration.py at all.

    cv2.VideoCapture is a C++ object whose methods are read-only, so
    monkey-patching .read() is not possible. This wrapper sidesteps
    that by implementing .read() itself and forwarding everything else
    via __getattr__ so the object still behaves like a VideoCapture.

    After calibration, call stop_recording() then unwrap() to get the
    original VideoCapture back for the main detection loop.
    """

    def __init__(self, cap):
        os.makedirs(CALIB_VIDEO_DIR, exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path  = os.path.join(CALIB_VIDEO_DIR, f"calib_{timestamp}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 20.0
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._path, fourcc, fps, (w, h))
        self._cap    = cap
        print(f"  Recording calibration session -> {self._path}")

    def __getattr__(self, name):
        # Forward any attribute not defined here to the real VideoCapture
        return getattr(self._cap, name)

    def read(self):
        ret, frame = self._cap.read()
        if ret and frame is not None:
            # calibrate_from_camera() flips the frame itself AFTER calling
            # read() (see calibration.py), so we must return the RAW frame
            # here to avoid double-flipping the on-screen calibration view.
            # But calibrate_from_video() does NOT flip when reading from a
            # saved file later — so the file we write must be the flipped
            # (mirrored) version to match what the live session looked like.
            mirrored = cv2.flip(frame, 1)
            self._writer.write(mirrored)
        return ret, frame

    def stop_recording(self):
        self._writer.release()
        print(f"  Calibration video saved -> {self._path}\n")

    def unwrap(self):
        """Return the underlying VideoCapture for use after calibration."""
        return self._cap


# ─────────────────────────────────────────────────────────────────
# Startup menu
# ─────────────────────────────────────────────────────────────────

def prompt_calibration_choice():
    """
    Ask user how they want to calibrate.
    Returns 'live' or 'video'.
    """
    print("\n" + "=" * 50)
    print("  DRIVER MONITORING SYSTEM")
    print("=" * 50)
    print("\n  How would you like to calibrate?\n")
    print("  [1]  Live camera  (records session for future use)")
    print("  [2]  Video file   (pick from saved calibration videos)")
    print()

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return "live" if choice == "1" else "video"
        print("  Invalid choice - please enter 1 or 2.")


def pick_video_from_directory():
    """
    List all video files in CALIB_VIDEO_DIR and let user pick one.
    Returns the full path to the chosen video, or None if none found.
    """
    os.makedirs(CALIB_VIDEO_DIR, exist_ok=True)

    videos = sorted([
        f for f in os.listdir(CALIB_VIDEO_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ])

    if not videos:
        print(f"\n  No calibration videos found in '{CALIB_VIDEO_DIR}/'.")
        print("  Run with live camera first to record one.\n")
        return None

    print(f"\n  Calibration videos in '{CALIB_VIDEO_DIR}/':\n")
    for i, name in enumerate(videos, start=1):
        full_path = os.path.join(CALIB_VIDEO_DIR, name)
        size_mb   = os.path.getsize(full_path) / (1024 * 1024)
        mtime     = datetime.fromtimestamp(os.path.getmtime(full_path))
        print(f"  [{i}]  {name}  ({size_mb:.1f} MB, saved {mtime.strftime('%Y-%m-%d %H:%M')})")

    print()
    while True:
        raw = input(f"  Enter number (1-{len(videos)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            chosen = os.path.join(CALIB_VIDEO_DIR, videos[int(raw) - 1])
            print(f"\n  Selected: {chosen}\n")
            return chosen
        print(f"  Please enter a number between 1 and {len(videos)}.")


# ─────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────

def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        exit(1)
    for _ in range(10):
        cap.read()
    ret, test = cap.read()
    if not ret:
        print("ERROR: Cannot read from camera.")
        cap.release()
        exit(1)
    h, w = test.shape[:2]
    print(f"Camera: {w}x{h}")
    return cap


# ─────────────────────────────────────────────────────────────────
# Calibration wrappers
# ─────────────────────────────────────────────────────────────────

def do_live_calibration():
    """
    Open camera, calibrate interactively, and record the entire
    calibration session to calib_videos/ for future reuse.
    Returns (cap, calibration) where cap is the raw VideoCapture.
    """
    cap           = open_camera()
    recording_cap = RecordingCapture(cap)

    # calibrate_from_camera receives the wrapper.
    # It calls .read() normally and has no idea frames are being recorded.
    calibration = calibrate_from_camera(recording_cap)

    recording_cap.stop_recording()

    # Unwrap back to the real VideoCapture for the detection loop
    return recording_cap.unwrap(), calibration


def do_video_calibration():
    """
    Let user pick a saved calibration video, calibrate from it,
    then open the camera for the detection loop.
    Returns (cap, calibration), or falls back to live if no video chosen.
    """
    video_path = pick_video_from_directory()

    if video_path is None:
        print("  Falling back to live camera calibration.\n")
        return do_live_calibration()

    calibration = calibrate_from_video(video_path)

    if calibration is None:
        print("  Video calibration failed - falling back to live camera.\n")
        return do_live_calibration()

    cap = open_camera()
    return cap, calibration


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    # ── Calibration source selection ──────────────────────────────
    # Ask BEFORE opening any camera window — nothing should appear
    # on screen until the user has picked live vs. video calibration.
    choice = prompt_calibration_choice()

    cv2.namedWindow(config.WINDOW, cv2.WINDOW_NORMAL)

    if choice == "live":
        cap, calibration = do_live_calibration()
    else:
        cap, calibration = do_video_calibration()

    calibration.print_summary()

    # ── Ready splash ──────────────────────────────────────────────
    splash_start = time.time()
    while time.time() - splash_start < 2.0:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        draw_ready_splash(frame, calibration, calibration.source)
        cv2.imshow(config.WINDOW, frame)
        cv2.waitKey(1)

    print("Detection active. Press ESC to quit.\n")

    # ── Init YOLO phone detector ──────────────────────────────────
    phone_detector = None
    if os.path.exists(config.YOLO_MODEL_PATH):
        phone_detector = PhoneDetector(config.YOLO_MODEL_PATH)
    else:
        print(f"YOLO model not found at '{config.YOLO_MODEL_PATH}' - phone detection disabled.")
        print(f"  Set YOLO_MODEL_PATH in config.py to enable it.\n")

    # ── Main detection loop ───────────────────────────────────────
    detector      = Detector(calibration)
    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        lms   = get_landmarks(frame, w, h)

        # ── YOLO phone detection (sampled) ────────────────────────
        if phone_detector is not None:
            phone_detector.tick()
            if phone_detector.should_run():
                phone_detector.detect(frame)

        if lms:
            no_face_count = 0
            metrics = get_all_metrics(lms)
            result  = detector.process_frame(metrics)
        else:
            no_face_count += 1
            detector.reset_counters()
            result = None
            if no_face_count > 10:
                draw_no_face(frame)

        # ── YOLO phone processing and display ─────────────────────
        # Runs every frame regardless of whether MediaPipe found a face.
        if phone_detector is not None:
            if result is None:
                result = DetectionResult()
            detector.process_phone(phone_detector.last_result, result)
            draw_phone_boxes(frame, phone_detector.last_result)

        if result is not None:
            draw_warnings(frame, result)
            if lms:
                draw_hud(frame, result, calibration)

        cv2.imshow(config.WINDOW, frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
