# ─────────────────────────────────────────────────────────────────
# main.py
# Entry point. Ties all modules together.
#
# Usage:
#   python main.py                        → runtime calibration
#   python main.py path/to/calib.mp4      → calibrate from video
#
# Or set CALIBRATION_VIDEO below to hardcode a path.
# ─────────────────────────────────────────────────────────────────

import cv2
import sys
import time
import os

import config
from landmarks      import get_landmarks, get_all_metrics
from calibration    import calibrate_from_video, calibrate_from_camera
from detection      import Detector
from display        import draw_warnings, draw_hud, draw_no_face, draw_ready_splash, draw_phone_boxes
from yolo_detection import PhoneDetector

# ── Optional: hardcode a calibration video path here ─────────────
# Leave as None to use runtime calibration when no CLI arg given.
CALIBRATION_VIDEO = None
# CALIBRATION_VIDEO = r"C:\path\to\your\calib_clip.mp4"


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


def main():
    cv2.namedWindow(config.WINDOW, cv2.WINDOW_NORMAL)

    # ── Determine calibration source ─────────────────────────────
    cal_video = CALIBRATION_VIDEO
    if len(sys.argv) > 1:
        cal_video = sys.argv[1]

    calibration = None

    if cal_video:
        # Path 1: calibrate from video, then open camera
        calibration = calibrate_from_video(cal_video)
        if calibration is None:
            print("Video calibration failed — falling back to runtime calibration.")
            cal_video = None

    if calibration is None:
        # Path 2: open camera first, then calibrate interactively
        cap = open_camera()
        calibration = calibrate_from_camera(cap)
    else:
        # Video calibration succeeded — open camera now
        cap = open_camera()

    calibration.print_summary()

    # ── Show ready splash ─────────────────────────────────────────
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
    # Gracefully disabled if the model file doesn't exist yet
    phone_detector = None
    if os.path.exists(config.YOLO_MODEL_PATH):
        phone_detector = PhoneDetector(config.YOLO_MODEL_PATH)
    else:
        print(f"YOLO model not found at '{config.YOLO_MODEL_PATH}' — phone detection disabled.")
        print(f"  Set YOLO_MODEL_PATH in config.py to enable it.\n")

    # ── Main detection loop ───────────────────────────────────────
    detector      = Detector(calibration)
    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame    = cv2.flip(frame, 1)
        h, w     = frame.shape[:2]
        lms      = get_landmarks(frame, w, h)

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
        # Phone detection must not depend on face tracking.
        if phone_detector is not None:
            if result is None:
                # No face — create a minimal result just to carry phone state
                from detection import DetectionResult
                result = DetectionResult()
            detector.process_phone(phone_detector.last_result, result)
            # Draw boxes before warnings so warnings render on top
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