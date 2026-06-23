# ─────────────────────────────────────────────────────────────────
# calibration.py
# Two calibration paths:
#   1. calibrate_from_video(path) — offline, from a recorded clip
#   2. calibrate_from_camera(cap) — interactive, at runtime
#
# Both return a CalibrationResult dataclass.
# main.py picks which one to use.
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import time
import os
from dataclasses import dataclass

import config
from landmarks import get_landmarks, get_all_metrics, calculate_avg_ear


# ── Result container ──────────────────────────────────────────────
@dataclass
class CalibrationResult:
    ear_threshold:        float
    mean_ear_open:        float
    look_side_threshold:  int
    baseline_x:           float
    baseline_face_h:      float
    baseline_nose_ratio:  float
    source:               str    # "video" or "runtime"

    def print_summary(self):
        print(f"\n  Calibration source     : {self.source}")
        print(f"  EAR open mean          : {self.mean_ear_open:.3f}")
        print(f"  EAR threshold          : {self.ear_threshold}")
        print(f"  Lateral threshold      : {self.look_side_threshold}px")
        print(f"  Baseline X             : {self.baseline_x:.1f}")
        print(f"  Baseline face height   : {self.baseline_face_h:.1f}px")
        print(f"  Baseline nose ratio    : {self.baseline_nose_ratio:.3f}")
        print(f"  Head-down confirmed if : pitch<{config.PITCH_THRESHOLD}"
              f" AND nose_ratio>{self.baseline_nose_ratio + config.NOSE_RATIO_DOWN_THRESHOLD:.3f}\n")


# ── Shared helpers ────────────────────────────────────────────────

def _dark_bar(frame, y_end=135):
    roi = frame[0:y_end, :]
    frame[0:y_end, :] = cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0)


def _pump(frame, window=config.WINDOW):
    cv2.imshow(window, frame)
    return cv2.waitKey(1) == 27


def _compute_thresholds(ear_series, lateral_vals, face_h_vals, nose_ratio_vals):
    """
    Shared threshold calculation used by both calibration paths.
    ear_series : list of floats (one per frame, None for frames without face)
    Returns a partial CalibrationResult (source filled by caller).
    """
    valid_ears  = [e for e in ear_series if e is not None]
    sorted_ears = sorted(valid_ears)
    n           = len(sorted_ears)

    # Open-eye EAR = top 30% of frames (driver alert, not blinking)
    top_start    = int(n * config.EAR_OPEN_PERCENTILE)
    mean_ear_open = float(np.mean(sorted_ears[top_start:]))

    # Closed-eye EAR = bottom 5% of frames (blink dips)
    bottom_end    = max(1, int(n * config.EAR_CLOSED_PERCENTILE))
    mean_ear_closed = float(np.mean(sorted_ears[:bottom_end]))

    if mean_ear_closed < mean_ear_open * config.EAR_BLINK_RATIO:
        # Real blinks detected — use measured values
        ear_threshold = mean_ear_closed + (mean_ear_open - mean_ear_closed) * config.EAR_THRESHOLD_BIAS
    else:
        # No clear blinks — fall back to ratio formula
        ear_threshold = mean_ear_open * config.EAR_FALLBACK_RATIO

    ear_threshold = float(np.clip(
        round(ear_threshold, 3),
        config.EAR_THRESHOLD_MIN,
        config.EAR_THRESHOLD_MAX
    ))

    # Pose baselines — use median for robustness against brief movements
    baseline_x          = float(np.median(lateral_vals))
    baseline_face_h     = float(np.median(face_h_vals))
    baseline_nose_ratio = float(np.median(nose_ratio_vals))

    pose_std_x          = float(np.std(lateral_vals))
    look_side_threshold = max(
        int(pose_std_x * config.SIDE_THRESHOLD_STD_MULTIPLIER),
        config.SIDE_THRESHOLD_MIN
    )

    return CalibrationResult(
        ear_threshold       = ear_threshold,
        mean_ear_open       = mean_ear_open,
        look_side_threshold = look_side_threshold,
        baseline_x          = baseline_x,
        baseline_face_h     = baseline_face_h,
        baseline_nose_ratio = baseline_nose_ratio,
        source              = "unknown",
    )


# ─────────────────────────────────────────────────────────────────
# PATH 1 — VIDEO CALIBRATION
# Pass a short clip of the driver sitting normally.
# The function processes every frame silently and derives
# all thresholds automatically — no user interaction needed.
# ─────────────────────────────────────────────────────────────────

def calibrate_from_video(video_path):
    """
    Calibrate from a pre-recorded video file.
    Returns CalibrationResult, or None on failure.

    The video should show the driver:
    - Eyes open, looking at the road (neutral pose)
    - Natural blinks will be detected automatically
    - Does NOT need to be mirrored — processed as-is

    Tip: a dashcam clip of the first 30 seconds of a drive works perfectly.
    """
    print(f"\nCalibrating from video: {video_path}")

    if not os.path.exists(video_path):
        print(f"  ERROR: File not found: {video_path}")
        return None

    vcap = cv2.VideoCapture(video_path)
    if not vcap.isOpened():
        print("  ERROR: Cannot open video.")
        return None

    fps   = vcap.get(cv2.CAP_PROP_FPS) or 30
    total = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_w   = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h   = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {v_w}x{v_h}  {fps:.1f}fps  {total} frames")

    ear_series      = []
    lateral_vals    = []
    face_h_vals     = []
    nose_ratio_vals = []
    frames_with_face = 0

    CAL_WIN = "Calibrating from video..."
    cv2.namedWindow(CAL_WIN, cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        ret, frame = vcap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        lms  = get_landmarks(frame, w, h)

        if lms:
            frames_with_face += 1
            metrics = get_all_metrics(lms)
            ear_series.append(metrics["ear"])
            lateral_vals.append(metrics["lateral_offset"])
            face_h_vals.append(metrics["face_height"])
            nose_ratio_vals.append(metrics["nose_ratio"])
        else:
            ear_series.append(None)

        # Progress display
        progress = frame_idx / max(total - 1, 1)
        bar_w    = int(progress * (w - 40))
        _dark_bar(frame, 85)
        cv2.putText(frame, f"Calibrating from video...  {int(progress*100)}%",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Frames with face: {frames_with_face}",
            (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)
        cv2.rectangle(frame, (20, 68), (w - 20, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 68), (20 + bar_w, 80), (0, 200, 100), -1)
        _pump(frame, CAL_WIN)
        frame_idx += 1

    vcap.release()
    cv2.destroyWindow(CAL_WIN)

    if frames_with_face < 20:
        print(f"  ERROR: Only {frames_with_face} frames had a face. Check the video.")
        return None

    print(f"  Processed {frame_idx} frames — {frames_with_face} with face.")

    result = _compute_thresholds(ear_series, lateral_vals, face_h_vals, nose_ratio_vals)
    result.source = "video"
    return result


# ─────────────────────────────────────────────────────────────────
# PATH 2 — RUNTIME CALIBRATION
# Three interactive phases shown on screen.
# Used when no calibration video is available.
# ─────────────────────────────────────────────────────────────────

def _run_phase(cap, title, instruction, bar_color, collect_fn,
               secs=4, min_samples=20, extra_height=0):
    """
    Generic timed calibration phase.
    collect_fn(lms) -> value to collect, or None to skip.
    Retries automatically if not enough face frames captured.
    Returns list of collected values.
    """
    while True:
        samples, start = [], time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]
            lms     = get_landmarks(frame, w, h)
            elapsed = time.time() - start

            bar_end = 135 + extra_height
            _dark_bar(frame, bar_end)
            cv2.putText(frame, title,
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
            cv2.putText(frame, instruction,
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
            cv2.putText(frame, f"{max(0, secs - elapsed):.1f}s  |  {len(samples)} samples",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)

            if lms:
                val = collect_fn(lms, frame, w)
                if val is not None:
                    samples.append(val)
                bar_w = int(min(elapsed / secs, 1.0) * (w - 40))
                cv2.rectangle(frame, (20, 108), (w - 20, 122), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, 108), (20 + bar_w, 122), bar_color, -1)
            else:
                cv2.putText(frame, "No face — move closer or fix lighting",
                    (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 80, 255), 2)

            if _pump(frame):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
            if elapsed >= secs:
                break

        if len(samples) >= min_samples:
            return samples

        # Not enough — show retry message and loop
        print(f"  Only {len(samples)} samples, retrying...")
        t = time.time()
        while time.time() - t < 1.5:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            _dark_bar(frame, 80)
            cv2.putText(frame, f"Not enough samples ({len(samples)}), retrying...",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 80, 255), 2)
            _pump(frame)


def calibrate_from_camera(cap):
    """
    Interactive 3-phase runtime calibration.
    cap: already-opened cv2.VideoCapture.
    Returns CalibrationResult.
    """
    print("\nStarting runtime calibration...")

    # ── Phase 1 — open-eye EAR ────────────────────────────────────
    print("[1/3] Open-eye EAR...")

    def collect_ear(lms, frame, w):
        return calculate_avg_ear(lms)

    ear_open_samples = _run_phase(
        cap, "STEP 1/3  -  open-eye calibration",
        "Eyes OPEN  |  look straight at camera",
        (0, 220, 100), collect_ear,
    )
    mean_ear_open = float(np.mean(ear_open_samples))
    print(f"  Open-eye EAR mean: {mean_ear_open:.3f}")

    # ── Phase 2 — blink EAR ───────────────────────────────────────
    print("[2/3] Blink calibration — blink slowly 3 times...")

    blink_samples = []
    blink_start   = time.time()
    BLINK_SECS    = 8

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        lms     = get_landmarks(frame, w, h)
        elapsed = time.time() - blink_start

        _dark_bar(frame, 150)
        cv2.putText(frame, "STEP 2/3  -  blink calibration",
            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
        cv2.putText(frame, "Blink slowly  3 times  (close eyes fully each time)",
            (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        cv2.putText(frame, f"{max(0, BLINK_SECS - elapsed):.1f}s remaining",
            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)

        if lms:
            ear = calculate_avg_ear(lms)
            blink_samples.append(ear)
            bar_w = int(min(elapsed / BLINK_SECS, 1.0) * (w - 40))
            cv2.rectangle(frame, (20, 108), (w - 20, 122), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 108), (20 + bar_w, 122), (0, 180, 255), -1)
            ear_bar = int(np.clip(ear / max(mean_ear_open, 0.01), 0, 1) * (w - 40))
            ear_col = (0, 200, 80) if ear > mean_ear_open * 0.7 else (0, 80, 255)
            cv2.rectangle(frame, (20, 128), (w - 20, 142), (30, 30, 30), -1)
            cv2.rectangle(frame, (20, 128), (20 + ear_bar, 142), ear_col, -1)
            label = "-- CLOSED --" if ear < mean_ear_open * 0.7 else "open"
            cv2.putText(frame, f"EAR: {ear:.3f}  {label}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        else:
            cv2.putText(frame, "No face detected",
                (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 80, 255), 2)

        if _pump(frame):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
        if elapsed >= BLINK_SECS:
            break

    # Build ear_series from blink phase — Nones for missed frames
    # (blink_samples has no Nones since we only append when face found,
    # so pass it directly as ear_series — _compute_thresholds handles it)
    print(f"  Collected {len(blink_samples)} blink samples.")

    # ── Phase 3 — head pose baseline ─────────────────────────────
    print("[3/3] Pose calibration — look at road position...")

    def collect_pose(lms, frame, w):
        metrics = get_all_metrics(lms)
        return (metrics["lateral_offset"], metrics["face_height"], metrics["nose_ratio"])

    pose_samples = _run_phase(
        cap, "STEP 3/3  -  head pose calibration",
        "Look at where the ROAD would be  (NOT this screen)",
        (0, 200, 255), collect_pose,
    )

    xs     = [s[0] for s in pose_samples]
    hs     = [s[1] for s in pose_samples]
    ratios = [s[2] for s in pose_samples]

    # Build a combined ear_series: open-eye samples + blink samples
    combined_ear_series = ear_open_samples + blink_samples

    result = _compute_thresholds(combined_ear_series, xs, hs, ratios)
    result.source = "runtime"
    return result
