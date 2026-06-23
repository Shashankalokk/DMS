# ─────────────────────────────────────────────────────────────────
# display.py
# Everything drawn on screen.
# Takes a frame + DetectionResult and draws in-place.
# No detection logic here — pure rendering only.
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import config


def draw_warnings(frame, result):
    """
    Draw active warning banners at the bottom of the frame.
    Stacked upward so multiple warnings don't overlap.
    """
    h, w = frame.shape[:2]
    y    = h - 30

    if result.warn_fatigue:
        cv2.rectangle(frame, (0, y - 42), (w, y + 8), (0, 0, 140), -1)
        cv2.putText(frame, "!! FATIGUE ALERT !!",
            (w // 2 - 175, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        y -= 55

    if result.warn_drowsy:
        cv2.rectangle(frame, (0, y - 38), (w, y + 8), (0, 0, 90), -1)
        cv2.putText(frame, "DROWSY",
            (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 60, 255), 3)
        y -= 50

    if result.warn_yawn:
        cv2.rectangle(frame, (0, y - 38), (w, y + 8), (0, 70, 70), -1)
        cv2.putText(frame, "YAWNING",
            (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        y -= 50

    if result.warn_side:
        cv2.rectangle(frame, (0, y - 38), (w, y + 8), (0, 70, 70), -1)
        cv2.putText(frame, f"LOOKING {result.warn_side_dir}",
            (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        y -= 50

    if result.warn_phone:
        cv2.rectangle(frame, (0, y - 38), (w, y + 8), (0, 0, 100), -1)
        cv2.putText(frame, f"PHONE DETECTED  {result.phone_conf:.0%}",
            (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 80, 255), 3)
        y -= 50

    if result.warn_down:
        cv2.rectangle(frame, (0, y - 38), (w, y + 8), (0, 55, 70), -1)
        cv2.putText(frame, "LOOKING DOWN",
            (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3)


def draw_phone_boxes(frame, yolo_result):
    """
    Draw YOLO bounding boxes around detected phones.
    Called every frame — uses last_result so boxes persist between samples.

    Boxes are stored as (x1, y1, x2, y2, conf, class_name) — 6-tuples.
    We only draw when phone_detected is True (hit streak confirmed).
    """
    if not yolo_result.phone_detected:
        return
    for (x1, y1, x2, y2, conf, class_name) in yolo_result.boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Phone {conf:.0%}",
            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_hud(frame, result, calibration):
    """
    Semi-transparent HUD in the top-left corner.
    Shows live metric values and progress bars for EAR and pitch.
    """
    hud_h, hud_w = 200, 280
    roi = frame[0:hud_h, 0:hud_w]
    frame[0:hud_h, 0:hud_w] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)

    cal = calibration

    # ── EAR bar ───────────────────────────────────────────────────
    ear_pct = min(result.ear / max(cal.mean_ear_open, 0.01), 1.0)
    ear_col = (0, 80, 255) if result.ear < cal.ear_threshold else (0, 200, 80)
    cv2.rectangle(frame, (8, 10), (208, 22), (60, 60, 60), -1)
    cv2.rectangle(frame, (8, 10), (8 + int(ear_pct * 200), 22), ear_col, -1)
    thr_x = 8 + int(np.clip(cal.ear_threshold / max(cal.mean_ear_open, 0.01), 0, 1) * 200)
    cv2.line(frame, (thr_x, 8), (thr_x, 24), (0, 200, 255), 2)

    # ── Pitch bar ─────────────────────────────────────────────────
    pitch_norm = np.clip((result.pitch_ratio - 0.5) / 0.7, 0, 1)
    p_col = (0, 80, 255) if result.pitch_ratio < config.PITCH_THRESHOLD else (0, 200, 80)
    cv2.rectangle(frame, (8, 26), (208, 36), (60, 60, 60), -1)
    cv2.rectangle(frame, (8, 26), (8 + int(pitch_norm * 200), 36), p_col, -1)
    thr_pitch_x = 8 + int(((config.PITCH_THRESHOLD - 0.5) / 0.7) * 200)
    cv2.line(frame, (thr_pitch_x, 24), (thr_pitch_x, 38), (0, 165, 255), 2)

    # ── Text rows ─────────────────────────────────────────────────
    nr_col = (0, 80, 255) if result.nose_ratio_delta > config.NOSE_RATIO_DOWN_THRESHOLD else (0, 200, 80)
    rows = [
        (f"EAR:    {result.ear:.3f}  thr={cal.ear_threshold}",                     (255, 110, 110)),
        (f"Pitch:  {result.pitch_ratio:.3f}  thr={config.PITCH_THRESHOLD}",        (110, 180, 255)),
        (f"NoseD:  {result.nose_ratio_delta:+.3f}  thr={config.NOSE_RATIO_DOWN_THRESHOLD}", nr_col),
        (f"MAR:    {result.mar:.3f}",                                               (255, 110, 255)),
        (f"Blinks: {result.blink_count}",                                           (110, 255, 110)),
        (f"Fatigue:{result.fatigue_score:.1f}",                                     (110, 255, 255)),
        (f"Adj X:  {result.adj_x:+d} / {cal.look_side_threshold}",                 (200, 200, 200)),
        (f"Down:   {result.down_frames}f / {config.LOOK_DOWN_FRAMES}",             (180, 180, 180)),
    ]
    for i, (txt, col) in enumerate(rows):
        cv2.putText(frame, txt, (8, 50 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)


def draw_no_face(frame):
    """Banner shown when no face is detected for several frames."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 60), -1)
    cv2.putText(frame, "NO FACE DETECTED — check lighting / distance",
        (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)


def draw_ready_splash(frame, calibration, source):
    """2-second splash shown after calibration completes."""
    h, w = frame.shape[:2]
    roi  = frame[0:115, :]
    frame[0:115, :] = cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0)
    cv2.putText(frame, f"Ready  (calibrated from {source})",
        (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 100), 2)
    cv2.putText(frame,
        f"EAR={calibration.ear_threshold}  "
        f"Side={calibration.look_side_threshold}px  "
        f"Pitch={config.PITCH_THRESHOLD}",
        (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
    cv2.putText(frame,
        f"NoseRatio baseline={calibration.baseline_nose_ratio:.3f}  "
        f"FaceH={calibration.baseline_face_h:.1f}px",
        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 200, 130), 1)