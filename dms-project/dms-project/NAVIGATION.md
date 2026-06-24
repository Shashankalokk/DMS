# Navigation Guide

A map of every file, what it's responsible for, and how data flows between
them. Read this if you're picking the project back up after a break, or
onboarding someone else onto it.

---

## Data flow, at a glance

```
   webcam / video frame
          │
          ▼
   landmarks.py  ──────────►  468 face landmarks  ──► get_all_metrics()
          │                                                  │
          │ (in parallel)                                    ▼
          ▼                                          { ear, mar, face_height,
   yolo_detection.py                                   nose_ratio, lateral_offset }
   (sampled every N frames)                                   │
          │                                                    ▼
          ▼                                          detection.py (Detector)
      YoloResult                                       .process_frame(metrics)
   (phone_detected,                                            │
    confidence, boxes)                                         ▼
          │                                            DetectionResult
          └─────────────►  detection.py  ◄────────────  (warn_drowsy, warn_yawn,
              .process_phone(yolo_result, result)         warn_phone, etc.)
                                                                 │
                                                                 ▼
                                                          display.py
                                                   (draws boxes, warnings, HUD)
                                                                 │
                                                                 ▼
                                                          on-screen frame
```

Everything is orchestrated by **`main.py`**, which calls into the other
modules in order every frame.

---

## File-by-file

### `main.py` — entry point

**Startup flow** (before any detection loop runs):
1. `prompt_calibration_choice()` shows a terminal menu — Live camera vs.
   Video file. Runs **before** any OpenCV window is created, so nothing
   visual appears until the user has answered.
2. Based on the choice:
   - **Live** → `do_live_calibration()` opens the camera, wraps it in
     `RecordingCapture` (see below), runs the normal 3-phase interactive
     calibration from `calibration.py`, and saves the session to
     `calib_videos/calib_<timestamp>.mp4`.
   - **Video** → `pick_video_from_directory()` lists every video file in
     `calib_videos/` and lets the user pick one by number, then
     `calibrate_from_video()` runs silently against it. Falls back to
     live calibration if no videos exist or calibration fails.
3. Only after calibration succeeds does `cv2.namedWindow(...)` actually
   get used for the ready-splash and the main detection loop.

**`RecordingCapture`** — a wrapper class around `cv2.VideoCapture`,
needed because `cv2.VideoCapture`'s methods are C++-backed and read-only,
so monkey-patching `.read()` directly raises `AttributeError`. The
wrapper instead implements `.read()` itself and forwards every other
attribute/method call to the real capture object via `__getattr__`, so
`calibration.py` can't tell the difference.

The important subtlety inside `.read()`: `calibrate_from_camera()` in
`calibration.py` flips every frame itself (`cv2.flip(frame, 1)`) **after**
calling `.read()`, for the on-screen mirror effect. `RecordingCapture`
must therefore return the **raw, unflipped** frame to avoid a double-flip
on screen — but it writes a **separately mirrored copy** to the video
file, so that a later video-based calibration session (which does *not*
flip on its own) sees the same orientation a live session would have.
Get this wrong and "looking left" during recording calibrates as
"looking right" during playback.

**Main detection loop**, once calibration is done. Each iteration:
1. Reads a frame from the camera
2. Runs `yolo_detection.py` (only every `YOLO_SAMPLE_INTERVAL` frames)
3. Runs `landmarks.py` to get face landmarks
4. If a face was found, runs `detection.py`'s `process_frame()`
5. Always runs `detection.py`'s `process_phone()` — phone detection doesn't
   depend on a face being present
6. Hands the result to `display.py` to draw everything
7. Shows the frame, checks for ESC to exit

**Touch this file when:** changing the overall loop order, adding a new
top-level feature, changing how calibration is selected, or changing the
startup menu's options.

---

### `config.py` — single source of truth for constants
No logic — just numbers and landmark index lists. Every other file imports
from here. If you're tuning sensitivity, this is almost always the only
file you need to edit.

Sections:
- MediaPipe landmark indices (which of the 468 points matter)
- Detection thresholds (EAR, MAR, pitch, lateral offset)
- Frame counters (how long a condition must persist before alerting)
- Fatigue scoring weights
- Alert cooldowns
- Calibration formula parameters
- YOLO confidence + timing constants

**Touch this file when:** tuning sensitivity, adding a new alert type's
threshold, changing the YOLO model path.

---

### `landmarks.py` — pure geometry
Wraps MediaPipe FaceMesh. Takes a raw frame, returns either `None` (no face)
or a list of 468 `(x, y)` pixel coordinates.

Also contains all the geometric metric calculations that turn raw landmarks
into meaningful numbers:
- `calculate_avg_ear()` — eye openness
- `calculate_mar()` — mouth openness
- `get_face_height()` — forehead-to-chin distance (pitch signal)
- `get_nose_ratio()` — nose position within face (up/down disambiguation)
- `get_lateral_offset()` — left/right head position

`get_all_metrics()` is the convenience function everything else calls — it
bundles all five metrics into one dict.

**Touch this file when:** adding a new geometric signal (e.g. a new face
landmark-based metric), or changing how an existing metric is calculated.
This file has zero knowledge of thresholds or alerts — it only computes
numbers.

---

### `calibration.py` — per-driver baseline setup
Two independent ways to establish a driver's personal baseline, both
producing the same `CalibrationResult`:

- **`calibrate_from_video(path)`** — silently processes every frame of a
  pre-recorded clip. No user interaction. Good for automated/CI use or
  when you already have a clean clip of the driver.

- **`calibrate_from_camera(cap)`** — interactive 3-phase flow:
  1. Open-eye EAR (driver looks at camera, eyes open)
  2. Blink calibration (driver blinks 3 times so the system learns what
     "closed" looks like for them specifically)
  3. Head-pose baseline (driver looks at the road position)

Both paths funnel into the shared `_compute_thresholds()` helper, which
derives the actual numbers (EAR threshold, lateral threshold, baseline
pose values) from the raw samples using percentile/std-dev based formulas
from `config.py`.

**Touch this file when:** changing how calibration phases work, adding a
new calibration phase, or adjusting the math that turns raw samples into
thresholds (though the formula constants themselves live in `config.py`).

---

### `detection.py` — the decision engine
Holds all state across frames (counters, scores, cooldown timestamps) in
the `Detector` class.

- **`process_frame(metrics)`** — the MediaPipe-based checks. Called once
  per frame *only when a face was detected*. Runs drowsiness, yawning,
  lateral distraction, and head-down checks, each with its own consecutive-
  frame counter so a single noisy frame doesn't trigger a false alert.
  Also decays and accumulates the overall fatigue score.

- **`process_phone(yolo_result, detection_result)`** — the YOLO-based
  check. Called **every frame regardless of face detection** — a driver
  looking down to use a phone is exactly the case where MediaPipe might
  lose the face, so this can't depend on `process_frame()` having run.
  Implements:
  - A **grace period** (`PHONE_GRACE_PERIOD`) so a single missed YOLO
    frame doesn't reset an in-progress detection
  - A **duration gate** (`PHONE_ALERT_DURATION`) so the phone must be
    visible continuously for ~2 seconds before the warning actually fires

- **`reset_counters()`** — called by `main.py` when no face is detected,
  so stale counters don't linger and fire incorrectly once the face
  returns.

**Touch this file when:** adding a new alert type, changing how an
existing alert is triggered or debounced, or changing fatigue scoring
logic.

---

### `yolo_detection.py` — phone detection
Wraps a trained YOLOv8 model (`PhoneDetector` class). Designed to be
sampled rather than run every frame for performance:

```python
phone_detector.tick()               # call every frame
if phone_detector.should_run():
    phone_detector.detect(frame)    # only runs every YOLO_SAMPLE_INTERVAL frames
result = phone_detector.last_result # always available, persists between samples
```

Detection logic per call to `detect()`:
1. Run inference at `YOLO_CONF_BASE` (a permissive threshold, to catch
   more candidate detections)
2. Filter out anything matching `YOLO_SAFE_CLASS_KEYWORD` (currently
   `"perfect"`) — that's the "no phone" class
3. Of what's left, only accept boxes at or above `YOLO_CONF_PHONE` (a
   stricter threshold) as a real hit
4. Track a consecutive-hit streak that decays by 2 (not reset to 0) on a
   miss, so one bad frame doesn't erase progress
5. `phone_detected` becomes `True` once the streak crosses
   `YOLO_PHONE_HIT_THRESHOLD`

Has a `if __name__ == "__main__":` debug block — run this file directly
(`python yolo_detection.py`) to test the model standalone against your
webcam with verbose per-frame console output, independent of the rest of
the pipeline.

**Touch this file when:** swapping in a new YOLO model, changing how raw
detections are filtered, or adjusting the hit-streak/decay logic.

---

### `display.py` — pure rendering
Takes a frame and a `DetectionResult` (or `YoloResult`/`CalibrationResult`)
and draws on top of the frame. Contains **no detection logic** — if you're
looking for *why* something triggers, you won't find it here; you'll find
it in `detection.py` or `yolo_detection.py`.

- `draw_warnings()` — stacked banners at the bottom of the screen for each
  active alert
- `draw_phone_boxes()` — YOLO bounding boxes around detected phones
- `draw_hud()` — top-left semi-transparent panel with live metric values
  and progress bars
- `draw_no_face()` — banner shown when face tracking is lost
- `draw_ready_splash()` — brief splash screen shown right after
  calibration finishes, summarizing the calibrated thresholds

**Touch this file when:** changing colors, layout, text, or adding a new
visual element. Never add detection/threshold logic here — keep it in
`detection.py` or `yolo_detection.py` instead.

---

## Typical changes and where they go

| I want to... | Edit this file |
|---|---|
| Make drowsiness detection more/less sensitive | `config.py` (`EAR_THRESHOLD`, `EYE_CLOSED_FRAMES`) |
| Make phone detection trigger faster/slower | `config.py` (`PHONE_ALERT_DURATION`, `YOLO_PHONE_HIT_THRESHOLD`) |
| Swap in a new YOLO model | `config.py` (`YOLO_MODEL_PATH`) — check class names match `YOLO_SAFE_CLASS_KEYWORD` |
| Add a new alert type (e.g. "yawning + drowsy combo") | `detection.py` (new field on `DetectionResult` + logic in `process_frame`), then `display.py` to render it |
| Change calibration phase wording/timing | `calibration.py` |
| Change what's shown on the HUD | `display.py` (`draw_hud`) |
| Add a brand-new geometric signal from face landmarks | `landmarks.py` (new function + add to `get_all_metrics`), then wire it into `detection.py` |
| Change the startup menu wording/options | `main.py` (`prompt_calibration_choice`) |
| Change where calibration videos are saved/read from | `main.py` (`CALIB_VIDEO_DIR` constant near the top) |
