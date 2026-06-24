# Driver Monitoring System (DMS)

A real-time driver fatigue and distraction detection system built in Python.
Combines **MediaPipe FaceMesh** (eye closure, yawning, head pose) with a
**YOLOv8** model (phone-use detection) to monitor a driver through a webcam
or dashcam feed and raise on-screen alerts.

---

## What it detects

| Signal | How |
|---|---|
| Drowsiness (eyes closed) | Eye Aspect Ratio (EAR) sustained below threshold |
| Yawning | Mouth Aspect Ratio (MAR) sustained above threshold |
| Looking left / right | Lateral nose offset from face center |
| Looking down | Face-height shrink + nose-ratio shift, combined |
| Phone use | YOLOv8 object detection, sustained over time with a grace period |
| Overall fatigue | A decaying score that accumulates from the above events |

---

## Project structure

See [`NAVIGATION.md`](./NAVIGATION.md) for a full file-by-file breakdown of
what each module does and how data flows between them.

```
.
├── main.py             # Entry point — startup menu + wires everything together
├── config.py           # All tunable thresholds and constants
├── landmarks.py         # MediaPipe face landmark extraction + geometry
├── calibration.py       # Per-driver threshold calibration (video or live)
├── detection.py         # Frame-by-frame detection state machine
├── yolo_detection.py     # YOLOv8 phone-detection wrapper
├── display.py           # All on-screen drawing (HUD, warnings, boxes)
├── calib_videos/         # Auto-created — stores recorded calibration sessions
├── README.md
└── NAVIGATION.md
```

---

## Requirements

- Python 3.9–3.11 recommended
- A webcam, or a pre-recorded driver video for calibration
- Your trained YOLOv8 phone-detection weights (`best.pt`) — **not included
  in this repo** (see [Model weights](#model-weights) below)

### Install dependencies

```bash
pip install opencv-python mediapipe numpy ultralytics
```

---

## Model weights

`best.pt` is a trained model file and is intentionally **not committed** to
this repository (binary weight files don't belong in git, and GitHub blocks
files over 100MB).

1. Place your trained weights file anywhere on disk.
2. Open `config.py` and update:

   ```python
   YOLO_MODEL_PATH = r"C:\path\to\your\best.pt"
   ```

3. If the path doesn't exist when you run the program, phone detection is
   automatically disabled and everything else still runs normally — you'll
   see a warning printed in the terminal.

---

## Running it

Just run:

```bash
python main.py
```

You'll be asked how to calibrate **before any window opens**:

```
==================================================
  DRIVER MONITORING SYSTEM
==================================================

  How would you like to calibrate?

  [1]  Live camera  (records session for future use)
  [2]  Video file   (pick from saved calibration videos)

  Enter 1 or 2:
```

### Option 1 — Live camera

Opens your webcam and walks you through a 3-step interactive calibration
(open eyes, blink, head pose). The entire calibration session is
automatically recorded to `calib_videos/calib_<timestamp>.mp4` as it
happens — no extra steps needed. You can reuse that recording next time
instead of recalibrating live.

### Option 2 — Video file

Lists every video found in `calib_videos/`, with file size and save date,
so you can pick one by number:

```
  Calibration videos in 'calib_videos/':

  [1]  calib_20260615_143022.mp4  (12.4 MB, saved 2026-06-15 14:30)
  [2]  calib_20260614_091500.mp4  (9.1 MB, saved 2026-06-14 09:15)

  Enter number (1-2):
```

The system calibrates silently from the chosen file, then opens the
webcam for live detection.

**Fallback behavior:** if `calib_videos/` is empty, or the chosen video
fails to calibrate (file not found, too few frames with a face, etc.),
it automatically falls back to live camera calibration.

**Orientation note:** recordings are saved already mirrored, matching
exactly what you see on screen during live calibration. This means a
recorded session played back through Option 2 calibrates the same way
the live session did — looking left stays looking left.

### Controls

- **ESC** — quit at any time, including during calibration

---

## Tuning

Almost everything tunable lives in `config.py`, grouped by section:

- **Detection thresholds** — EAR, MAR, pitch, lateral offset
- **Frame counters** — how many consecutive frames before an alert fires
- **YOLO confidence** — `YOLO_CONF_BASE` (passed to the model) and
  `YOLO_CONF_PHONE` (minimum to count as a real hit)
- **Phone timing** — `PHONE_GRACE_PERIOD` (tolerate brief misses) and
  `PHONE_ALERT_DURATION` (how long phone must be visible before alerting)
- **Alert cooldowns** — minimum seconds between repeated alerts per type

Calibration-derived thresholds (EAR, lateral offset, pose baseline) are
computed automatically per driver and don't need manual tuning — only
touch the constants in `config.py` if the defaults clearly don't fit your
camera setup or lighting.

---

## Known limitations

- Single-face only (`max_num_faces=1` in `landmarks.py`)
- YOLO inference runs on a sampled interval (`YOLO_SAMPLE_INTERVAL` in
  `config.py`), not every frame, to keep the live feed smooth on modest
  hardware
- Calibration assumes a roughly front-facing camera angle similar to a
  dashboard-mounted dashcam
