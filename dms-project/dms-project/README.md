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
├── main.py             # Entry point — wires everything together
├── config.py           # All tunable thresholds and constants
├── landmarks.py         # MediaPipe face landmark extraction + geometry
├── calibration.py       # Per-driver threshold calibration (video or live)
├── detection.py         # Frame-by-frame detection state machine
├── yolo_detection.py     # YOLOv8 phone-detection wrapper
├── display.py           # All on-screen drawing (HUD, warnings, boxes)
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

### Option 1 — Calibrate live, every time

```bash
python main.py
```

This opens your webcam and walks you through a 3-step interactive
calibration (open eyes, blink, head pose), then starts live detection.

### Option 2 — Calibrate from a pre-recorded video

```bash
python main.py path/to/calibration_clip.mp4
```

Use a short clip (~20–30s) of the driver sitting normally and looking at
the road. The system silently calibrates from it, then opens the webcam
for live detection — no interaction needed for calibration.

You can also hardcode a path instead of passing it as a CLI argument —
see the `CALIBRATION_VIDEO` variable near the top of `main.py`.

If video calibration fails for any reason (file not found, too few frames
with a face, etc.), it automatically falls back to live calibration.

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
