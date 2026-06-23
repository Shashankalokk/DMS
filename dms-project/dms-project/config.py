# ─────────────────────────────────────────────────────────────────
# config.py
# All tunable constants in one place.
# Change thresholds here — nothing else needs to be touched.
# ─────────────────────────────────────────────────────────────────

# ── MediaPipe landmark indices ────────────────────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP   = 13
LOWER_LIP   = 14
LEFT_MOUTH  = 78
RIGHT_MOUTH = 308

NOSE_TIP   = 1
LEFT_FACE  = 234
RIGHT_FACE = 454
FOREHEAD   = 10
CHIN       = 152

# ── Detection thresholds (overwritten after calibration) ──────────
EAR_THRESHOLD             = 0.25   # eye aspect ratio — below = eye closed
MAR_THRESHOLD             = 0.60   # mouth aspect ratio — above = yawning
PITCH_THRESHOLD           = 0.82   # face_height/baseline ratio — below = head down
LOOK_SIDE_THRESHOLD       = 35     # lateral pixel offset — above = looking left/right
NOSE_RATIO_DOWN_THRESHOLD = 0.05   # extra confirmation for head-down vs head-up

# ── Frame counters — how many consecutive frames before alert ─────
EYE_CLOSED_FRAMES  = 15   # ~0.75s at 20fps
YAWN_FRAMES        = 15
LOOK_DOWN_FRAMES   = 20   # ~1.0s
DISTRACTION_FRAMES = 30   # ~1.5s

# ── Scoring ───────────────────────────────────────────────────────
FATIGUE_DECAY_RATE = 0.05   # points lost per frame when no event active
FATIGUE_ALERT_THRESHOLD = 5

# ── Alert cooldowns (seconds between repeated alerts per type) ────
ALERT_COOLDOWN = {
    "drowsy":  3,
    "yawn":    4,
    "side":    2,
    "down":    2,
    "fatigue": 5,
}

# ── Calibration ───────────────────────────────────────────────────
# How far into the top EAR distribution = "open eye" reference
EAR_OPEN_PERCENTILE    = 0.70   # top 30% of frames
# How far into the bottom EAR distribution = "closed eye" reference
EAR_CLOSED_PERCENTILE  = 0.05   # bottom 5% of frames
# Threshold bias: 0.4 = 40% of the way from closed toward open
EAR_THRESHOLD_BIAS     = 0.40
# Min ratio between closed and open EAR to confirm real blinks were seen
EAR_BLINK_RATIO        = 0.85
# Fallback formula if no blinks detected in calibration video
EAR_FALLBACK_RATIO     = 0.65
# Clamp range for final EAR threshold
EAR_THRESHOLD_MIN      = 0.15
EAR_THRESHOLD_MAX      = 0.42
# Side threshold: N * std dev of lateral movement during calibration
SIDE_THRESHOLD_STD_MULTIPLIER = 3
SIDE_THRESHOLD_MIN            = 25

# ── YOLO phone detection ──────────────────────────────────────────
# Path to your trained YOLOv8 weights file
# Updated to new model (train-10) — only detects "perfect" vs "dist_mob"
YOLO_MODEL_PATH = r"C:\Users\gamem\runs\detect\train-10\weights\best.pt"

# Run YOLO every N frames — keeps live feed smooth.
# 5 = ~4fps checks on a 20fps feed. Increase if CPU is struggling.
YOLO_SAMPLE_INTERVAL = 5

# Dual-confidence threshold:
#   YOLO_CONF_BASE  — minimum conf passed to the model (catches more detections)
#   YOLO_CONF_PHONE — minimum conf to actually count as a phone hit
# New model's standalone test script used a single CONF_THRESHOLD=0.5 —
# base kept slightly lower so the hit-streak/grace-period logic still
# has signal to work with, phone threshold matches the tested value.
YOLO_CONF_BASE  = 0.5   # passed to model(conf=...)
YOLO_CONF_PHONE = 0.5    # minimum to count as a real phone detection

# Safe class keyword — anything whose class name contains this string
# is ignored. Everything else is treated as a distraction.
# New model only has two classes: "perfect" and "dist_mob".
YOLO_SAFE_CLASS_KEYWORD = "perfect"

# Consecutive YOLO-positive sample-frames required before alerting.
# With SAMPLE_INTERVAL=5: threshold=3 means ~15 frames (~0.75s) sustained.
# Lower than before because dual-conf already filters weak hits.
YOLO_PHONE_HIT_THRESHOLD = 3

# Grace period (seconds): keep phone alert alive this long after detection
# drops, to handle motion blur / hand occlusion without flickering.
PHONE_GRACE_PERIOD = 0.8

# Duration (seconds) phone must be continuously detected before alerting.
# Prevents single-frame false positives from triggering the warning.
PHONE_ALERT_DURATION = 2.0

# Alert cooldown for phone detection (seconds)
ALERT_COOLDOWN_PHONE = 5

# ── UI ────────────────────────────────────────────────────────────
WINDOW = "Driver Monitor"