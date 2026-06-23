# ─────────────────────────────────────────────────────────────────
# landmarks.py
# Landmark extraction and all geometric metric calculations.
# No detection logic here — pure geometry only.
# ─────────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
from math import hypot
from config import (
    LEFT_EYE, RIGHT_EYE, UPPER_LIP, LOWER_LIP, LEFT_MOUTH, RIGHT_MOUTH,
    NOSE_TIP, LEFT_FACE, RIGHT_FACE, FOREHEAD, CHIN
)

# ── MediaPipe setup ───────────────────────────────────────────────
_mp_face_mesh = mp.solutions.face_mesh
face_mesh = _mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_landmarks(frame, w, h):
    """
    Run MediaPipe on a BGR frame.
    Returns list of (x, y) pixel tuples for all 468 landmarks,
    or None if no face detected.
    """
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return [
        (int(lm.x * w), int(lm.y * h))
        for lm in results.multi_face_landmarks[0].landmark
    ]


def euclidean(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(eye_points, lms):
    """
    Eye Aspect Ratio.
    eye_points: 6 landmark indices [corner, upper1, upper2, corner, lower1, lower2]
    Returns float — lower = more closed.
    """
    p = [lms[i] for i in eye_points]
    vertical1  = euclidean(p[1], p[4])
    vertical2  = euclidean(p[2], p[5])
    horizontal = euclidean(p[0], p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_avg_ear(lms):
    """Bilateral average EAR across both eyes."""
    left  = calculate_ear(LEFT_EYE,  lms)
    right = calculate_ear(RIGHT_EYE, lms)
    return (left + right) / 2.0


def calculate_mar(lms):
    """
    Mouth Aspect Ratio.
    Returns float — higher = mouth more open (yawning).
    """
    height = euclidean(lms[UPPER_LIP], lms[LOWER_LIP])
    width  = euclidean(lms[LEFT_MOUTH], lms[RIGHT_MOUTH])
    return height / max(width, 1e-6)


def get_face_height(lms):
    """
    Pixel distance from forehead (lm 10) to chin (lm 152).
    Shrinks when head tilts forward (looking down) or back (looking up).
    Used as pitch signal — compare against baseline.
    """
    return euclidean(lms[FOREHEAD], lms[CHIN])


def get_nose_ratio(lms):
    """
    Nose vertical position as a ratio within the forehead→chin span.
    ~0.55 at neutral.
    Increases when looking DOWN (nose moves toward chin).
    Decreases when looking UP (nose moves toward forehead).
    Used alongside face height to disambiguate up vs down tilt.
    """
    forehead_y = lms[FOREHEAD][1]
    chin_y     = lms[CHIN][1]
    nose_y     = lms[NOSE_TIP][1]
    span = chin_y - forehead_y
    if span < 1:
        return 0.5
    return (nose_y - forehead_y) / span


def get_lateral_offset(lms):
    """
    Horizontal offset of nose from face center.
    Positive = nose right of center, negative = left.
    Used for left/right distraction detection.
    """
    nose = lms[NOSE_TIP]
    fc_x = (lms[LEFT_FACE][0] + lms[RIGHT_FACE][0]) // 2
    return nose[0] - fc_x


def get_all_metrics(lms):
    """
    Convenience function — returns all metrics in one call.
    Returns a dict so callers can pick what they need.
    """
    return {
        "ear":          calculate_avg_ear(lms),
        "mar":          calculate_mar(lms),
        "face_height":  get_face_height(lms),
        "nose_ratio":   get_nose_ratio(lms),
        "lateral_offset": get_lateral_offset(lms),
    }
