# ─────────────────────────────────────────────────────────────────
# detection.py
# All detection state and per-frame logic.
# Takes metrics (from landmarks.py) + calibration (from calibration.py)
# and returns a DetectionResult each frame.
# ─────────────────────────────────────────────────────────────────

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import config


@dataclass
class DetectionResult:
    """What happened this frame. display.py reads this."""
    warn_drowsy:   bool = False
    warn_yawn:     bool = False
    warn_side:     bool = False
    warn_side_dir: str  = ""
    warn_down:     bool = False
    warn_fatigue:  bool = False
    warn_phone:    bool = False   # phone detected by YOLO
    phone_conf:    float = 0.0    # highest confidence this result
    blink_count:   int  = 0
    fatigue_score: float = 0.0
    # Raw metrics for HUD display
    ear:           float = 0.0
    mar:           float = 0.0
    pitch_ratio:   float = 1.0
    nose_ratio_delta: float = 0.0
    adj_x:         int   = 0
    closed_frames: int   = 0
    down_frames:   int   = 0


class Detector:
    """
    Stateful detector — call process_frame() each frame.
    Holds all counters and scores internally.
    """

    def __init__(self, calibration):
        """
        calibration: CalibrationResult from calibration.py
        """
        self.cal = calibration

        # Frame counters
        self.closed_frames      = 0
        self.yawn_frames        = 0
        self.down_frames        = 0
        self.distraction_frames = 0

        # Blink state
        self.blink_count  = 0
        self.blink_active = False

        # Fatigue score
        self.fatigue_score = 0.0

        # Alert cooldown timestamps
        self.last_alert_time: Dict[str, float] = {
            k: 0.0 for k in config.ALERT_COOLDOWN
        }
        self._last_phone_alert      = 0.0
        self._phone_start_time: Optional[float] = None   # when continuous detection began
        self._last_phone_seen:  Optional[float] = None   # last time raw detection fired

    def _can_alert(self, key):
        t = time.time()
        if t - self.last_alert_time[key] > config.ALERT_COOLDOWN[key]:
            self.last_alert_time[key] = t
            return True
        return False

    def reset_counters(self):
        """Call when face disappears so stale counts don't linger."""
        self.closed_frames      = 0
        self.yawn_frames        = 0
        self.down_frames        = 0
        self.distraction_frames = 0

    def process_phone(self, yolo_result, detection_result):
        """
        Incorporate a YOLO phone detection result into DetectionResult.
        Call this in main.py every frame, regardless of whether MediaPipe
        found a face — phone detection is independent of face tracking.

        Logic (mirrors colleague's approach):
          - yolo_result.phone_detected is True when hit streak >= threshold
          - Grace period: keep start_time alive for PHONE_GRACE_PERIOD seconds
            after detection drops, to handle motion blur / occlusion frames
          - Duration gate: only alert after phone held for PHONE_ALERT_DURATION
            seconds continuously — prevents single-frame false positives

        yolo_result:      YoloResult from PhoneDetector.last_result
        detection_result: DetectionResult to update in-place
        """
        now = time.time()

        if yolo_result.phone_detected:
            # Active detection — update last-seen timestamp
            self._last_phone_seen = now
            if self._phone_start_time is None:
                self._phone_start_time = now

        else:
            # No detection — check if still within grace period
            if (self._last_phone_seen is not None and
                    now - self._last_phone_seen <= config.PHONE_GRACE_PERIOD):
                # Within grace — preserve start_time, treat as still detected
                pass
            else:
                # Grace expired — reset
                self._phone_start_time = None
                self._last_phone_seen  = None

        # Only alert after phone held continuously for PHONE_ALERT_DURATION
        if self._phone_start_time is not None:
            phone_duration = now - self._phone_start_time
            if phone_duration >= config.PHONE_ALERT_DURATION:
                detection_result.warn_phone = True
                detection_result.phone_conf = yolo_result.confidence
                self.fatigue_score += 2   # phone use contributes to fatigue score
                if now - self._last_phone_alert > config.ALERT_COOLDOWN_PHONE:
                    self._last_phone_alert = now

    def process_frame(self, metrics):
        """
        Run one frame of detection.
        metrics: dict from landmarks.get_all_metrics()
        Returns DetectionResult.
        """
        result = DetectionResult()

        ear    = metrics["ear"]
        mar    = metrics["mar"]
        face_h = metrics["face_height"]
        nr     = metrics["nose_ratio"]
        off_x  = metrics["lateral_offset"]

        cal = self.cal

        # ── Derived values ────────────────────────────────────────
        adj_x       = off_x - cal.baseline_x
        pitch_ratio = face_h / max(cal.baseline_face_h, 1.0)
        nr_delta    = nr - cal.baseline_nose_ratio

        # Looking down requires BOTH face shortening AND nose moving
        # toward chin — prevents looking-UP from triggering the alert
        looking_down = (
            pitch_ratio < config.PITCH_THRESHOLD and
            nr_delta    > config.NOSE_RATIO_DOWN_THRESHOLD
        )

        # ── Fatigue score decay ───────────────────────────────────
        self.fatigue_score = max(0.0, self.fatigue_score - config.FATIGUE_DECAY_RATE)

        # ── Blink detection ───────────────────────────────────────
        if ear < cal.ear_threshold and not self.blink_active:
            self.blink_active = True
            self.blink_count += 1
        elif ear >= cal.ear_threshold:
            self.blink_active = False

        # ── Drowsiness ────────────────────────────────────────────
        self.closed_frames = self.closed_frames + 1 if ear < cal.ear_threshold else 0
        if self.closed_frames >= config.EYE_CLOSED_FRAMES:
            self.fatigue_score += 3
            result.warn_drowsy  = True
            self._can_alert("drowsy")

        # ── Yawning ───────────────────────────────────────────────
        self.yawn_frames = self.yawn_frames + 1 if mar > config.MAR_THRESHOLD else 0
        if self.yawn_frames >= config.YAWN_FRAMES:
            self.fatigue_score += 2
            result.warn_yawn    = True
            self._can_alert("yawn")

        # ── Lateral distraction ───────────────────────────────────
        self.distraction_frames = (
            self.distraction_frames + 1
            if abs(adj_x) > cal.look_side_threshold else 0
        )
        if self.distraction_frames >= config.DISTRACTION_FRAMES:
            self.fatigue_score    += 2
            result.warn_side       = True
            result.warn_side_dir   = "LEFT" if adj_x < 0 else "RIGHT"
            self._can_alert("side")

        # ── Head down ─────────────────────────────────────────────
        self.down_frames = self.down_frames + 1 if looking_down else 0
        if self.down_frames >= config.LOOK_DOWN_FRAMES:
            self.fatigue_score += 2
            result.warn_down    = True
            self._can_alert("down")

        # ── Fatigue alert ─────────────────────────────────────────
        if self.fatigue_score >= config.FATIGUE_ALERT_THRESHOLD:
            result.warn_fatigue = True
            if self._can_alert("fatigue"):
                self.fatigue_score = 0

        # ── Pass through values for HUD ───────────────────────────
        result.blink_count      = self.blink_count
        result.fatigue_score    = self.fatigue_score
        result.ear              = ear
        result.mar              = mar
        result.pitch_ratio      = pitch_ratio
        result.nose_ratio_delta = nr_delta
        result.adj_x            = int(adj_x)
        result.closed_frames    = self.closed_frames
        result.down_frames      = self.down_frames

        return result