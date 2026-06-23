# ─────────────────────────────────────────────────────────────────
# yolo_detection.py
# Phone detection using your trained YOLOv8 model.
#
# Detection logic mirrors your working standalone script:
#   - Safe class keyword check (not just class name matching)
#   - consecutive_hits counter with decay-by-2 (no flicker on missed frames)
#   - Confidence threshold from config
#
# Runs every YOLO_SAMPLE_INTERVAL frames to keep the feed smooth.
# last_result persists between samples so display stays consistent.
# ─────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List, Tuple
import config


@dataclass
class YoloResult:
    """
    Output of one YOLO inference pass.
    Persists between sample frames so display doesn't flicker.
    """
    phone_detected:  bool        = False   # True when hit streak >= threshold
    raw_detected:    bool        = False   # True when model fires this frame
    confidence:      float       = 0.0
    hit_streak:      int         = 0       # current consecutive hit count
    boxes:           List[Tuple] = field(default_factory=list)


class PhoneDetector:
    """
    Wraps your trained YOLOv8 model.

    Tick/sample pattern:
        phone_detector.tick()               # every frame
        if phone_detector.should_run():
            phone_detector.detect(frame)    # only every N frames
        result = phone_detector.last_result # always up to date
    """

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        print(f"\nLoading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print(f"  Classes     : {self.model.names}")
        print(f"  Conf base   : {config.YOLO_CONF_BASE}")
        print(f"  Conf phone  : {config.YOLO_CONF_PHONE}")
        print(f"  Interval    : every {config.YOLO_SAMPLE_INTERVAL} frames")
        print(f"  Safe keyword: '{config.YOLO_SAFE_CLASS_KEYWORD}'\n")

        self._frame_idx      = 0
        self._consecutive_hits = 0
        self.last_result     = YoloResult()

    def should_run(self) -> bool:
        return self._frame_idx % config.YOLO_SAMPLE_INTERVAL == 0

    def tick(self):
        """Advance internal frame counter. Call once per main loop iteration."""
        self._frame_idx += 1

    def detect(self, frame) -> YoloResult:
        """
        Run inference. Updates last_result and hit streak.
        Only call when should_run() is True.
        """
        # Run inference at low base conf to catch more candidates,
        # then filter by YOLO_CONF_PHONE to suppress weak false positives.
        results    = self.model(frame, conf=config.YOLO_CONF_BASE, verbose=False)
        detections = results[0].boxes

        boxes         = []
        raw_detected  = False
        best_conf     = 0.0

        if detections is not None and len(detections) > 0:
            for box in detections:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                class_name = self.model.names[cls_id]

                # Use same safe-keyword logic as your working standalone script:
                # anything that is NOT the safe class = distraction detection
                is_safe = config.YOLO_SAFE_CLASS_KEYWORD.lower() in class_name.lower()

                # Apply phone-specific confidence gate AFTER safe-keyword check
                if not is_safe and conf >= config.YOLO_CONF_PHONE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2, conf, class_name))
                    raw_detected = True
                    best_conf    = max(best_conf, conf)

        # ── Consecutive hit counter with decay ────────────────────
        # Mirrors your working script exactly:
        #   hit    → increment
        #   miss   → decay by 2 (not zero) so one missed frame doesn't reset progress
        if raw_detected:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = max(0, self._consecutive_hits - 2)

        phone_confirmed = self._consecutive_hits >= config.YOLO_PHONE_HIT_THRESHOLD

        self.last_result = YoloResult(
            phone_detected = phone_confirmed,
            raw_detected   = raw_detected,
            confidence     = best_conf,
            hit_streak     = self._consecutive_hits,
            boxes          = boxes,
        )
        return self.last_result


if __name__ == "__main__":
    # ── Quick debug — run this file directly to test your model ──
    # python yolo_detection.py
    import cv2
    import config

    detector = PhoneDetector(config.YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Debug mode — ALL detections printed regardless of safe keyword.")
    print("Press ESC to quit.\n")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ── Run raw inference with very low conf — catch everything ──
        results = detector.model(frame, conf=0.1, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:
            for box in detections:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                class_name = detector.model.names[cls_id]
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                is_safe = config.YOLO_SAFE_CLASS_KEYWORD.lower() in class_name.lower()
                color   = (100, 100, 100) if is_safe else (0, 0, 255)
                label   = f"{'[SAFE] ' if is_safe else '[DIST] '}{class_name} {conf:.2f}"

                # Print every detection so nothing is hidden
                print(f"  frame={frame_idx:04d}  {label}  box=({x1},{y1},{x2},{y2})")

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Print when nothing detected so we can see the model is running
            if frame_idx % 10 == 0:
                print(f"  frame={frame_idx:04d}  no detections")

        # Show current config values on screen
        h, w = frame.shape[:2]
        cv2.putText(frame, f"conf_thr={config.YOLO_CONF_PHONE}  safe_keyword='{config.YOLO_SAFE_CLASS_KEYWORD}'",
            (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("YOLO Debug", frame)
        frame_idx += 1

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()