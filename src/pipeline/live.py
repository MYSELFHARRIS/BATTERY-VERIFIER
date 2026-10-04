"""Live camera session for real-time verification and logging."""

from typing import Any
import numpy as np

from src.detection.detector import BoxDetection
from src.logic.state_machine import Event, EventType, StepEngine
from src.logic.verification_log import VerificationLog
from src.pipeline.adapter import Slot, to_detections


class LiveSession:
    """Manages real-time detection, step state machine, and tamper-evident logging."""

    def __init__(
        self,
        detector: Any,
        slots: list[Slot],
        engine: StepEngine,
        log: VerificationLog,
        sop_version: int = 1,
    ) -> None:
        self.detector = detector
        self.slots = slots
        self.engine = engine
        self.log = log
        self.sop_version = sop_version

        # Keep initial engine parameters for clean reset
        self._steps = list(engine.steps)
        self._confirm_frames = engine.confirm_frames
        self._min_confidence = engine.min_confidence

        # Deduplication memory and last event status
        self._last_logged_key: tuple[EventType, int | None] | None = None
        self.last_event_text: str = "Ready"
        self.last_event_type: EventType | None = None

    def process(self, frame_bgr: np.ndarray) -> tuple[list[BoxDetection], Event]:
        """Process a single frame: detect objects, adapt to slots, update state engine, and log events."""
        boxes = self.detector.detect(frame_bgr)
        frame_h, frame_w = frame_bgr.shape[:2]
        dets = to_detections(boxes, frame_w, frame_h, self.slots)
        event = self.engine.update(dets)

        if event.type != EventType.IN_PROGRESS:
            self.last_event_text = event.message
            self.last_event_type = event.type

        # Log loggable events applying the dedupe rule
        if event.type in (
            EventType.STEP_VERIFIED,
            EventType.WRONG_ORIENTATION,
            EventType.SKIPPED_STEP,
        ):
            event_key = (event.type, event.step_id)
            if event_key != self._last_logged_key:
                self._last_logged_key = event_key

                conf: float = 0.0
                target_step = next(
                    (s for s in self.engine.steps if s.id == event.step_id), None
                )
                if target_step is not None:
                    slot_dets = [d for d in dets if d.slot == target_step.slot]
                    if slot_dets:
                        conf = float(slot_dets[0].confidence)
                    elif event.type == EventType.SKIPPED_STEP:
                        step_idx = next(
                            (
                                i
                                for i, s in enumerate(self.engine.steps)
                                if s.id == event.step_id
                            ),
                            -1,
                        )
                        later_slots = {
                            s.slot for s in self.engine.steps[step_idx + 1 :]
                        }
                        skip_dets = [d for d in dets if d.slot in later_slots]
                        if skip_dets:
                            conf = float(skip_dets[0].confidence)

                if conf == 0.0 and event.confidence is not None:
                    conf = float(event.confidence)

                result_str = (
                    "VERIFIED"
                    if event.type == EventType.STEP_VERIFIED
                    else event.type.value
                )
                assert event.step_id is not None
                self.log.append(
                    step_id=event.step_id,
                    result=result_str,
                    confidence=float(conf),
                    sop_version=self.sop_version,
                )

        return boxes, event

    def reset(self) -> None:
        """Reset the state engine to initial step and clear deduplication memory without creating a new log."""
        self.engine = StepEngine(
            steps=self._steps,
            confirm_frames=self._confirm_frames,
            min_confidence=self._min_confidence,
        )
        self._last_logged_key = None
        self.last_event_text = "Ready"
        self.last_event_type = None
