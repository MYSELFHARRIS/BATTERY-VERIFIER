"""Pipeline runner executing vision detection, slot adaptation, state engine, and logging."""

import time
from typing import Any, Callable, Iterable
import numpy as np

from src.logic.state_machine import Event, EventType, StepEngine
from src.logic.verification_log import VerificationLog
from src.pipeline.adapter import Slot, to_detections


def run(
    frames: Iterable[np.ndarray],
    detector: Any,
    slots: list[Slot],
    engine: StepEngine,
    log: VerificationLog,
    sop_version: int = 1,
    on_event: Callable[[int, Event], None] | None = None,
    on_frame_time: Callable[[int, float], None] | None = None,
) -> list[Event]:
    """Run verification pipeline over frames.

    Args:
        frames: Iterable of BGR numpy image arrays.
        detector: Object with detect(frame) -> list[BoxDetection].
        slots: List of Slot definitions.
        engine: StepEngine state machine instance.
        log: VerificationLog instance.
        sop_version: SOP version number (default: 1).
        on_event: Optional callback invoked for each non-IN_PROGRESS event: (frame_idx, event).
        on_frame_time: Optional callback invoked with (frame_idx, elapsed_seconds).

    Returns:
        List of all non-IN_PROGRESS events encountered up to COMPLETE.
    """
    collected_events: list[Event] = []

    for frame_idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        boxes = detector.detect(frame)
        frame_h, frame_w = frame.shape[:2]
        dets = to_detections(boxes, frame_w, frame_h, slots)
        event = engine.update(dets)
        elapsed_sec = time.perf_counter() - t0

        if on_frame_time is not None:
            on_frame_time(frame_idx, elapsed_sec)

        if event.type in (
            EventType.STEP_VERIFIED,
            EventType.WRONG_ORIENTATION,
            EventType.SKIPPED_STEP,
        ):
            conf: float = 0.0
            target_step = next(
                (s for s in engine.steps if s.id == event.step_id), None
            )
            if target_step is not None:
                slot_dets = [d for d in dets if d.slot == target_step.slot]
                if slot_dets:
                    conf = float(slot_dets[0].confidence)
                elif event.type == EventType.SKIPPED_STEP:
                    step_idx = next(
                        (
                            i
                            for i, s in enumerate(engine.steps)
                            if s.id == event.step_id
                        ),
                        -1,
                    )
                    later_slots = {s.slot for s in engine.steps[step_idx + 1 :]}
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
            log.append(
                step_id=event.step_id,
                result=result_str,
                confidence=float(conf),
                sop_version=sop_version,
            )

        if event.type != EventType.IN_PROGRESS:
            collected_events.append(event)
            if on_event is not None:
                on_event(frame_idx, event)

        if event.type == EventType.COMPLETE:
            break

    return collected_events
