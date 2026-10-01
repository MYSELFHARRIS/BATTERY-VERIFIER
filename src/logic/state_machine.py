"""State machine implementation for battery verification process."""

from dataclasses import dataclass
from enum import Enum
from src.logic.sop_loader import Step


@dataclass(frozen=True)
class Detection:
    cls: str
    slot: int
    confidence: float


class EventType(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    STEP_VERIFIED = "STEP_VERIFIED"
    WRONG_ORIENTATION = "WRONG_ORIENTATION"
    SKIPPED_STEP = "SKIPPED_STEP"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class Event:
    type: EventType
    step_id: int | None
    message: str
    confidence: float | None


class StepEngine:
    """State machine that tracks and verifies sequential SOP steps from frame detections."""

    def __init__(
        self,
        steps: list[Step],
        confirm_frames: int = 5,
        min_confidence: float = 0.5,
    ) -> None:
        if not steps:
            raise ValueError("Steps list cannot be empty.")
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1.")

        self.steps = list(steps)
        self.confirm_frames = confirm_frames
        self.min_confidence = min_confidence

        self._current_step_index = 0
        self._streak_kind: str | None = None
        self._streak_count = 0
        self._good_confidences: list[float] = []

    @property
    def current_step(self) -> Step | None:
        """Return the current unverified step, or None if complete."""
        if self._current_step_index < len(self.steps):
            return self.steps[self._current_step_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Return True if all steps have been verified."""
        return self._current_step_index >= len(self.steps)

    def update(self, detections: list[Detection]) -> Event:
        """Process one frame of detections and update state."""
        if self.is_complete:
            return Event(
                type=EventType.COMPLETE,
                step_id=None,
                message="All steps complete",
                confidence=None,
            )

        step = self.current_step
        assert step is not None

        # 1. Drop detections with confidence < min_confidence
        valid = [d for d in detections if d.confidence >= self.min_confidence]

        # 2. In current step's slot, keep the highest-confidence one
        slot_detections = [d for d in valid if d.slot == step.slot]
        frame_kind: str
        frame_conf: float | None = None

        if slot_detections:
            best_det = max(slot_detections, key=lambda d: d.confidence)
            frame_conf = float(best_det.confidence)
            if best_det.cls == step.required_class:
                frame_kind = "good"
            else:
                frame_kind = "wrong"
        else:
            # Check later steps' slots
            later_steps = self.steps[self._current_step_index + 1 :]
            is_skip = any(
                d.slot == later_step.slot and d.cls == later_step.required_class
                for later_step in later_steps
                for d in valid
            )
            if is_skip:
                frame_kind = "skip"
            else:
                frame_kind = "empty"

        # 4. Streak counter management
        if frame_kind == "empty":
            self._streak_kind = "empty"
            self._streak_count = 0
            self._good_confidences.clear()
        elif frame_kind != self._streak_kind:
            self._streak_kind = frame_kind
            self._streak_count = 1
            self._good_confidences = [frame_conf] if frame_kind == "good" and frame_conf is not None else []
        else:
            self._streak_count += 1
            if frame_kind == "good" and frame_conf is not None:
                self._good_confidences.append(frame_conf)

        # 5. Check if streak reached confirm_frames
        if self._streak_kind == "good" and self._streak_count >= self.confirm_frames:
            verified_step = step
            mean_conf = float(sum(self._good_confidences) / len(self._good_confidences))
            self._current_step_index += 1
            self._streak_kind = None
            self._streak_count = 0
            self._good_confidences.clear()
            return Event(
                type=EventType.STEP_VERIFIED,
                step_id=verified_step.id,
                message=f"Step {verified_step.id} verified: {verified_step.name}",
                confidence=mean_conf,
            )

        if self._streak_kind == "wrong" and self._streak_count >= self.confirm_frames:
            return Event(
                type=EventType.WRONG_ORIENTATION,
                step_id=step.id,
                message=f"Wrong orientation detected for step {step.id}",
                confidence=None,
            )

        if self._streak_kind == "skip" and self._streak_count >= self.confirm_frames:
            return Event(
                type=EventType.SKIPPED_STEP,
                step_id=step.id,
                message=f"Skipped step detected while on step {step.id}",
                confidence=None,
            )

        # 6. Otherwise return IN_PROGRESS
        return Event(
            type=EventType.IN_PROGRESS,
            step_id=step.id,
            message=f"Step {step.id} in progress",
            confidence=None,
        )
