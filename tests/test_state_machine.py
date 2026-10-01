"""Unit tests for the StepEngine state machine."""

import pytest
from src.logic.sop_loader import Step
from src.logic.state_machine import Detection, EventType, StepEngine


@pytest.fixture
def sop_steps() -> list[Step]:
    """3-step SOP: step 1 cell_up slot 1, step 2 cell_down slot 2, step 3 cell_up slot 3."""
    return [
        Step(id=1, name="Inspect cell 1", required_class="cell_up", slot=1),
        Step(id=2, name="Inspect cell 2", required_class="cell_down", slot=2),
        Step(id=3, name="Inspect cell 3", required_class="cell_up", slot=3),
    ]


def test_correct_sequence_verified_in_order_then_complete(sop_steps: list[Step]) -> None:
    """1. Correct sequence: 5 frames each -> verified in order 1,2,3, then COMPLETE."""
    engine = StepEngine(sop_steps, confirm_frames=5)

    # Step 1: 5 frames cell_up in slot 1
    for _ in range(4):
        ev = engine.update([Detection("cell_up", 1, 0.9)])
        assert ev.type == EventType.IN_PROGRESS
    ev1 = engine.update([Detection("cell_up", 1, 0.9)])
    assert ev1.type == EventType.STEP_VERIFIED
    assert ev1.step_id == 1
    assert engine.current_step == sop_steps[1]

    # Step 2: 5 frames cell_down in slot 2
    for _ in range(4):
        ev = engine.update([Detection("cell_down", 2, 0.85)])
        assert ev.type == EventType.IN_PROGRESS
    ev2 = engine.update([Detection("cell_down", 2, 0.85)])
    assert ev2.type == EventType.STEP_VERIFIED
    assert ev2.step_id == 2
    assert engine.current_step == sop_steps[2]

    # Step 3: 5 frames cell_up in slot 3
    for _ in range(4):
        ev = engine.update([Detection("cell_up", 3, 0.95)])
        assert ev.type == EventType.IN_PROGRESS
    ev3 = engine.update([Detection("cell_up", 3, 0.95)])
    assert ev3.type == EventType.STEP_VERIFIED
    assert ev3.step_id == 3
    assert engine.is_complete is True
    assert engine.current_step is None

    # Next frame -> COMPLETE
    ev_complete = engine.update([])
    assert ev_complete.type == EventType.COMPLETE
    assert ev_complete.step_id is None
    assert ev_complete.confidence is None


def test_only_four_good_frames_no_verification(sop_steps: list[Step]) -> None:
    """2. Only 4 good frames -> no STEP_VERIFIED."""
    engine = StepEngine(sop_steps, confirm_frames=5)
    for _ in range(4):
        ev = engine.update([Detection("cell_up", 1, 0.9)])
        assert ev.type == EventType.IN_PROGRESS

    assert engine.current_step == sop_steps[0]
    assert not engine.is_complete


def test_flicker_resets_streak(sop_steps: list[Step]) -> None:
    """3. Flicker: 4 good, 1 empty, 4 good -> not verified; 5 more good -> verified."""
    engine = StepEngine(sop_steps, confirm_frames=5)
    det = Detection("cell_up", 1, 0.9)

    # 4 good
    for _ in range(4):
        assert engine.update([det]).type == EventType.IN_PROGRESS
    # 1 empty
    assert engine.update([]).type == EventType.IN_PROGRESS
    # 4 good -> not verified
    for _ in range(4):
        assert engine.update([det]).type == EventType.IN_PROGRESS
    assert engine.current_step == sop_steps[0]

    # 5 more good -> verified
    events = [engine.update([det]) for _ in range(5)]
    assert any(e.type == EventType.STEP_VERIFIED and e.step_id == 1 for e in events)


def test_flipped_cell_wrong_orientation(sop_steps: list[Step]) -> None:
    """4. Flipped cell in slot 1 for 5 frames -> WRONG_ORIENTATION, step stays 1; then 5 correct frames -> verified."""
    engine = StepEngine(sop_steps, confirm_frames=5)
    wrong_det = Detection("cell_down", 1, 0.9)
    correct_det = Detection("cell_up", 1, 0.9)

    for _ in range(4):
        assert engine.update([wrong_det]).type == EventType.IN_PROGRESS
    ev = engine.update([wrong_det])
    assert ev.type == EventType.WRONG_ORIENTATION
    assert ev.step_id == 1
    assert engine.current_step == sop_steps[0]

    for _ in range(4):
        assert engine.update([correct_det]).type == EventType.IN_PROGRESS
    ev_verified = engine.update([correct_det])
    assert ev_verified.type == EventType.STEP_VERIFIED
    assert ev_verified.step_id == 1
    assert engine.current_step == sop_steps[1]


def test_skipped_step_detected(sop_steps: list[Step]) -> None:
    """5. Cell_down in slot 2 while on step 1, 5 frames -> SKIPPED_STEP, step stays 1."""
    engine = StepEngine(sop_steps, confirm_frames=5)
    skip_det = Detection("cell_down", 2, 0.9)

    for _ in range(4):
        assert engine.update([skip_det]).type == EventType.IN_PROGRESS
    ev = engine.update([skip_det])
    assert ev.type == EventType.SKIPPED_STEP
    assert ev.step_id == 1
    assert engine.current_step == sop_steps[0]


def test_low_confidence_ignored(sop_steps: list[Step]) -> None:
    """6. Detections with confidence 0.3 are ignored (never verify)."""
    engine = StepEngine(sop_steps, confirm_frames=5, min_confidence=0.5)
    low_conf_det = Detection("cell_up", 1, 0.3)

    for _ in range(10):
        ev = engine.update([low_conf_det])
        assert ev.type == EventType.IN_PROGRESS

    assert engine.current_step == sop_steps[0]
    assert not engine.is_complete


def test_verified_confidence_equals_mean_and_float(sop_steps: list[Step]) -> None:
    """7. STEP_VERIFIED confidence equals the mean and isinstance float."""
    engine = StepEngine(sop_steps, confirm_frames=5)
    confidences = [0.6, 0.7, 0.8, 0.9, 1.0]

    for c in confidences[:-1]:
        assert engine.update([Detection("cell_up", 1, c)]).type == EventType.IN_PROGRESS

    ev = engine.update([Detection("cell_up", 1, confidences[-1])])
    assert ev.type == EventType.STEP_VERIFIED
    assert isinstance(ev.confidence, float)
    assert ev.confidence == pytest.approx(0.8)


def test_after_complete_further_updates_return_complete(sop_steps: list[Step]) -> None:
    """8. After COMPLETE, further updates still return COMPLETE."""
    engine = StepEngine(sop_steps, confirm_frames=1)

    # 1 frame each to quickly complete
    engine.update([Detection("cell_up", 1, 0.9)])
    engine.update([Detection("cell_down", 2, 0.9)])
    engine.update([Detection("cell_up", 3, 0.9)])
    assert engine.is_complete is True

    for _ in range(5):
        ev = engine.update([Detection("cell_up", 1, 0.9)])
        assert ev.type == EventType.COMPLETE
        assert ev.step_id is None
        assert ev.confidence is None
