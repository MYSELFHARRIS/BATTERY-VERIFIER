"""Unit tests for the full pipeline runner."""

from pathlib import Path
import sqlite3
import numpy as np
import pytest

from src.detection.detector import BoxDetection
from src.logic.sop_loader import Step
from src.logic.state_machine import EventType, StepEngine
from src.logic.verification_log import VerificationLog
from src.pipeline.adapter import Slot
from src.pipeline.runner import run


class FakeDetector:
    """Scripted detector returning predefined box detections per frame."""

    def __init__(self, script: list[list[BoxDetection]]) -> None:
        self.script = script
        self.idx = 0

    def detect(self, frame: np.ndarray) -> list[BoxDetection]:
        if self.idx < len(self.script):
            boxes = self.script[self.idx]
            self.idx += 1
            return boxes
        return []


@pytest.fixture
def sop_steps() -> list[Step]:
    """Standard 3-step verification sequence."""
    return [
        Step(id=1, name="Inspect cell 1", required_class="cell_up", slot=1),
        Step(id=2, name="Inspect cell 2", required_class="cell_down", slot=2),
        Step(id=3, name="Inspect cell 3", required_class="cell_up", slot=3),
    ]


@pytest.fixture
def vertical_slots() -> list[Slot]:
    """Vertical third slots for a 300x100 frame."""
    return [
        Slot(id=1, x_min=0.0, x_max=0.3333, y_min=0.0, y_max=1.0),
        Slot(id=2, x_min=0.3333, x_max=0.6667, y_min=0.0, y_max=1.0),
        Slot(id=3, x_min=0.6667, x_max=1.0, y_min=0.0, y_max=1.0),
    ]


def test_correct_run(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """Correct run: 3 STEP_VERIFIED events then COMPLETE, and verify_chain() is True."""
    confirm_frames = 3
    engine = StepEngine(sop_steps, confirm_frames=confirm_frames)

    box_step1 = [BoxDetection("cell_up", 0.90, (40.0, 40.0, 60.0, 60.0))]
    box_step2 = [BoxDetection("cell_down", 0.92, (140.0, 40.0, 160.0, 60.0))]
    box_step3 = [BoxDetection("cell_up", 0.95, (240.0, 40.0, 260.0, 60.0))]

    script = (
        [box_step1] * confirm_frames
        + [box_step2] * confirm_frames
        + [box_step3] * confirm_frames
        + [[]]
    )

    detector = FakeDetector(script)
    frames = [np.zeros((100, 300, 3), dtype=np.uint8) for _ in range(len(script))]
    db_path = tmp_path / "correct.db"

    with VerificationLog(db_path) as log:
        events = run(frames, detector, vertical_slots, engine, log)
        valid, bad_row = log.verify_chain()

    assert [e.type for e in events] == [
        EventType.STEP_VERIFIED,
        EventType.STEP_VERIFIED,
        EventType.STEP_VERIFIED,
        EventType.COMPLETE,
    ]
    assert [e.step_id for e in events] == [1, 2, 3, None]
    assert valid is True
    assert bad_row is None


def test_flipped_run_logs_wrong_orientation(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """Flipped run: cell_up where cell_down is required at slot 2 -> WRONG_ORIENTATION logged."""
    confirm_frames = 3
    engine = StepEngine(sop_steps, confirm_frames=confirm_frames)

    box_step1 = [BoxDetection("cell_up", 0.90, (40.0, 40.0, 60.0, 60.0))]
    # Flipped cell at slot 2 (cell_up instead of cell_down)
    box_wrong2 = [BoxDetection("cell_up", 0.88, (140.0, 40.0, 160.0, 60.0))]

    script = [box_step1] * confirm_frames + [box_wrong2] * confirm_frames
    detector = FakeDetector(script)
    frames = [np.zeros((100, 300, 3), dtype=np.uint8) for _ in range(len(script))]
    db_path = tmp_path / "flipped.db"

    with VerificationLog(db_path) as log:
        events = run(frames, detector, vertical_slots, engine, log)
        valid, bad_row = log.verify_chain()

    assert any(e.type == EventType.WRONG_ORIENTATION and e.step_id == 2 for e in events)
    assert valid is True
    assert bad_row is None

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT step_id, result, confidence FROM log WHERE result = 'WRONG_ORIENTATION'")
        rows = cursor.fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == 2
        assert isinstance(rows[0][2], float)
    finally:
        conn.close()


def test_empty_detections_returns_no_events(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """Empty detections for all frames -> no events returned."""
    engine = StepEngine(sop_steps, confirm_frames=3)
    frames_count = 10
    detector = FakeDetector([[] for _ in range(frames_count)])
    frames = [np.zeros((100, 300, 3), dtype=np.uint8) for _ in range(frames_count)]
    db_path = tmp_path / "empty.db"

    with VerificationLog(db_path) as log:
        events = run(frames, detector, vertical_slots, engine, log)
        valid, bad_row = log.verify_chain()

    assert events == []
    assert valid is True
    assert bad_row is None

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM log")
        assert cursor.fetchone()[0] == 0
    finally:
        conn.close()


def test_log_only_contains_allowed_event_rows(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """The log only contains STEP_VERIFIED, WRONG_ORIENTATION, and SKIPPED_STEP rows."""
    confirm_frames = 2
    engine = StepEngine(sop_steps, confirm_frames=confirm_frames)

    box_step1 = [BoxDetection("cell_up", 0.90, (40.0, 40.0, 60.0, 60.0))]
    box_wrong2 = [BoxDetection("cell_up", 0.88, (140.0, 40.0, 160.0, 60.0))]
    box_skip2 = [BoxDetection("cell_up", 0.91, (240.0, 40.0, 260.0, 60.0))]
    box_correct2 = [BoxDetection("cell_down", 0.93, (140.0, 40.0, 160.0, 60.0))]
    box_correct3 = [BoxDetection("cell_up", 0.96, (240.0, 40.0, 260.0, 60.0))]

    script = (
        [box_step1] * confirm_frames       # STEP_VERIFIED (1)
        + [box_wrong2] * confirm_frames    # WRONG_ORIENTATION (2)
        + [box_skip2] * confirm_frames     # SKIPPED_STEP (2)
        + [box_correct2] * confirm_frames  # STEP_VERIFIED (2)
        + [box_correct3] * confirm_frames  # STEP_VERIFIED (3)
        + [[]]                             # COMPLETE
    )

    detector = FakeDetector(script)
    frames = [np.zeros((100, 300, 3), dtype=np.uint8) for _ in range(len(script))]
    db_path = tmp_path / "all_events.db"

    with VerificationLog(db_path) as log:
        events = run(frames, detector, vertical_slots, engine, log)
        valid, bad_row = log.verify_chain()

    assert valid is True
    assert bad_row is None
    assert any(e.type == EventType.COMPLETE for e in events)

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT result FROM log")
        results = {r[0] for r in cursor.fetchall()}
        assert results.issubset({"VERIFIED", "WRONG_ORIENTATION", "SKIPPED_STEP"})
        assert "IN_PROGRESS" not in results
        assert "COMPLETE" not in results
    finally:
        conn.close()


def test_dedupe_twelve_frames_wrong_orientation(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """12 frames of wrong-orientation produce 1 returned event and 1 log row with dedupe=True, and 12 with dedupe=False."""
    # confirm_frames=1 so every frame triggers WRONG_ORIENTATION for step 1
    box_wrong = [BoxDetection("cell_down", 0.90, (40.0, 40.0, 60.0, 60.0))]
    script = [box_wrong] * 12
    frames = [np.zeros((100, 300, 3), dtype=np.uint8) for _ in range(12)]

    # 1. dedupe=True
    engine_dedupe = StepEngine(sop_steps, confirm_frames=1)
    detector_dedupe = FakeDetector(script)
    db_dedupe = tmp_path / "dedupe_true.db"

    with VerificationLog(db_dedupe) as log:
        events_dedupe = run(
            frames, detector_dedupe, vertical_slots, engine_dedupe, log, dedupe=True
        )
        valid_dedupe, bad_row_dedupe = log.verify_chain()

    assert len(events_dedupe) == 1
    assert events_dedupe[0].type == EventType.WRONG_ORIENTATION
    assert valid_dedupe is True
    assert bad_row_dedupe is None

    conn = sqlite3.connect(str(db_dedupe))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM log")
        assert cursor.fetchone()[0] == 1
    finally:
        conn.close()

    # 2. dedupe=False
    engine_no_dedupe = StepEngine(sop_steps, confirm_frames=1)
    detector_no_dedupe = FakeDetector(script)
    db_no_dedupe = tmp_path / "dedupe_false.db"

    with VerificationLog(db_no_dedupe) as log:
        events_no_dedupe = run(
            frames,
            detector_no_dedupe,
            vertical_slots,
            engine_no_dedupe,
            log,
            dedupe=False,
        )
        valid_no_dedupe, bad_row_no_dedupe = log.verify_chain()

    assert len(events_no_dedupe) == 12
    assert all(e.type == EventType.WRONG_ORIENTATION for e in events_no_dedupe)
    assert valid_no_dedupe is True
    assert bad_row_no_dedupe is None

    conn = sqlite3.connect(str(db_no_dedupe))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM log")
        assert cursor.fetchone()[0] == 12
    finally:
        conn.close()


def test_verified_step_followed_by_wrong_orientation_produces_two_rows(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """A verified step followed by a wrong orientation at the next step produce 2 log rows."""
    confirm_frames = 2
    engine = StepEngine(sop_steps, confirm_frames=confirm_frames)

    box_step1 = [BoxDetection("cell_up", 0.90, (40.0, 40.0, 60.0, 60.0))]
    box_wrong2 = [BoxDetection("cell_up", 0.88, (140.0, 40.0, 160.0, 60.0))]

    # 2 frames step 1 (verified), then 4 frames of wrong orientation at step 2 (logged once)
    script = [box_step1] * confirm_frames + [box_wrong2] * (confirm_frames + 2)
    detector = FakeDetector(script)
    frames = [np.zeros((100, 300, 3), dtype=np.uint8) for _ in range(len(script))]
    db_path = tmp_path / "two_rows.db"

    with VerificationLog(db_path) as log:
        events = run(frames, detector, vertical_slots, engine, log, dedupe=True)
        valid, bad_row = log.verify_chain()

    assert len(events) == 2
    assert events[0].type == EventType.STEP_VERIFIED and events[0].step_id == 1
    assert events[1].type == EventType.WRONG_ORIENTATION and events[1].step_id == 2
    assert valid is True
    assert bad_row is None

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT step_id, result FROM log")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, "VERIFIED")
        assert rows[1] == (2, "WRONG_ORIENTATION")
    finally:
        conn.close()
