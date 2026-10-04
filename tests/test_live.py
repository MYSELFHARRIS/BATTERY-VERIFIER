"""Unit tests for the LiveSession pipeline class."""

from pathlib import Path
import sqlite3
import numpy as np
import pytest

from src.detection.detector import BoxDetection
from src.logic.sop_loader import Step
from src.logic.state_machine import EventType, StepEngine
from src.logic.verification_log import VerificationLog
from src.pipeline.adapter import Slot
from src.pipeline.live import LiveSession


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


def test_live_session_process_and_dedupe(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """LiveSession logs incidents with deduplication and updates status."""
    confirm_frames = 2
    engine = StepEngine(sop_steps, confirm_frames=confirm_frames)
    db_path = tmp_path / "live_test.db"

    # Step 1 correct (2 frames), then 4 frames of wrong orientation at step 2
    box_good1 = [BoxDetection("cell_up", 0.90, (40.0, 40.0, 60.0, 60.0))]
    box_wrong2 = [BoxDetection("cell_up", 0.88, (140.0, 40.0, 160.0, 60.0))]
    script = [box_good1] * confirm_frames + [box_wrong2] * 4

    detector = FakeDetector(script)
    dummy_frame = np.zeros((100, 300, 3), dtype=np.uint8)

    with VerificationLog(db_path) as log:
        session = LiveSession(
            detector=detector,
            slots=vertical_slots,
            engine=engine,
            log=log,
            sop_version=1,
        )

        for _ in range(len(script)):
            boxes, event = session.process(dummy_frame)
            assert isinstance(boxes, list)

        valid, bad_row = log.verify_chain()
        assert valid is True
        assert bad_row is None

        # Verify last event tracking
        assert session.last_event_type == EventType.WRONG_ORIENTATION
        assert "Wrong orientation" in session.last_event_text

    # Log should contain exactly 2 rows: 1 for STEP_VERIFIED (step 1), 1 for WRONG_ORIENTATION (step 2)
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


def test_live_session_reset(
    tmp_path: Path, sop_steps: list[Step], vertical_slots: list[Slot]
) -> None:
    """LiveSession.reset restarts state engine and clears dedupe memory without creating new log."""
    confirm_frames = 1
    engine = StepEngine(sop_steps, confirm_frames=confirm_frames)
    db_path = tmp_path / "live_reset.db"

    box_good1 = [BoxDetection("cell_up", 0.90, (40.0, 40.0, 60.0, 60.0))]
    script = [box_good1, box_good1]
    detector = FakeDetector(script)
    dummy_frame = np.zeros((100, 300, 3), dtype=np.uint8)

    with VerificationLog(db_path) as log:
        session = LiveSession(
            detector=detector,
            slots=vertical_slots,
            engine=engine,
            log=log,
            sop_version=1,
        )

        # Process frame 1 -> STEP_VERIFIED (step 1)
        _, ev1 = session.process(dummy_frame)
        assert ev1.type == EventType.STEP_VERIFIED
        assert ev1.step_id == 1
        assert session.engine.current_step == sop_steps[1]

        # Reset session
        session.reset()
        assert session.engine.current_step == sop_steps[0]
        assert session._last_logged_key is None
        assert session.last_event_text == "Ready"
        assert session.last_event_type is None

        # Process frame 2 with same detection -> STEP_VERIFIED (step 1) again
        _, ev2 = session.process(dummy_frame)
        assert ev2.type == EventType.STEP_VERIFIED
        assert ev2.step_id == 1

        valid, bad_row = log.verify_chain()
        assert valid is True
        assert bad_row is None

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM log WHERE result = 'VERIFIED' AND step_id = 1")
        assert cursor.fetchone()[0] == 2
    finally:
        conn.close()
