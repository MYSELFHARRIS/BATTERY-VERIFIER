"""Unit tests for overlay drawing and visual feedback rendering."""

import numpy as np
import pytest

from src.detection.detector import BoxDetection
from src.logic.state_machine import EventType
from src.pipeline.adapter import Slot
from src.pipeline.overlay import draw_overlay


@pytest.fixture
def sample_slots() -> list[Slot]:
    """Sample vertical third slots for testing."""
    return [
        Slot(id=1, x_min=0.0, x_max=0.3333, y_min=0.0, y_max=1.0),
        Slot(id=2, x_min=0.3333, x_max=0.6667, y_min=0.0, y_max=1.0),
        Slot(id=3, x_min=0.6667, x_max=1.0, y_min=0.0, y_max=1.0),
    ]


def test_draw_overlay_returns_same_shape_and_dtype(
    sample_slots: list[Slot],
) -> None:
    """draw_overlay returns the same shape and dtype as the input."""
    frame = np.zeros((864, 1920, 3), dtype=np.uint8)
    boxes = [BoxDetection("cell_up", 0.95, (100.0, 100.0, 200.0, 300.0))]
    out = draw_overlay(
        frame_bgr=frame,
        boxes=boxes,
        slots=sample_slots,
        step_text="Step 1/3: Inspect cell 1",
        event_text="Step 1 in progress",
        event_type=EventType.IN_PROGRESS,
        frame_index=0,
    )
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_draw_overlay_does_not_modify_input_array(
    sample_slots: list[Slot],
) -> None:
    """The input array is not modified."""
    frame = np.ones((864, 1920, 3), dtype=np.uint8) * 128
    original = frame.copy()
    boxes = [BoxDetection("cell_up", 0.95, (100.0, 100.0, 200.0, 300.0))]
    out = draw_overlay(
        frame_bgr=frame,
        boxes=boxes,
        slots=sample_slots,
        step_text="Step 1/3: Inspect cell 1",
        event_text="Step 1 in progress",
        event_type=EventType.IN_PROGRESS,
        frame_index=1,
    )
    assert np.array_equal(frame, original)
    assert not np.array_equal(out, frame)
    assert out is not frame


def test_draw_overlay_empty_boxes_and_none_event_type(
    sample_slots: list[Slot],
) -> None:
    """It works with an empty boxes list and with event_type None."""
    frame = np.zeros((864, 1920, 3), dtype=np.uint8)
    out = draw_overlay(
        frame_bgr=frame,
        boxes=[],
        slots=sample_slots,
        step_text="Step 1/3: Inspect cell 1",
        event_text="Ready",
        event_type=None,
        frame_index=0,
    )
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.STEP_VERIFIED,
        EventType.WRONG_ORIENTATION,
        EventType.SKIPPED_STEP,
    ],
)
def test_draw_overlay_event_types(
    event_type: EventType, sample_slots: list[Slot]
) -> None:
    """It works with each of STEP_VERIFIED, WRONG_ORIENTATION and SKIPPED_STEP."""
    frame = np.zeros((864, 1920, 3), dtype=np.uint8)
    boxes = [BoxDetection("cell_up", 0.90, (100.0, 100.0, 200.0, 300.0))]
    out = draw_overlay(
        frame_bgr=frame,
        boxes=boxes,
        slots=sample_slots,
        step_text="Step 1/3: Inspect cell 1",
        event_text=f"Event: {event_type.value}",
        event_type=event_type,
        frame_index=15,
    )
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
