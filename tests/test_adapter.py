"""Unit tests for the model boxes to StepEngine detections adapter."""

from pathlib import Path
import numpy as np
import pytest

from src.detection.detector import BoxDetection
from src.logic.state_machine import Detection
from src.pipeline.adapter import Slot, load_slots, to_detections


@pytest.fixture
def vertical_third_slots() -> list[Slot]:
    """Standard 3-slot layout splitting frame horizontally into vertical thirds."""
    return [
        Slot(id=1, x_min=0.0, x_max=0.3333, y_min=0.0, y_max=1.0),
        Slot(id=2, x_min=0.3333, x_max=0.6667, y_min=0.0, y_max=1.0),
        Slot(id=3, x_min=0.6667, x_max=1.0, y_min=0.0, y_max=1.0),
    ]


def test_cell_up_in_left_third_gives_slot_1(vertical_third_slots: list[Slot]) -> None:
    """A cell_up box with its center in the left third gives slot 1."""
    # frame 300x100, box center at (45, 50) -> norm_x = 0.15 (left third)
    box = BoxDetection(
        label="cell_up", confidence=0.88, xyxy=(30.0, 20.0, 60.0, 80.0)
    )
    detections = to_detections([box], frame_w=300, frame_h=100, slots=vertical_third_slots)

    assert len(detections) == 1
    assert detections[0] == Detection(cls="cell_up", slot=1, confidence=0.88)


def test_cell_down_in_middle_third_gives_slot_2(
    vertical_third_slots: list[Slot],
) -> None:
    """A cell_down box with its center in the middle third gives slot 2."""
    # frame 300x100, box center at (150, 50) -> norm_x = 0.5 (middle third)
    box = BoxDetection(
        label="cell_down", confidence=0.92, xyxy=(130.0, 20.0, 170.0, 80.0)
    )
    detections = to_detections([box], frame_w=300, frame_h=100, slots=vertical_third_slots)

    assert len(detections) == 1
    assert detections[0] == Detection(cls="cell_down", slot=2, confidence=0.92)


def test_unknown_label_is_ignored(vertical_third_slots: list[Slot]) -> None:
    """An unknown label is ignored."""
    boxes = [
        BoxDetection(label="battery", confidence=0.95, xyxy=(30.0, 20.0, 60.0, 80.0)),
        BoxDetection(label="cell", confidence=0.90, xyxy=(130.0, 20.0, 170.0, 80.0)),
        BoxDetection(label="other", confidence=0.85, xyxy=(220.0, 20.0, 280.0, 80.0)),
    ]
    detections = to_detections(boxes, frame_w=300, frame_h=100, slots=vertical_third_slots)
    assert len(detections) == 0


def test_box_outside_every_slot_is_skipped(
    vertical_third_slots: list[Slot],
) -> None:
    """A box whose center is outside every slot is skipped."""
    # Box center at (350, 50) in a 300-wide frame -> norm_x = 1.167 (outside)
    outside_box = BoxDetection(
        label="cell_up", confidence=0.99, xyxy=(340.0, 20.0, 360.0, 80.0)
    )
    detections = to_detections(
        [outside_box], frame_w=300, frame_h=100, slots=vertical_third_slots
    )
    assert len(detections) == 0


def test_two_boxes_in_one_slot_keeps_higher_confidence(
    vertical_third_slots: list[Slot],
) -> None:
    """Two boxes in one slot: only the higher confidence one is kept."""
    low_conf = BoxDetection(
        label="cell_up", confidence=0.60, xyxy=(30.0, 20.0, 50.0, 60.0)
    )
    high_conf = BoxDetection(
        label="cell_down", confidence=0.91, xyxy=(35.0, 25.0, 55.0, 65.0)
    )

    # Order 1: low then high
    dets1 = to_detections(
        [low_conf, high_conf],
        frame_w=300,
        frame_h=100,
        slots=vertical_third_slots,
    )
    assert len(dets1) == 1
    assert dets1[0] == Detection(cls="cell_down", slot=1, confidence=0.91)

    # Order 2: high then low
    dets2 = to_detections(
        [high_conf, low_conf],
        frame_w=300,
        frame_h=100,
        slots=vertical_third_slots,
    )
    assert len(dets2) == 1
    assert dets2[0] == Detection(cls="cell_down", slot=1, confidence=0.91)


def test_confidence_is_type_float(vertical_third_slots: list[Slot]) -> None:
    """Confidence in the output is of type float."""
    box_int_conf = BoxDetection(
        label="cell_up", confidence=1, xyxy=(30.0, 20.0, 60.0, 80.0)
    )
    box_numpy_conf = BoxDetection(
        label="cell_down",
        confidence=np.float32(0.85),
        xyxy=(130.0, 20.0, 170.0, 80.0),
    )

    detections = to_detections(
        [box_int_conf, box_numpy_conf],
        frame_w=300,
        frame_h=100,
        slots=vertical_third_slots,
    )

    assert len(detections) == 2
    for det in detections:
        assert isinstance(det.confidence, float)
        assert type(det.confidence) is float


def test_load_slots_duplicate_ids_raises_value_error(tmp_path: Path) -> None:
    """load_slots raises ValueError on duplicate ids."""
    duplicate_yaml = tmp_path / "duplicate_slots.yaml"
    duplicate_yaml.write_text(
        """
slots:
  - id: 1
    x_min: 0.0
    x_max: 0.5
    y_min: 0.0
    y_max: 1.0
  - id: 1
    x_min: 0.5
    x_max: 1.0
    y_min: 0.0
    y_max: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate slot id"):
        load_slots(duplicate_yaml)


def test_load_slots_empty_list_raises_value_error(tmp_path: Path) -> None:
    """load_slots raises ValueError on an empty list."""
    empty_yaml = tmp_path / "empty_slots.yaml"
    empty_yaml.write_text("slots: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be empty"):
        load_slots(empty_yaml)


def test_shipped_slots_yaml_loads_and_has_ids_1_2_3() -> None:
    """The shipped configs/slots.yaml loads and has slot ids 1, 2, 3."""
    slots = load_slots("configs/slots.yaml")
    assert len(slots) == 3
    assert [s.id for s in slots] == [1, 2, 3]

    for s in slots:
        assert isinstance(s.id, int)
        assert 0.0 <= s.x_min < s.x_max <= 1.0
        assert 0.0 <= s.y_min < s.y_max <= 1.0


def test_invalid_frame_dimensions_raises_value_error(
    vertical_third_slots: list[Slot],
) -> None:
    """to_detections raises ValueError if frame_w or frame_h is not positive."""
    box = BoxDetection(
        label="cell_up", confidence=0.9, xyxy=(10.0, 10.0, 20.0, 20.0)
    )
    with pytest.raises(ValueError, match="must be positive"):
        to_detections([box], frame_w=0, frame_h=100, slots=vertical_third_slots)

    with pytest.raises(ValueError, match="must be positive"):
        to_detections([box], frame_w=100, frame_h=-5, slots=vertical_third_slots)
