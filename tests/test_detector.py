"""Tests for YOLO detector wrapper and bounding box visualization."""

from pathlib import Path
import numpy as np
import pytest
from src.detection.detector import (
    BoxDetection,
    YoloDetector,
    draw_boxes,
    unstretch_xyxy,
)


def test_missing_weights_raises_file_not_found(tmp_path: Path) -> None:
    """Missing weights path raises FileNotFoundError without attempting download."""
    missing_path = tmp_path / "models" / "yolov8n.pt"
    with pytest.raises(FileNotFoundError, match="Model weights file not found"):
        YoloDetector(missing_path)


def test_draw_boxes_same_shape_and_unmodified_input() -> None:
    """draw_boxes returns the same shape and does not modify the input array."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame_original = frame.copy()
    detections = [
        BoxDetection(
            label="cell_up", confidence=0.92, xyxy=(10.0, 15.0, 50.0, 80.0)
        ),
        BoxDetection(
            label="cell_down", confidence=0.85, xyxy=(60.0, 20.0, 100.0, 90.0)
        ),
    ]

    annotated = draw_boxes(frame, detections)

    assert annotated.shape == frame.shape
    assert np.array_equal(frame, frame_original)
    assert not np.array_equal(annotated, frame)


def test_draw_boxes_empty_detections() -> None:
    """draw_boxes with no detections returns an identical copy without modification."""
    frame = np.ones((80, 80, 3), dtype=np.uint8) * 128
    annotated = draw_boxes(frame, [])

    assert annotated.shape == frame.shape
    assert np.array_equal(annotated, frame)
    assert annotated is not frame


def test_box_detection_dataclass() -> None:
    """BoxDetection dataclass holds label, confidence, and xyxy bounding box."""
    det = BoxDetection(label="battery", confidence=0.99, xyxy=(0.0, 1.0, 2.0, 3.0))
    assert det.label == "battery"
    assert det.confidence == 0.99
    assert det.xyxy == (0.0, 1.0, 2.0, 3.0)


def test_unstretch_xyxy_full_frame() -> None:
    """unstretch_xyxy maps (0, 0, 640, 640) with a 1920x864 frame to (0, 0, 1920, 864)."""
    mapped = unstretch_xyxy((0, 0, 640, 640), orig_w=1920, orig_h=864, size=640)
    assert mapped == (0.0, 0.0, 1920.0, 864.0)


def test_unstretch_xyxy_quadrant() -> None:
    """unstretch_xyxy maps (320, 320, 640, 640) to (960, 432, 1920, 864)."""
    mapped = unstretch_xyxy(
        (320, 320, 640, 640), orig_w=1920, orig_h=864, size=640
    )
    assert mapped == (960.0, 432.0, 1920.0, 864.0)


def test_unstretch_xyxy_zero_width_raises_value_error() -> None:
    """unstretch_xyxy raises ValueError for a zero width."""
    with pytest.raises(ValueError, match="must be positive"):
        unstretch_xyxy((0, 0, 640, 640), orig_w=0, orig_h=864, size=640)

    with pytest.raises(ValueError, match="must be positive"):
        unstretch_xyxy((0, 0, 640, 640), orig_w=1920, orig_h=0, size=640)

    with pytest.raises(ValueError, match="must be positive"):
        unstretch_xyxy((0, 0, 640, 640), orig_w=1920, orig_h=864, size=0)
