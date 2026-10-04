"""Overlay generation for battery verification visual feedback."""

import cv2
import numpy as np

from src.detection.detector import BoxDetection, draw_boxes
from src.logic.state_machine import EventType
from src.pipeline.adapter import Slot


def get_event_color(event_type: EventType | str | None) -> tuple[int, int, int]:
    """Return BGR color corresponding to the event type."""
    if isinstance(event_type, EventType):
        name = event_type.name
    elif isinstance(event_type, str):
        name = event_type
    else:
        name = None

    if name == "STEP_VERIFIED":
        return (0, 255, 0)  # Green
    elif name == "WRONG_ORIENTATION":
        return (0, 0, 255)  # Red
    elif name == "SKIPPED_STEP":
        return (0, 165, 255)  # Orange (BGR)
    else:
        return (255, 255, 255)  # White


def draw_overlay(
    frame_bgr: np.ndarray,
    boxes: list[BoxDetection],
    slots: list[Slot],
    step_text: str,
    event_text: str,
    event_type: EventType | str | None,
    frame_index: int,
) -> np.ndarray:
    """Render bounding boxes, slot bands, and status panel overlay on a copy of the frame.

    Args:
        frame_bgr: Input BGR image array.
        boxes: List of BoxDetection objects to draw.
        slots: List of Slot definitions with normalized coordinates.
        step_text: Step information (e.g. "Step 1/3: Inspect cell 1" or "COMPLETE").
        event_text: Recent event message.
        event_type: EventType enum, string, or None.
        frame_index: 0-based frame number.

    Returns:
        New numpy array with all overlays rendered.
    """
    # Draw boxes on a copy of frame_bgr using draw_boxes
    canvas = draw_boxes(frame_bgr, boxes)
    frame_h, frame_w = canvas.shape[:2]

    # Calculate scale factor relative to 1920x864
    scale_w = frame_w / 1920.0
    scale_h = frame_h / 864.0
    scale = max(0.4, min(scale_w, scale_h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2 if frame_w >= 1000 else 1

    # Status panel dimensions
    panel_height = max(50, int(frame_h * 0.10))

    # Draw each slot as a thin vertical band outline with a label "S1", "S2", ...
    slot_color = (220, 220, 220)

    for slot in slots:
        x_min = int(slot.x_min * frame_w)
        x_max = int(slot.x_max * frame_w)
        y_min = int(slot.y_min * frame_h)
        y_max = int(slot.y_max * frame_h)

        # Thin vertical band outline
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), slot_color, 1)

        # Slot label "S{id}"
        slot_label = f"S{slot.id}"
        label_scale = max(0.4, scale * 0.8)
        (tw, th), _ = cv2.getTextSize(slot_label, font, label_scale, 1)
        label_x = x_min + max(4, int(10 * scale))
        label_y = min(y_max - 5, max(y_min + th + 5, panel_height + th + 10))
        cv2.putText(
            canvas,
            slot_label,
            (label_x, label_y),
            font,
            label_scale,
            slot_color,
            1,
            cv2.LINE_AA,
        )

    # Draw dark status panel at the top
    cv2.rectangle(canvas, (0, 0), (frame_w, panel_height), (30, 30, 30), -1)
    cv2.line(canvas, (0, panel_height), (frame_w, panel_height), (60, 60, 60), 1)

    text_scale = max(0.45, scale * 0.9)
    event_color = get_event_color(event_type)

    line1_y = int(panel_height * 0.42)
    line2_y = int(panel_height * 0.84)
    left_x = max(10, int(20 * scale))

    # Line 1: step_text
    cv2.putText(
        canvas,
        step_text,
        (left_x, line1_y),
        font,
        text_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Line 2: event_text
    cv2.putText(
        canvas,
        event_text,
        (left_x, line2_y),
        font,
        text_scale,
        event_color,
        thickness,
        cv2.LINE_AA,
    )

    # Right side: "frame N" (0-based)
    frame_str = f"frame {frame_index}"
    (ft_w, ft_h), _ = cv2.getTextSize(frame_str, font, text_scale, thickness)
    right_x = max(left_x, frame_w - ft_w - max(10, int(20 * scale)))
    cv2.putText(
        canvas,
        frame_str,
        (right_x, line1_y),
        font,
        text_scale,
        (200, 200, 200),
        thickness,
        cv2.LINE_AA,
    )

    return canvas
