"""Adapter converting raw vision bounding boxes to StepEngine Detections."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import yaml

from src.detection.detector import BoxDetection
from src.logic.state_machine import Detection

ALLOWED_LABELS = {"cell_up", "cell_down"}


@dataclass(frozen=True)
class Slot:
    """Normalized spatial region definition for a verification slot."""

    id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def load_slots(path: str | Path) -> list[Slot]:
    """Load and validate slots configuration from a YAML file.

    Raises:
        ValueError: On empty slots list, missing required fields, or duplicate slot IDs.
        FileNotFoundError: If the configuration file is not found.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Slots config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError("Slots configuration file is empty.")

    if isinstance(data, dict) and "slots" in data:
        raw_slots = data["slots"]
    elif isinstance(data, list):
        raw_slots = data
    else:
        raise ValueError(
            "Invalid slots file format: root must be a list or a dictionary with 'slots' key."
        )

    if not isinstance(raw_slots, list) or len(raw_slots) == 0:
        raise ValueError("Slots list cannot be empty.")

    slots: list[Slot] = []
    seen_ids: set[int] = set()

    for item in raw_slots:
        if not isinstance(item, dict):
            raise ValueError(f"Slot entry must be a dictionary, got: {type(item)}")

        for req in ("id", "x_min", "x_max", "y_min", "y_max"):
            if req not in item:
                raise ValueError(f"Slot missing required field '{req}': {item}")

        slot_id = int(item["id"])
        if slot_id in seen_ids:
            raise ValueError(f"Duplicate slot id: {slot_id}")
        seen_ids.add(slot_id)

        slots.append(
            Slot(
                id=slot_id,
                x_min=float(item["x_min"]),
                x_max=float(item["x_max"]),
                y_min=float(item["y_min"]),
                y_max=float(item["y_max"]),
            )
        )

    return slots


def to_detections(
    boxes: Sequence[BoxDetection],
    frame_w: int | float,
    frame_h: int | float,
    slots: Sequence[Slot],
) -> list[Detection]:
    """Convert model box detections to state machine Detections assigned to slots.

    Args:
        boxes: Iterable of BoxDetection objects (label, confidence, xyxy).
        frame_w: Frame width in pixels (> 0).
        frame_h: Frame height in pixels (> 0).
        slots: Iterable of Slot definitions.

    Returns:
        List of Detection objects (cls, slot, confidence).

    Raises:
        ValueError: If frame_w or frame_h is not positive.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(
            f"frame_w and frame_h must be positive, got w={frame_w}, h={frame_h}"
        )

    best_by_slot: dict[int, tuple[float, Detection]] = {}

    for box in boxes:
        # Only labels "cell_up" and "cell_down" are used. Map by label NAME.
        if box.label not in ALLOWED_LABELS:
            continue

        x1, y1, x2, y2 = box.xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        norm_x = cx / frame_w
        norm_y = cy / frame_h

        matched_slot: Slot | None = None
        for slot in slots:
            if (
                slot.x_min <= norm_x <= slot.x_max
                and slot.y_min <= norm_y <= slot.y_max
            ):
                matched_slot = slot
                break

        if matched_slot is None:
            continue

        conf = float(box.confidence)
        detection = Detection(
            cls=box.label,
            slot=matched_slot.id,
            confidence=conf,
        )

        if (
            matched_slot.id not in best_by_slot
            or conf > best_by_slot[matched_slot.id][0]
        ):
            best_by_slot[matched_slot.id] = (conf, detection)

    return [det for _, (_, det) in sorted(best_by_slot.items())]
