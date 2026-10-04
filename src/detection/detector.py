"""YOLO-based object detection and bounding box visualization."""

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class BoxDetection:
    """Detected bounding box with label, confidence, and xyxy coordinates."""

    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def unstretch_xyxy(
    xyxy: tuple[float, float, float, float] | list[float],
    orig_w: int | float,
    orig_h: int | float,
    size: int | float,
) -> tuple[float, float, float, float]:
    """Convert bounding box from stretched-square coordinates back to original frame coordinates.

    Args:
        xyxy: Coordinates in stretched square space (x1, y1, x2, y2).
        orig_w: Original frame width (> 0).
        orig_h: Original frame height (> 0).
        size: Stretched square dimension (> 0).

    Returns:
        Unstretched coordinates in original frame space (x1, y1, x2, y2).

    Raises:
        ValueError: If orig_w, orig_h, or size is not positive.
    """
    if orig_w <= 0 or orig_h <= 0 or size <= 0:
        raise ValueError(
            f"orig_w, orig_h, and size must be positive, got orig_w={orig_w}, orig_h={orig_h}, size={size}"
        )

    x1, y1, x2, y2 = xyxy
    scale_x = float(orig_w) / float(size)
    scale_y = float(orig_h) / float(size)
    return (
        float(x1 * scale_x),
        float(y1 * scale_y),
        float(x2 * scale_x),
        float(y2 * scale_y),
    )


class YoloDetector:
    """YOLO model wrapper with disabled auto-download.

    Note: cells_v3 was trained on stretched 640x640 images; set stretch_size=None
    for any model trained with letterboxing.
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.25,
        stretch_size: int | None = 640,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights file not found: '{self.weights_path}'. "
                "Automatic download is disabled."
            )
        self.conf = conf
        self.stretch_size = stretch_size

        # Lazy import of ultralytics to allow importing without weights/dependencies
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights_path))

    def detect(self, frame_bgr: np.ndarray) -> list[BoxDetection]:
        """Run object detection on a BGR image frame and return list of BoxDetection."""
        orig_h, orig_w = frame_bgr.shape[:2]
        if self.stretch_size is not None:
            input_frame = cv2.resize(
                frame_bgr, (self.stretch_size, self.stretch_size)
            )
        else:
            input_frame = frame_bgr

        results = self.model(input_frame, conf=self.conf, verbose=False)
        detections: list[BoxDetection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = (
                    result.names[cls_id]
                    if result.names and cls_id in result.names
                    else str(cls_id)
                )
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_xyxy = (float(x1), float(y1), float(x2), float(y2))

                if self.stretch_size is not None:
                    final_xyxy = unstretch_xyxy(
                        raw_xyxy,
                        orig_w=orig_w,
                        orig_h=orig_h,
                        size=self.stretch_size,
                    )
                else:
                    final_xyxy = raw_xyxy

                detections.append(
                    BoxDetection(
                        label=label,
                        confidence=confidence,
                        xyxy=final_xyxy,
                    )
                )

        return detections


def draw_boxes(
    frame_bgr: np.ndarray, detections: list[BoxDetection]
) -> np.ndarray:
    """Draw bounding boxes and label text on a copy of the frame.

    Does not modify the input frame array.
    """
    annotated = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        # Draw bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label and confidence text
        text = f"{det.label} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        text_y = max(y1 - 5, text_h + 5)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return annotated
