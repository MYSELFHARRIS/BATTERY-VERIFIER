"""Detect objects in a single video frame using YOLO and save annotated image."""

import argparse
from pathlib import Path
import cv2

from src.detection.detector import YoloDetector, draw_boxes
from src.detection.frames import probe


def main(argv: list[str] | None = None) -> int:
    """CLI to detect objects on a specific video frame."""
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on a video frame."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to process (default: 0).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolov8n.pt",
        help="Path to YOLO weights file (default: models/yolov8n.pt).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/boxed.jpg",
        help="Path to save annotated output image (default: outputs/boxed.jpg).",
    )

    args = parser.parse_args(argv)

    video_path = Path(args.video)
    info = probe(video_path)

    if args.frame < 0 or args.frame >= info["frame_count"]:
        raise IndexError(
            f"Frame index {args.frame} is out of bounds (0..{info['frame_count'] - 1})."
        )

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {args.frame} from '{video_path}'.")

    detector = YoloDetector(args.weights)
    detections = detector.detect(frame)

    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f"  {det.label}: {det.confidence:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_boxes(frame, detections)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")

    return 0


if __name__ == "__main__":
    main()
