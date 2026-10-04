"""Script generating video with detection and status overlays."""

import argparse
from pathlib import Path
import time
import cv2

from src.detection.detector import YoloDetector
from src.detection.frames import probe
from src.logic.sop_loader import load_sop
from src.logic.state_machine import EventType, StepEngine
from src.pipeline.adapter import load_slots, to_detections
from src.pipeline.overlay import draw_overlay


def main(argv: list[str] | None = None) -> int:
    """CLI to render annotated overlay video."""
    parser = argparse.ArgumentParser(
        description="Generate annotated overlay video for battery verification."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/cells_v3.pt",
        help="Path to YOLO weights file (default: models/cells_v3.pt).",
    )
    parser.add_argument(
        "--slots",
        type=str,
        default="configs/slots.yaml",
        help="Path to slots configuration YAML (default: configs/slots.yaml).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/annotated.mp4",
        help="Path to output video file (default: outputs/annotated.mp4).",
    )
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    info = probe(video_path)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]

    # Build detector, slots, SOP and StepEngine same as run_pipeline.py
    slots = load_slots(args.slots)
    sop_version, steps = load_sop("configs/sop.yaml")
    engine = StepEngine(steps)
    detector = YoloDetector(args.weights)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    last_event_text = "Ready"
    last_event_type: EventType | None = None
    frame_times_sec: list[float] = []
    frames_written = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            boxes = detector.detect(frame)
            frame_h, frame_w = frame.shape[:2]
            dets = to_detections(boxes, frame_w, frame_h, slots)
            event = engine.update(dets)
            elapsed_sec = time.perf_counter() - t0
            frame_times_sec.append(elapsed_sec)

            if event.type != EventType.IN_PROGRESS:
                last_event_text = event.message
                last_event_type = event.type

            if engine.is_complete:
                step_text = "COMPLETE"
            elif engine.current_step is not None:
                step_text = (
                    f"Step {engine.current_step.id}/{len(engine.steps)}: {engine.current_step.name}"
                )
            else:
                step_text = "COMPLETE"

            annotated = draw_overlay(
                frame_bgr=frame,
                boxes=boxes,
                slots=slots,
                step_text=step_text,
                event_text=last_event_text,
                event_type=last_event_type,
                frame_index=frame_idx,
            )

            writer.write(annotated)
            frames_written += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"Frames written: {frames_written}")
    print(f"Output saved to: {out_path}")
    avg_ms = (
        (sum(frame_times_sec) / len(frame_times_sec) * 1000.0)
        if frame_times_sec
        else 0.0
    )
    print(f"Average time per frame: {avg_ms:.2f} ms")

    return 0


if __name__ == "__main__":
    main()
