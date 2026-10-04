"""Run the full battery verification pipeline on a video file."""

import argparse
from collections import defaultdict
from pathlib import Path
import cv2

from src.detection.detector import YoloDetector
from src.detection.frames import probe
from src.logic.sop_loader import load_sop
from src.logic.state_machine import Event, StepEngine
from src.logic.verification_log import VerificationLog
from src.pipeline.adapter import load_slots
from src.pipeline.runner import run


def read_video_frames(video_path: Path, stride: int = 1):
    """Yield frames from a video file with an optional stride."""
    probe(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield frame
            idx += 1
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    """CLI for running the full battery verification pipeline."""
    parser = argparse.ArgumentParser(
        description="Run battery verification pipeline on a video."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/cells_v3.pt",
        help="Path to YOLO model weights (default: models/cells_v3.pt).",
    )
    parser.add_argument(
        "--slots",
        type=str,
        default="configs/slots.yaml",
        help="Path to slots configuration YAML (default: configs/slots.yaml).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="outputs/verification.db",
        help="Path to output SQLite log (default: outputs/verification.db).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1).",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    slots = load_slots(args.slots)
    sop_version, steps = load_sop("configs/sop.yaml")
    engine = StepEngine(steps)
    detector = YoloDetector(args.weights)

    event_counts: dict[str, int] = defaultdict(int)
    frame_times_sec: list[float] = []

    def on_event(processed_idx: int, event: Event) -> None:
        actual_frame = processed_idx * args.stride
        step_str = f"step {event.step_id}" if event.step_id is not None else "step -"
        print(f"frame {actual_frame:05d}  {event.type.value}  {step_str}  {event.message}")
        event_counts[event.type.value] += 1

    def on_frame_time(processed_idx: int, elapsed_sec: float) -> None:
        frame_times_sec.append(elapsed_sec)

    frame_generator = read_video_frames(video_path, stride=args.stride)

    with VerificationLog(db_path) as log:
        run(
            frames=frame_generator,
            detector=detector,
            slots=slots,
            engine=engine,
            log=log,
            sop_version=sop_version,
            on_event=on_event,
            on_frame_time=on_frame_time,
        )
        chain_ok, bad_row = log.verify_chain()

    frames_processed = len(frame_times_sec)
    print(f"\nFrames processed: {frames_processed}")
    print("Event counts by type:")
    for ev_type in sorted(event_counts.keys()):
        print(f"  {ev_type}: {event_counts[ev_type]}")

    if chain_ok:
        print("Chain verification: CHAIN OK")
    else:
        print(f"Chain verification: CHAIN BROKEN at row {bad_row}")

    avg_ms = (
        (sum(frame_times_sec) / len(frame_times_sec) * 1000.0)
        if frame_times_sec
        else 0.0
    )
    print(f"Average time per frame: {avg_ms:.2f} ms")

    return 0


if __name__ == "__main__":
    main()
