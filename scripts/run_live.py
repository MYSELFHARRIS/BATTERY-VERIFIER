"""Live camera runner for real-time battery verification."""

import argparse
from pathlib import Path
import sys
import cv2

from src.detection.detector import YoloDetector
from src.logic.sop_loader import load_sop
from src.logic.state_machine import StepEngine
from src.logic.verification_log import VerificationLog
from src.pipeline.adapter import load_slots
from src.pipeline.live import LiveSession
from src.pipeline.overlay import draw_overlay


def main(argv: list[str] | None = None) -> int:
    """CLI to run battery verification live on camera."""
    parser = argparse.ArgumentParser(
        description="Run live camera battery verification."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
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
        default="outputs/live.db",
        help="Path to verification SQLite database (default: outputs/live.db).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture height (default: 720).",
    )
    parser.add_argument(
        "--save-raw",
        type=str,
        default=None,
        help="Optional .mp4 path to save raw camera frames.",
    )
    parser.add_argument(
        "--save-overlay",
        type=str,
        default=None,
        help="Optional .mp4 path to save annotated overlay frames.",
    )

    args = parser.parse_args(argv)

    # Initialize camera using DirectShow
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(
            f"Error: Could not open camera {args.camera} using CAP_DSHOW.",
            file=sys.stderr,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read back actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    # Load resources
    slots = load_slots(args.slots)
    sop_version, steps = load_sop("configs/sop.yaml")
    engine = StepEngine(steps)
    detector = YoloDetector(args.weights)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer: cv2.VideoWriter | None = None
    overlay_writer: cv2.VideoWriter | None = None

    if args.save_raw:
        raw_p = Path(args.save_raw)
        raw_p.parent.mkdir(parents=True, exist_ok=True)
        raw_writer = cv2.VideoWriter(str(raw_p), fourcc, fps, (actual_w, actual_h))

    if args.save_overlay:
        overlay_p = Path(args.save_overlay)
        overlay_p.parent.mkdir(parents=True, exist_ok=True)
        overlay_writer = cv2.VideoWriter(
            str(overlay_p), fourcc, fps, (actual_w, actual_h)
        )

    print("Camera running. Press 'q' to quit, 'r' to reset sequence.")

    frame_index = 0
    with VerificationLog(db_path) as log:
        session = LiveSession(
            detector=detector,
            slots=slots,
            engine=engine,
            log=log,
            sop_version=sop_version,
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to grab frame from camera.", file=sys.stderr)
                    break

                boxes, event = session.process(frame)

                # Determine step status text
                if session.engine.is_complete:
                    step_text = "COMPLETE"
                elif session.engine.current_step is not None:
                    step_text = (
                        f"Step {session.engine.current_step.id}/{len(session.engine.steps)}: "
                        f"{session.engine.current_step.name}"
                    )
                else:
                    step_text = "COMPLETE"

                annotated = draw_overlay(
                    frame_bgr=frame,
                    boxes=boxes,
                    slots=session.slots,
                    step_text=step_text,
                    event_text=session.last_event_text,
                    event_type=session.last_event_type,
                    frame_index=frame_index,
                )

                if raw_writer is not None:
                    raw_writer.write(frame)
                if overlay_writer is not None:
                    overlay_writer.write(annotated)

                cv2.imshow("Battery Verifier - Live", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    session.reset()
                    print("Sequence reset.")

                frame_index += 1
        finally:
            cap.release()
            if raw_writer is not None:
                raw_writer.release()
            if overlay_writer is not None:
                overlay_writer.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
