"""Video probing and frame extraction utilities."""

import argparse
from pathlib import Path
from typing import Any
import cv2


def probe(video_path: str | Path) -> dict[str, Any]:
    """Probe video metadata including fps, frame_count, width, height, and duration_sec.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open the video file or file is corrupted.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV could not open video file: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata or corrupted video file: {path}")

    duration_sec = float(frame_count / fps) if fps > 0 else 0.0

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    every_sec: float = 0.3,
) -> int:
    """Extract frames from video and save as JPEGs into out_dir.

    Frames are named <video_stem>_<frame_index:06d>.jpg.
    Keeps one frame per every_sec seconds based on video fps.

    Returns:
        Total number of extracted frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open the video file or file is corrupted.
    """
    if every_sec <= 0:
        raise ValueError("every_sec must be positive.")

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV could not open video file: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata or corrupted video file: {path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    video_stem = path.stem
    step = max(1, round(every_sec * fps))

    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            filename = f"{video_stem}_{frame_idx:06d}.jpg"
            dest = out_path / filename
            cv2.imwrite(str(dest), frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    return saved_count


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for video probe and frame extraction."""
    parser = argparse.ArgumentParser(description="Video probing and frame extraction utility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # probe subcommand
    probe_parser = subparsers.add_parser("probe", help="Probe video metadata.")
    probe_parser.add_argument("video", type=str, help="Path to video file.")

    # extract subcommand
    extract_parser = subparsers.add_parser("extract", help="Extract frames from video.")
    extract_parser.add_argument("video", type=str, help="Path to video file.")
    extract_parser.add_argument(
        "--every",
        type=float,
        default=0.3,
        help="Extraction interval in seconds (default: 0.3).",
    )
    extract_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for extracted frames (default: data/frames/<video_stem>).",
    )

    args = parser.parse_args(argv)

    if args.command == "probe":
        info = probe(args.video)
        for k, v in info.items():
            print(f"{k}: {v}")
    elif args.command == "extract":
        video_p = Path(args.video)
        out_dir = Path(args.out) if args.out is not None else Path("data/frames") / video_p.stem
        count = extract_frames(video_p, out_dir, every_sec=args.every)
        print(f"Extracted {count} frames to {out_dir}")

    return 0


if __name__ == "__main__":
    main()
