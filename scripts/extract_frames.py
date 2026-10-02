"""Extract frames from videos into train/val/test folders (split by video)."""
from pathlib import Path

import cv2

VIDEO_DIR = Path("data/videos")
OUT_DIR = Path("data/frames")
EVERY_SEC = 0.3
SPLITS = {"train": [1, 2, 3, 4], "val": [5], "test": [6]}


def extract(video: Path, out_dir: Path, every_sec: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_sec))
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            name = f"{video.stem}_{idx:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        idx += 1
    cap.release()
    return saved


def main() -> None:
    for split, numbers in SPLITS.items():
        total = 0
        for n in numbers:
            video = VIDEO_DIR / f"video_{n:02d}.mp4"
            count = extract(video, OUT_DIR / split, EVERY_SEC)
            print(f"{split:5s} {video.name}: {count} frames")
            total += count
        print(f"{split:5s} TOTAL: {total}\n")


if __name__ == "__main__":
    main()