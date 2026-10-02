"""Tests for video probing and frame extraction."""

from pathlib import Path
import cv2
import numpy as np
import pytest
from src.detection.frames import extract_frames, main, probe


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a synthetic video with MJPG codec, .avi, 10 fps, 30 frames, 64x64."""
    video_path = tmp_path / "synthetic.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(30):
        writer.write(frame)
    writer.release()
    return video_path


def test_probe_returns_fps_and_frame_count(synthetic_video: Path) -> None:
    """a. probe returns fps 10, frame_count 30."""
    info = probe(synthetic_video)
    assert info["fps"] == 10
    assert info["frame_count"] == 30
    assert info["width"] == 64
    assert info["height"] == 64
    assert info["duration_sec"] == pytest.approx(3.0)


def test_extract_frames_produces_six_files(synthetic_video: Path, tmp_path: Path) -> None:
    """b. extract with every_sec=0.5 produces 6 files."""
    out_dir = tmp_path / "frames"
    count = extract_frames(synthetic_video, out_dir, every_sec=0.5)
    assert count == 6

    saved_files = sorted(out_dir.glob("*.jpg"))
    assert len(saved_files) == 6
    expected_filenames = [
        "synthetic_000000.jpg",
        "synthetic_000005.jpg",
        "synthetic_000010.jpg",
        "synthetic_000015.jpg",
        "synthetic_000020.jpg",
        "synthetic_000025.jpg",
    ]
    assert [f.name for f in saved_files] == expected_filenames


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """c. missing file raises FileNotFoundError."""
    missing = tmp_path / "non_existent.avi"
    with pytest.raises(FileNotFoundError):
        probe(missing)
    with pytest.raises(FileNotFoundError):
        extract_frames(missing, tmp_path / "out")


def test_renamed_text_file_raises_value_error(tmp_path: Path) -> None:
    """d. a text file renamed .avi raises ValueError."""
    fake_avi = tmp_path / "fake.avi"
    fake_avi.write_text("plain text, not video", encoding="utf-8")
    with pytest.raises(ValueError):
        probe(fake_avi)
    with pytest.raises(ValueError):
        extract_frames(fake_avi, tmp_path / "out")


def test_cli_probe_and_extract(
    synthetic_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI probe and extract commands."""
    main(["probe", str(synthetic_video)])
    captured = capsys.readouterr()
    assert "frame_count: 30" in captured.out

    out_dir = tmp_path / "cli_out"
    main(["extract", str(synthetic_video), "--every", "0.5", "--out", str(out_dir)])
    captured = capsys.readouterr()
    assert "Extracted 6 frames" in captured.out
    assert len(list(out_dir.glob("*.jpg"))) == 6
