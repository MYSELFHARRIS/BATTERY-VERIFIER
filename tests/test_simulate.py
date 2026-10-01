"""Tests for simulate module scenarios and execution."""

from pathlib import Path
import pytest
from src.logic.sop_loader import Step, load_sop
from src.logic.state_machine import EventType
from src.simulate import build_scenario, run_scenario


def test_correct_scenario(tmp_path: Path) -> None:
    """1. correct: 3 rows, chain_ok True, last event type is COMPLETE."""
    db_path = tmp_path / "correct.db"
    result = run_scenario("correct", db_path)

    assert result["rows"] == 3
    assert result["chain_ok"] is True
    assert result["bad_row"] is None
    assert len(result["events"]) > 0
    assert result["events"][-1].type == EventType.COMPLETE


def test_flipped_scenario(tmp_path: Path) -> None:
    """2. flipped: events include WRONG_ORIENTATION, rows == 3, chain_ok True."""
    db_path = tmp_path / "flipped.db"
    result = run_scenario("flipped", db_path)

    assert result["rows"] == 3
    assert result["chain_ok"] is True
    assert result["bad_row"] is None
    event_types = [ev.type for ev in result["events"]]
    assert EventType.WRONG_ORIENTATION in event_types
    assert result["events"][-1].type == EventType.COMPLETE


def test_skipped_scenario(tmp_path: Path) -> None:
    """3. skipped: events include SKIPPED_STEP, rows == 3, chain_ok True."""
    db_path = tmp_path / "skipped.db"
    result = run_scenario("skipped", db_path)

    assert result["rows"] == 3
    assert result["chain_ok"] is True
    assert result["bad_row"] is None
    event_types = [ev.type for ev in result["events"]]
    assert EventType.SKIPPED_STEP in event_types
    assert result["events"][-1].type == EventType.COMPLETE


def test_build_scenario_confidence_and_structure() -> None:
    """Verify build_scenario confidence is 0.9 and shapes are valid."""
    _, steps = load_sop("configs/sop.yaml")

    for scenario_name in ("correct", "flipped", "skipped"):
        frames = build_scenario(scenario_name, steps)
        assert len(frames) > 0
        for frame in frames:
            for det in frame:
                assert det.confidence == pytest.approx(0.9)


def test_build_scenario_invalid_name() -> None:
    """Verify build_scenario rejects invalid scenario name."""
    _, steps = load_sop("configs/sop.yaml")
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_scenario("invalid_scenario", steps)


def test_existing_db_is_deleted_on_run(tmp_path: Path) -> None:
    """Verify run_scenario deletes preexisting db file."""
    db_path = tmp_path / "test_delete.db"
    db_path.write_text("dummy old data", encoding="utf-8")
    assert db_path.exists()

    result = run_scenario("correct", db_path)
    assert result["rows"] == 3
    assert result["chain_ok"] is True
