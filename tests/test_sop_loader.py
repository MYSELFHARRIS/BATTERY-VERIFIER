"""Tests for SOP configuration loader."""

from pathlib import Path
import pytest
from src.logic.sop_loader import Step, load_sop


def test_valid_sop() -> None:
    """Test loading the default valid SOP configuration."""
    version, steps = load_sop("configs/sop.yaml")
    assert version == 1
    assert len(steps) == 3
    assert steps[0] == Step(id=1, name="Inspect first cell orientation", required_class="cell_up", slot=1)
    assert steps[1] == Step(id=2, name="Inspect second cell orientation", required_class="cell_down", slot=2)
    assert steps[2] == Step(id=3, name="Inspect third cell orientation", required_class="cell_up", slot=3)


def test_missing_field(tmp_path: Path) -> None:
    """Test that missing required fields raise ValueError."""
    bad_sop = tmp_path / "missing_field.yaml"
    bad_sop.write_text(
        """
version: 1
steps:
  - id: 1
    name: "Missing require step"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_sop(bad_sop)


def test_invalid_class(tmp_path: Path) -> None:
    """Test that an invalid cell class raises ValueError."""
    bad_sop = tmp_path / "invalid_class.yaml"
    bad_sop.write_text(
        """
version: 1
steps:
  - id: 1
    name: "Invalid class step"
    require:
      class: upside_down
      slot: 1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid class"):
        load_sop(bad_sop)


def test_out_of_order_ids(tmp_path: Path) -> None:
    """Test that non-sequential step IDs raise ValueError."""
    bad_sop = tmp_path / "out_of_order.yaml"
    bad_sop.write_text(
        """
version: 1
steps:
  - id: 2
    name: "Wrong first id"
    require:
      class: cell_up
      slot: 1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Step IDs must be sequential"):
        load_sop(bad_sop)
