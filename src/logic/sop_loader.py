"""Loader and validator for SOP configuration files."""

from dataclasses import dataclass
from pathlib import Path
import yaml

VALID_CLASSES = {"cell_up", "cell_down"}


@dataclass(frozen=True)
class Step:
    id: int
    name: str
    required_class: str
    slot: int


def load_sop(path: str | Path) -> tuple[int, list[Step]]:
    """Load and validate an SOP configuration file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid SOP file: root must be a mapping.")

    if "version" not in data:
        raise ValueError("Missing required field: 'version'")
    if "steps" not in data:
        raise ValueError("Missing required field: 'steps'")

    version = data["version"]
    raw_steps = data["steps"]
    if not isinstance(raw_steps, list):
        raise ValueError("'steps' must be a list")

    steps: list[Step] = []
    for expected_id, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(f"Step {expected_id} must be a dictionary")

        for field in ("id", "name", "require"):
            if field not in raw_step:
                raise ValueError(f"Step missing required field: '{field}'")

        step_id = raw_step["id"]
        if step_id != expected_id:
            raise ValueError(
                f"Step IDs must be sequential 1..N. Expected {expected_id}, got {step_id}"
            )

        require = raw_step["require"]
        if not isinstance(require, dict):
            raise ValueError(f"Step {step_id} 'require' must be a dictionary")

        for req_field in ("class", "slot"):
            if req_field not in require:
                raise ValueError(f"Step {step_id} require missing field: '{req_field}'")

        required_class = require["class"]
        if required_class not in VALID_CLASSES:
            raise ValueError(
                f"Invalid class '{required_class}' in step {step_id}. Must be one of: {sorted(VALID_CLASSES)}"
            )

        steps.append(
            Step(
                id=step_id,
                name=str(raw_step["name"]),
                required_class=str(required_class),
                slot=int(require["slot"]),
            )
        )

    return version, steps
