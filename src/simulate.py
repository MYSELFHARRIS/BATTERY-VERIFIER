"""Simulation runner for battery verification procedures."""

import argparse
from pathlib import Path
from typing import Any

from src.logic.sop_loader import VALID_CLASSES, Step, load_sop
from src.logic.state_machine import Detection, Event, EventType, StepEngine
from src.logic.verification_log import VerificationLog


def _resolve_sop_path(sop_path: str | Path | None = None) -> Path:
    """Resolve the path to configs/sop.yaml."""
    if sop_path is not None:
        p = Path(sop_path)
        if p.exists():
            return p
    cwd_path = Path("configs/sop.yaml")
    if cwd_path.exists():
        return cwd_path
    repo_root = Path(__file__).resolve().parent.parent
    repo_path = repo_root / "configs" / "sop.yaml"
    if repo_path.exists():
        return repo_path
    raise FileNotFoundError("Could not find configs/sop.yaml")


def _get_opposite_class(cls: str, all_classes: set[str]) -> str:
    """Derive the opposite class dynamically."""
    candidates = all_classes - {cls}
    if candidates:
        return sorted(candidates)[0]
    return "cell_down" if cls == "cell_up" else "cell_up"


def _step_frames(step: Step, count: int = 8, confidence: float = 0.9) -> list[list[Detection]]:
    """Generate detection frames for a step."""
    return [[Detection(cls=step.required_class, slot=step.slot, confidence=confidence)] for _ in range(count)]


def _empty_frames(count: int = 2) -> list[list[Detection]]:
    """Generate empty detection frames."""
    return [[] for _ in range(count)]


def _correct_step(step: Step) -> list[list[Detection]]:
    """Generate 8 required detection frames followed by 2 empty frames."""
    return _step_frames(step, count=8, confidence=0.9) + _empty_frames(count=2)


def build_scenario(name: str, steps: list[Step]) -> list[list[Detection]]:
    """Build simulated frames for a given scenario name and steps."""
    if not steps:
        raise ValueError("Steps list cannot be empty.")

    all_classes = {s.required_class for s in steps} | VALID_CLASSES

    if name == "correct":
        # correct: for each step in order, 8 frames of its required detection, then 2 empty frames.
        frames: list[list[Detection]] = []
        for step in steps:
            frames.extend(_correct_step(step))
        return frames

    if len(steps) < 3:
        raise ValueError("Scenarios 'flipped' and 'skipped' require at least 3 steps.")

    step1, step2, step3 = steps[0], steps[1], steps[2]

    if name == "flipped":
        # flipped: step 1 correct; then 8 frames of the OPPOSITE class in step 2's
        # slot; then 8 correct frames for step 2; then step 3 correct.
        opp_class = _get_opposite_class(step2.required_class, all_classes)
        frames: list[list[Detection]] = []
        frames.extend(_correct_step(step1))
        frames.extend([[Detection(cls=opp_class, slot=step2.slot, confidence=0.9)] for _ in range(8)])
        frames.extend(_correct_step(step2))
        frames.extend(_correct_step(step3))
        for step in steps[3:]:
            frames.extend(_correct_step(step))
        return frames

    if name == "skipped":
        # skipped: step 1 correct; then 8 frames of step 3's required detection
        # while step 2 is current; then step 2 correct; then step 3 correct.
        frames: list[list[Detection]] = []
        frames.extend(_correct_step(step1))
        frames.extend([[Detection(cls=step3.required_class, slot=step3.slot, confidence=0.9)] for _ in range(8)])
        frames.extend(_correct_step(step2))
        frames.extend(_correct_step(step3))
        for step in steps[3:]:
            frames.extend(_correct_step(step))
        return frames

    raise ValueError(f"Unknown scenario: '{name}'. Expected 'correct', 'flipped', or 'skipped'.")


def run_scenario(
    name: str,
    db_path: str | Path,
    sop_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a verification scenario and record STEP_VERIFIED events into the VerificationLog."""
    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()
    db_file.parent.mkdir(parents=True, exist_ok=True)

    sop_file = _resolve_sop_path(sop_path)
    sop_version, steps = load_sop(sop_file)

    frames = build_scenario(name, steps)
    engine = StepEngine(steps)

    events: list[Event] = []
    with VerificationLog(db_file) as log:
        for frame_dets in frames:
            event = engine.update(frame_dets)
            events.append(event)
            if event.type == EventType.STEP_VERIFIED:
                assert event.step_id is not None
                assert event.confidence is not None
                log.append(
                    step_id=event.step_id,
                    result="VERIFIED",
                    confidence=float(event.confidence),
                    sop_version=sop_version,
                )

        assert log.conn is not None
        cursor = log.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM log")
        rows = cursor.fetchone()[0]
        chain_ok, bad_row = log.verify_chain()

    return {
        "events": events,
        "rows": int(rows),
        "chain_ok": bool(chain_ok),
        "bad_row": bad_row,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running simulation scenarios."""
    parser = argparse.ArgumentParser(description="Simulate battery verification scenarios.")
    parser.add_argument(
        "scenario",
        choices=["correct", "flipped", "skipped"],
        help="Scenario to run: correct | flipped | skipped",
    )
    args = parser.parse_args(argv)

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / f"{args.scenario}.db"

    result = run_scenario(args.scenario, db_path)

    prev_event_type: EventType | None = None
    for i, ev in enumerate(result["events"]):
        if ev.type != prev_event_type:
            step_str = f"step {ev.step_id}" if ev.step_id is not None else "step -"
            conf_str = f"conf {ev.confidence:.2f}" if ev.confidence is not None else "conf ----"
            print(f"frame {i:03d}  {ev.type.value}  {step_str}  {conf_str}  {ev.message}")
            prev_event_type = ev.type

    print(f"{result['rows']} rows")
    if result["chain_ok"]:
        print("CHAIN OK")
    else:
        print(f"CHAIN BROKEN at row {result['bad_row']}")

    return 0


if __name__ == "__main__":
    main()
