"""Unit test verifying Antigravity adapter schema compliance across all captured runs and steps."""

import os
import sys
import pytest

from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from longhorizon_guard.storage.reader import load_dataset, validate_metadata
ANTIGRAVITY_SESSIONS_PATH = str(REPO_ROOT / "findings" / "antigravity_sessions.json")
if not os.path.exists(ANTIGRAVITY_SESSIONS_PATH):
    ANTIGRAVITY_SESSIONS_PATH = str(REPO_ROOT.parent / "findings" / "antigravity_sessions.json")


def test_antigravity_adapter_schema_compliance():
    assert os.path.exists(ANTIGRAVITY_SESSIONS_PATH), f"Missing {ANTIGRAVITY_SESSIONS_PATH}"

    runs = load_dataset(ANTIGRAVITY_SESSIONS_PATH)
    assert len(runs) > 0, "No runs found in antigravity_sessions.json"

    total_steps_checked = 0

    for run_idx, run in enumerate(runs):
        meta = run.get("metadata", {})
        traj = run.get("trajectory", {})

        # Metadata validation
        assert validate_metadata(meta), f"Run {run_idx} ({meta.get('run_id')}) failed metadata validation"
        assert meta.get("final_status") in ("completed_unverified", "timeout", "error"), f"Unexpected status: {meta.get('final_status')}"
        assert meta.get("source_dataset") == "antigravity_session"
        assert isinstance(meta.get("source_llm_model"), str)

        # Trajectory & steps validation
        assert traj is not None, f"Run {run_idx} missing trajectory"
        steps = traj.get("steps", [])
        assert len(steps) > 0, f"Run {run_idx} has empty steps"

        for step in steps:
            total_steps_checked += 1
            idx = step.get("step_index")

            # Strict field type assertions
            assert isinstance(step.get("step_index"), int), f"Step {idx} step_index is not int: {type(step.get('step_index'))}"
            assert isinstance(step.get("reasoning"), str), f"Step {idx} reasoning is not str: {type(step.get('reasoning'))}"
            assert isinstance(step.get("action_name"), str), f"Step {idx} action_name is not str: {type(step.get('action_name'))}"
            assert isinstance(step.get("action_args"), dict), f"Step {idx} action_args is not dict: {type(step.get('action_args'))}"

            # Optional field type assertions if present
            if step.get("timestamp") is not None:
                assert isinstance(step.get("timestamp"), float), f"Step {idx} timestamp is not float"
            if step.get("tool_response") is not None:
                assert isinstance(step.get("tool_response"), str), f"Step {idx} tool_response is not str"

    print(f"\nOK: Verified {len(runs)} run(s) and {total_steps_checked} total steps. All step types conform to SCHEMA.md!")


if __name__ == "__main__":
    test_antigravity_adapter_schema_compliance()
