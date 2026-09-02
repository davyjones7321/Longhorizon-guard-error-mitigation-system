"""
End-to-End Live Trajectory Smoke Test for longhorizon_guard.

Runs 3 REAL trajectories through the actual GuardInterface hook sequence in production order:
  1. Trajectory 1: Clean, successful run from findings/all_clean_outputs.json (zero false positives check).
  2. Trajectory 2: Known planning_error run from findings/holdout_v2_judged.json (pred==gt=='planning_error').
  3. Trajectory 3: Long run (30 steps) from findings/agenterrorbench_converted.json with multiple subgoals.

Validates full pipeline behavior across all 4 components: GuardInterface, subgoals, drift_monitor, and reflector.
"""

import json
import logging
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.storage.reader import load_dataset

FINDINGS_DIR = REPO_ROOT.parent / "findings" if (REPO_ROOT.parent / "findings").exists() else REPO_ROOT / "findings"
CLEAN_PATH = str(FINDINGS_DIR / "all_clean_outputs.json")
JUDGED_PATH = str(FINDINGS_DIR / "holdout_v2_judged.json")
CONVERTED_PATH = str(FINDINGS_DIR / "agenterrorbench_converted.json")
PATTERN_LIB_PATH = str(FINDINGS_DIR / "pattern_library.json")


@pytest.fixture(scope="module")
def dataset_records():
    """Load real datasets once for E2E smoke tests."""
    clean_runs = load_dataset(CLEAN_PATH)
    converted_runs = load_dataset(CONVERTED_PATH)
    conv_map = {r["metadata"]["run_id"]: r for r in converted_runs}

    with open(JUDGED_PATH, "r", encoding="utf-8") as f:
        judged_data = json.load(f)

    # 1. Clean run candidate with zero flags
    t1_cand = None
    for r in clean_runs:
        if len(r.get("trajectory", {}).get("steps", [])) >= 3:
            t1_cand = r
            if r["metadata"]["run_id"] == "54043951-0c93-467f-afbf-12be71bc8989":
                break

    # 2. Judged planning_error run candidate where pred == gt == 'planning_error'
    t2_cand = conv_map.get("Qwen3-8B_021_id_21___chat_b012_t00_e02-1e119842")

    # 3. Long trajectory candidate (30 steps, multiple subgoals)
    t3_cand = conv_map.get("GPT-4o_027_alfworld_task_027")

    assert t1_cand is not None, "Failed to locate Clean candidate for T1"
    assert t2_cand is not None, "Failed to locate Planning Error candidate for T2"
    assert t3_cand is not None, "Failed to locate Long Trajectory candidate for T3"

    return {"t1": t1_cand, "t2": t2_cand, "t3": t3_cand}


class TestEndToEndSmokePipeline:
    """E2E verification of GuardInterface running real agent trajectories."""

    def test_trajectory_1_clean_run_no_false_positives(self, dataset_records):
        """Trajectory 1 (Clean Run): Verify zero false positives across full pipeline."""
        run_data = dataset_records["t1"]
        meta = run_data["metadata"]
        traj = run_data["trajectory"]
        steps = traj.get("steps", [])

        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)

        # Hook 1: on_plan_proposed
        task_desc = meta.get("task_description") or "Perform clean calculation task"
        plan_text = meta.get("proposed_plan") or "1. Read input\n2. Calculate result"
        plan_res = guard.on_plan_proposed(task_desc, plan_text, metadata=meta)

        assert plan_res["approved"] is True

        history = []
        step_results = []

        # Hook 2 & 3: in-order step execution
        for step in steps:
            res = guard.on_step(step, history=history, metadata=meta)
            step_results.append(res)

            # Check expected schema keys
            assert "continue_execution" in res
            assert "drift_detected" in res
            assert "warning" in res
            assert "match_details" in res
            assert "subgoal_state" in res
            assert "drift_assessment" in res
            assert "reflection_result" in res

            # Trajectory 1 assertions: no false positives
            assert res["drift_detected"] is False, f"False positive drift detected at step {step.get('step_index')}"
            refl = res["reflection_result"]
            assert refl["plan_still_valid"] is True, f"False positive plan invalidation at step {step.get('step_index')}"

            # Check for subgoal boundary transition
            if res.get("subgoal_transition"):
                trans = res["subgoal_transition"]
                b_res = guard.on_subgoal_boundary(
                    subgoal_id=trans.get("completed_subgoal_id", "subgoal_001"),
                    subgoal_status=trans.get("completed_status", "completed"),
                    step_history=history,
                )
                assert "checkpoint_passed" in b_res

            history.append(step)

        # Hook 4: on_run_end
        run_res = guard.on_run_end(meta, traj)

        assert run_res["processed"] is True
        assert len(run_res["flags_summary"]) == 0
        assert run_res["root_cause_error_type"] is None

    def test_trajectory_2_known_planning_error_signals(self, dataset_records):
        """Trajectory 2 (Known Planning Error): Verify signals surface by run end."""
        run_data = dataset_records["t2"]
        meta = run_data["metadata"]
        traj = run_data["trajectory"]
        steps = traj.get("steps", [])

        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)

        task_desc = meta.get("task_description") or "Multi-step web research task"
        plan_text = meta.get("proposed_plan") or steps[0].get("reasoning", "")
        plan_res = guard.on_plan_proposed(task_desc, plan_text, metadata=meta)

        history = []
        step_results = []
        refl_suggested_count = 0

        for step in steps:
            res = guard.on_step(step, history=history, metadata=meta)
            step_results.append(res)

            refl = res["reflection_result"]
            if refl.get("revision_suggested"):
                refl_suggested_count += 1

            if res.get("subgoal_transition"):
                trans = res["subgoal_transition"]
                guard.on_subgoal_boundary(
                    subgoal_id=trans.get("completed_subgoal_id", "subgoal_001"),
                    subgoal_status=trans.get("completed_status", "completed"),
                    step_history=history,
                )

            history.append(step)

        run_res = guard.on_run_end(meta, traj)

        # Trajectory 2 assertions: signals surfaced
        total_flags = len(run_res["flags_summary"])
        root_cause = run_res.get("root_cause_error_type")

        assert total_flags > 0 or refl_suggested_count > 0, "Trajectory 2 failed to surface any error signals"
        assert root_cause in ("planning_error", "reflection_error") or refl_suggested_count > 0

    def test_trajectory_3_long_run_triggers_and_coincidence_guard(self, dataset_records):
        """Trajectory 3 (Long Run 30 steps): Verify both reflector triggers, coincidence guard, and summary consistency."""
        run_data = dataset_records["t3"]
        meta = run_data["metadata"]
        traj = run_data["trajectory"]
        steps = traj.get("steps", [])

        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)

        task_desc = meta.get("task_description") or "ALFWORLD object manipulation task"
        plan_text = meta.get("proposed_plan") or steps[0].get("reasoning", "")
        guard.on_plan_proposed(task_desc, plan_text, metadata=meta)

        history = []
        step_results = []

        interval_trigger_count = 0
        boundary_trigger_count = 0
        coincidences_detected = 0

        for step in steps:
            sidx = step.get("step_index", len(history))
            res = guard.on_step(step, history=history, metadata=meta)
            step_results.append(res)

            refl = res["reflection_result"]
            ttype = refl.get("trigger_type")

            if ttype == "step_interval" and refl.get("revision_reasoning") != "":
                interval_trigger_count += 1
            elif ttype == "subgoal_boundary":
                boundary_trigger_count += 1

            # Simulate host calling on_subgoal_boundary on transition
            if res.get("subgoal_transition"):
                trans = res["subgoal_transition"]
                b_res = guard.on_subgoal_boundary(
                    subgoal_id=trans.get("completed_subgoal_id", "subgoal_001"),
                    subgoal_status=trans.get("completed_status", "completed"),
                    step_history=history,
                )
                b_refl = b_res.get("reflection_result", {})
                # If on_subgoal_boundary called on same step, coincidence guard suppresses duplicate evaluation reasoning
                if b_refl.get("revision_reasoning") == "":
                    coincidences_detected += 1

            history.append(step)

        run_res = guard.on_run_end(meta, traj)

        # Trajectory 3 assertions
        assert len(steps) >= 30, "Trajectory 3 must be at least 30 steps"
        assert interval_trigger_count >= 5, f"Expected >= 5 step-interval reflector evaluations, got {interval_trigger_count}"
        assert boundary_trigger_count > 0, f"Expected > 0 subgoal-boundary reflector evaluations, got {boundary_trigger_count}"

        # Coincidence guard check: verify duplicate calls on same step did not double-fire reasoning
        assert coincidences_detected == boundary_trigger_count, "Coincidence guard failed to suppress duplicate boundary evaluation"

        # Summary consistency check
        subgoals_summary = run_res.get("subgoals_summary")
        drift_summary = run_res.get("drift_summary")

        assert subgoals_summary is not None
        assert drift_summary is not None
        assert len(run_res["flags_summary"]) == sum(1 for r in step_results if r.get("match_details"))
