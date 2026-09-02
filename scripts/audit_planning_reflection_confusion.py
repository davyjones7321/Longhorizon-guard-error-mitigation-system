"""Standalone, read-only analysis script to audit planning_error vs reflection_error mismatches."""

import json
import os
from pathlib import Path

def load_ground_truth():
    """Load ground-truth labels from findings/agenterrorbench_converted.json or sorted.json."""
    gt_file = Path("findings/agenterrorbench_converted.json")
    if not gt_file.exists():
        gt_file = Path("sorted.json")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    runs_list = gt_raw.get("runs", gt_raw) if isinstance(gt_raw, dict) else gt_raw
    return {r["metadata"]["run_id"]: r for r in runs_list}


def audit_planning_reflection_confusion():
    """Audit planning_error -> reflection_error mismatches from the holdout run output."""
    gt_runs_map = load_ground_truth()

    output_file = Path("findings/test_localization_judged.json")
    if not output_file.exists():
        output_file = Path("findings/holdout_v2_judged.json")

    if not output_file.exists():
        print(f"[ERROR] Target judged output file not found: {output_file}")
        return

    print("=" * 80)
    print(f"AUDITING PLANNING vs REFLECTION CONFUSION")
    print(f"Target Output File : {output_file}")
    print(f"Ground-Truth File  : findings/agenterrorbench_converted.json")
    print("=" * 80 + "\n")

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    judged_map = data.get("judgments", {})

    target_cases = []

    for run_id, judged in judged_map.items():
        gt_run = gt_runs_map.get(run_id)
        if not gt_run:
            continue

        gt_cat = gt_run["metadata"].get("root_cause_error_type")
        gt_step = gt_run["metadata"].get("root_cause_step_index")

        pred_cat = judged.get("root_cause_error_type")
        pred_step = judged.get("root_cause_step_index")
        justification = judged.get("root_cause_justification", "")

        if gt_cat == "planning_error" and pred_cat == "reflection_error":
            target_cases.append({
                "run_id": run_id,
                "gt_cat": gt_cat,
                "gt_step": gt_step,
                "pred_cat": pred_cat,
                "pred_step": pred_step,
                "justification": justification,
                "gt_run": gt_run
            })

    for idx, case in enumerate(target_cases, 1):
        run_id = case["run_id"]
        gt_step = case["gt_step"]
        pred_step = case["pred_step"]
        gt_run = case["gt_run"]
        traj_steps = gt_run.get("trajectory", {}).get("steps", [])

        gt_step_obj = next((s for s in traj_steps if s.get("step_index") == gt_step), None)
        pred_step_obj = next((s for s in traj_steps if s.get("step_index") == pred_step), None)

        print("-" * 80)
        print(f"CASE {idx} of {len(target_cases)} | Run ID: {run_id}")
        print(f"  Ground-Truth Category : {case['gt_cat']} @ Step {gt_step}")
        print(f"  Predicted Category    : {case['pred_cat']} @ Step {pred_step}")
        print(f"  Model Justification   : {case['justification']}")
        print("-" * 80)

        # Ground Truth Step Content
        print(f"--- GROUND-TRUTH STEP CONTENT (Step {gt_step}) ---")
        if gt_step_obj:
            print(f"  Action Name: {gt_step_obj.get('action_name')}")
            print("  Reasoning / Thought Text:")
            reasoning = gt_step_obj.get("reasoning")
            if reasoning:
                for l in str(reasoning).splitlines():
                    print(f"    | {l}")
            else:
                print("    | [Empty / None]")
        else:
            print("  [Step Not Found in Trajectory]")

        print()

        # Predicted Step Content
        print(f"--- PREDICTED STEP CONTENT (Step {pred_step}) ---")
        if pred_step_obj:
            print(f"  Action Name: {pred_step_obj.get('action_name')}")
            print("  Reasoning / Thought Text:")
            reasoning = pred_step_obj.get("reasoning")
            if reasoning:
                for l in str(reasoning).splitlines():
                    print(f"    | {l}")
            else:
                print("    | [Empty / None]")
        else:
            print("  [Step Not Found in Trajectory]")

        print("\n" + "=" * 80 + "\n")

    print(f"SUMMARY: TOTAL PLANNING -> REFLECTION MISMATCHES FOUND: {len(target_cases)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    audit_planning_reflection_confusion()
