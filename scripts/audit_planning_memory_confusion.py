"""Standalone, read-only analysis script to audit planning_error vs memory_error mismatches across provider outputs."""

import json
import os
from pathlib import Path

def load_ground_truth():
    """Load ground-truth labels from sorted.json or parsed-data/sorted.json."""
    gt_file = Path("sorted.json")
    if not gt_file.exists():
        gt_file = Path("parsed-data/sorted.json")
    if not gt_file.exists():
        gt_file = Path("findings/agenterrorbench_converted.json")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    runs_list = gt_raw.get("runs", gt_raw) if isinstance(gt_raw, dict) else gt_raw
    return {r["metadata"]["run_id"]: r for r in runs_list}


def audit_provider_files():
    """Audit planning_error vs memory_error mismatches across provider JSON output files."""
    gt_runs_map = load_ground_truth()

    provider_files = [
        ("Cloudflare (Llama 3.3 70B)", "findings/providers/cloudflare_llama33_judged.json"),
        ("Google Gemini (3.6 Flash)", "findings/providers/gemini_judged.json"),
        ("Groq (Qwen 2.5 72B/27B)", "findings/providers/groq_judged.json"),
    ]

    total_mismatches = 0

    for provider_name, filepath in provider_files:
        print("=" * 80)
        print(f"AUDIT PROVIDER: {provider_name}")
        print(f"File Path     : {filepath}")
        print("=" * 80 + "\n")

        if not os.path.exists(filepath):
            print(f"[WARNING] Provider file not found: {filepath}\n")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            judged_list = json.load(f)

        file_mismatch_count = 0

        for idx, judged_item in enumerate(judged_list, 1):
            run_id = judged_item.get("run_id")
            rec_idx = judged_item.get("record_index", idx)
            pred_cat = judged_item.get("root_cause_error_type")
            pred_step = judged_item.get("root_cause_step_index")

            gt_run = gt_runs_map.get(run_id)
            if not gt_run:
                continue

            gt_cat = gt_run["metadata"].get("root_cause_error_type")
            gt_step = gt_run["metadata"].get("root_cause_step_index")

            # Filter specifically to (planning_error, memory_error) pairs in either direction
            pair = (gt_cat, pred_cat)
            if pred_cat != gt_cat and pair in [("planning_error", "memory_error"), ("memory_error", "planning_error")]:
                file_mismatch_count += 1
                total_mismatches += 1

                # Locate trajectory step at predicted step index
                traj_steps = gt_run.get("trajectory", {}).get("steps", [])
                target_step = None
                for s in traj_steps:
                    if s.get("step_index") == pred_step:
                        target_step = s
                        break

                reasoning_text = target_step.get("reasoning") if target_step else None
                action_name = target_step.get("action_name") if target_step else "[N/A]"

                print("-" * 80)
                print(f"Record #{rec_idx:03d} | Run ID: {run_id}")
                print(f"  Ground-Truth Category : {gt_cat} (Step {gt_step})")
                print(f"  Predicted Category    : {pred_cat} (Step {pred_step})")
                print(f"  Predicted Step Index  : {pred_step}")
                print(f"  Action Name at Step   : {action_name}")
                print(f"  Reasoning / Thought Text at Step {pred_step}:")
                if reasoning_text:
                    for line in str(reasoning_text).splitlines():
                        print(f"    | {line}")
                else:
                    print("    | [Empty / None]")
                print()

        print(f"Subtotal planning/memory mismatches for {provider_name}: {file_mismatch_count}\n")

    print("=" * 80)
    print(f"SUMMARY: TOTAL PLANNING / MEMORY MISMATCHES ACROSS ALL FILES: {total_mismatches}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    audit_provider_files()
