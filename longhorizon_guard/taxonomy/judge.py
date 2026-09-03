#!/usr/bin/env python3
"""
Automated multi-provider LLM-as-Judge tagger (Phase 3) for longhorizon_guard.

Requires explicit --provider selection to analyze trajectory logs, identify per-step
error tags, and determine the earliest root-cause failure step (root_cause_step_index)
and root-cause category (root_cause_error_type) according to the 7-category taxonomy.

Usage:
    python -m longhorizon_guard.taxonomy.judge \
        --provider gemini \
        --data-dir findings/agenterrorbench_converted.json \
        --output findings/agenterrorbench_judged.json \
        --confident-only
"""

import argparse
import asyncio
import json
import os
import random as _random_module
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("longhorizon_guard"))

from longhorizon_guard.taxonomy.categories import DEFAULT_TAGS
from longhorizon_guard.storage.reader import load_dataset

JUDGE_SYSTEM_PROMPT = """You are an expert AI agent failure auditor specializing in root-cause error analysis for multi-step agent trajectories.

Your task is to analyze an agent's trajectory for a failed task and perform two evaluations:
1. Identify per-step errors (if any) across the trajectory.
2. Identify the single EARLIEST root-cause failure step (root_cause_step_index) and assign it one of the 7 standardized error categories.

### Standardized Error Taxonomy (7 Categories)

1. planning_error:
   - Definition: The agent formulated an incorrect overall strategy, sequence, or approach.
   - EXCEPTION OVERRIDE: An incomplete initial plan (e.g. omitting a sub-goal like 'clean') is NOT planning_error if the agent later had an opportunity to observe the environment and verify state. In such cases, tag reflection_error at the observation step, NOT planning_error at Step 0.
2. reflection_error:
   - Definition: The agent failed to evaluate tool output, misjudged environment state, misinterpreted task completion, or failed to recognize an incomplete/incorrect outcome after observing environment feedback.
3. tool_use_error:
   - Definition: The agent selected an invalid tool, formatted arguments incorrectly, passed invalid parameters, or omitted required parameters/constraints in a tool call.
4. memory_error:
   - Definition: The agent forgot previously observed information, duplicated actions in a loop, or lost context over long horizons.
5. external_error:
   - Definition: Failures caused by system limits, environment cutoffs, network timeouts, or tool execution errors outside agent control.
6. grader_error:
   - Definition: The agent succeeded according to task requirements, but the automated evaluator incorrectly graded it as a failure.
7. other:
   - Definition: Ambiguous failure or edge-case error not cleanly covered by the above 6 categories.

### Root-Cause Disambiguation & Primacy Rules
1. Mechanical Step Limit Override (HIGHEST PRIORITY):
   - Check final_status and grader_notes FIRST.
   - ONLY if grader_notes or final_status unambiguously indicates that an environment- or system-imposed step/turn limit was reached before task completion (e.g., 'step limit reached', 'max steps cap', 'max turns hit', 'timeout limit'), you MUST tag root_cause_error_type as external_error.
   - Do NOT tag external_error merely because words like 'limit' or 'truncated' appear in text if an actual turn/step cutoff did not cause the termination.
   - Do NOT evaluate whether the agent's behavior during those steps seems inefficient, looped, or planning-flawed — the mechanical termination mechanism determines this tag. This check takes priority over all other rules in this prompt.
2. Reflection Error Precedence over Initial Plan Text (CRITICAL):
   - Do NOT tag planning_error at Step 0 merely because the agent's initial written plan or reasoning block omitted a required sub-step (e.g., 'clean').
   - An initial plan omission is ONLY a planning_error if the agent NEVER received any subsequent observation or feedback before declaring the task done.
   - If the agent later observed environment state (e.g. picked up, inspected, or placed an object, or received product options) and had feedback available at that point to recognize the task was incomplete but failed to do so, you MUST tag root_cause_error_type as reflection_error at that later observation step — REGARDLESS of what the initial plan stated or omitted at Step 0.
3. Environment vs. Agent Causality:
   - If Step K's tool execution returned erroneous, corrupted, or misleading output despite a valid agent request, mark STEP K as the root cause (external_error).
   - If Step K's tool output contained valid choices and the agent picked the wrong one at Step K+1, mark STEP K+1 as the root cause (tool_use_error or planning_error).
4. Action Parameter Omission vs. Planning (Pattern A - ABSOLUTE PRIORITY):
   - You MUST tag tool_use_error at the action step whenever a search query or action parameter omits a required term, includes a conflicting term (e.g. 'men' and 'women'), or uses improper search keywords.
   - Do NOT tag planning_error for search query formulation or parameter wording errors under ANY circumstances — even if the search query terms appear inside an initial reasoning, thinking, or plan block at step 0.
   - Reserve planning_error strictly for cases where the agent selected the wrong high-level action category (e.g. clicking 'buy' before searching), NOT for search query terms or parameter values.
5. Earliest Point of Failure (Primacy):
   - Always assign root_cause_step_index to the EARLIEST step where the trajectory deviated from a valid path to task completion.
   - Do NOT mark downstream cascading errors as the root cause.
6. Grader Error Constraint:
   - You MAY ONLY assign grader_error if there is explicit evidence in grader_notes or trajectory output showing that the agent's final answer was objectively correct according to the task description, but was erroneously marked as a failure.
   - If grader_notes or ground-truth details are absent or unverified, NEVER assign grader_error; assign other or the execution error instead.
7. Precise Planning Error Step Indexing Rule:
   - For planning_error, tag the exact step index where the flawed plan, invalid search query, or inadmissible action was generated or executed by the agent (e.g. Step 3, Step 6, Step 9, Step 10).
   - Do NOT default planning_error to Step 0 unless the initial prompt plan itself was the sole root cause.
8. Mandatory Reflection Error Override on Mid-Trajectory Observations:
   - If the trajectory contains any intermediate step where the agent observed environment output (e.g. inspected an object, searched a location, or received search results) and failed to update its goal or recognize an incomplete state, tag reflection_error at that observation step.
   - You MUST tag reflection_error even if the initial plan at Step 0 was incomplete or flawed.

### Required Output JSON Format
You MUST respond with valid JSON matching this exact structure:
{
  "root_cause_step_index": 7,
  "root_cause_error_type": "planning_error",
  "confidence": 0.95,
  "root_cause_justification": "Brief 1-2 sentence explanation of why this step was the root cause.",
  "step_annotations": [
    {
      "step_index": 7,
      "error_tag": "planning_error",
      "justification": "Brief justification for step tag"
    }
  ]
}
"""


def _truncate_field(text, max_len: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...[TRUNCATED, {len(text)} chars total]"


def _prepare_trajectory_payload(run: Dict[str, Any], full_trajectory: bool = False) -> Dict[str, Any]:
    meta = run.get("metadata", {}) or {}
    traj = run.get("trajectory", {}) or {}
    steps = traj.get("steps", [])

    n = len(steps)
    if full_trajectory or n <= 6:
        selected_indices = list(range(n))
    else:
        selected = set(range(3)) | set(range(n - 3, n))
        signal_keywords = ("error", "fail", "cannot", "invalid", "not found", "unsuccessful")
        for i, s in enumerate(steps):
            blob = f"{s.get('reasoning','')} {s.get('tool_response','')}".lower()
            if any(kw in blob for kw in signal_keywords):
                selected.add(i)
        selected_indices = sorted(selected)

    formatted_steps = []
    for i in selected_indices:
        s = steps[i]
        raw_args = s.get("action_args", {})
        clean_args = {}
        if isinstance(raw_args, dict):
            for k, v in raw_args.items():
                clean_args[k] = str(v) if (full_trajectory or v is None) else _truncate_field(v, 150)
        else:
            clean_args = {"args": str(raw_args) if full_trajectory else _truncate_field(raw_args, 150)}

        reasoning_val = str(s.get("reasoning") or "") if full_trajectory else _truncate_field(s.get("reasoning") or "", 500)
        tool_resp_val = str(s.get("tool_response") or "") if full_trajectory else _truncate_field(s.get("tool_response"), 500)

        formatted_steps.append({
            "step_index": s.get("step_index"),
            "reasoning": reasoning_val,
            "action_name": s.get("action_name") or "none",
            "action_args": clean_args,
            "tool_response": tool_resp_val,
        })

    desc_val = str(meta.get("task_description") or meta.get("task_id") or "") if full_trajectory else _truncate_field(meta.get("task_description") or meta.get("task_id") or "", 300)
    notes_val = str(meta.get("grader_notes") or "") if full_trajectory else _truncate_field(meta.get("grader_notes") or "", 300)

    return {
        "run_id": meta.get("run_id"),
        "task_id": meta.get("task_id"),
        "task_description": desc_val,
        "final_status": meta.get("final_status"),
        "grader_notes": notes_val,
        "total_steps": meta.get("total_steps_taken", len(steps)),
        "steps": formatted_steps,
    }


def _load_dotenv_if_needed() -> None:
    """Load API keys from .env file into os.environ if present."""
    from pathlib import Path
    for env_file in [Path(".env"), Path("eval") / ".env"]:
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

_load_dotenv_if_needed()

from longhorizon_guard.engines.llm_provider import call_llm, PROVIDER_DETAILS, _get_provider_credentials

def calculate_metrics(confident_runs: List[Dict[str, Any]], judged_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate exact match rate, step match rate, and confusion matrix."""
    total = 0
    cat_correct = 0
    step_exact_correct = 0
    step_within1_correct = 0

    per_cat_total = Counter()
    per_cat_correct = Counter()
    confusion_matrix = defaultdict(Counter)
    disagreements = []

    for r in confident_runs:
        meta = r["metadata"]
        run_id = meta["run_id"]
        true_cat = meta["root_cause_error_type"]
        true_step = meta["root_cause_step_index"]

        judged = judged_map.get(run_id)
        if not judged:
            continue

        pred_cat = judged.get("root_cause_error_type")
        pred_step = judged.get("root_cause_step_index")

        # Normalize predicted category
        if pred_cat not in DEFAULT_TAGS:
            pred_cat = "other"

        total += 1
        per_cat_total[true_cat] += 1
        confusion_matrix[true_cat][pred_cat] += 1

        is_cat_match = (pred_cat == true_cat)
        if is_cat_match:
            cat_correct += 1
            per_cat_correct[true_cat] += 1
        else:
            disagreements.append({
                "run_id": run_id,
                "task_id": meta.get("task_id"),
                "true_category": true_cat,
                "pred_category": pred_cat,
                "true_step": true_step,
                "pred_step": pred_step,
                "justification": judged.get("root_cause_justification"),
                "original_subtype": meta.get("original_subtype"),
                "grader_notes": meta.get("grader_notes"),
            })

        if true_step is not None and pred_step is not None:
            if pred_step == true_step:
                step_exact_correct += 1
                step_within1_correct += 1
            elif abs(pred_step - true_step) <= 1:
                step_within1_correct += 1

    category_accuracy = (cat_correct / total) if total else 0.0
    step_exact_accuracy = (step_exact_correct / total) if total else 0.0
    step_within1_accuracy = (step_within1_correct / total) if total else 0.0

    per_cat_match_rates = {}
    for cat in DEFAULT_TAGS:
        c_tot = per_cat_total[cat]
        c_cor = per_cat_correct[cat]
        per_cat_match_rates[cat] = {
            "total": c_tot,
            "correct": c_cor,
            "match_rate": (c_cor / c_tot) if c_tot else 0.0
        }

    return {
        "total_evaluated": total,
        "overall_category_accuracy": category_accuracy,
        "step_exact_accuracy": step_exact_accuracy,
        "step_within1_accuracy": step_within1_accuracy,
        "per_category": per_cat_match_rates,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
        "disagreements": disagreements,
    }


def _load_tuning_case_ids(path: str = "findings/tuning_case_run_ids.txt") -> Set[str]:
    """Load tuning case run_ids from a text file (one run_id per line).

    Lines starting with '#' or empty lines are ignored.
    Returns empty set if the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return set()
    ids: Set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


async def main_async(args: argparse.Namespace) -> None:
    provider_details = PROVIDER_DETAILS[args.provider]
    _get_provider_credentials(args.provider)
    print("=" * 60)
    print(f"Selected provider: {provider_details['label']} ({args.provider})")
    print(f"Selected model   : {provider_details['model']}")
    print("=" * 60)
    print(f"Loading dataset from {args.data_dir}...")
    runs = load_dataset(args.data_dir)

    # Filter to confident records only
    if args.confident_only:
        confident_runs = [r for r in runs if r["metadata"].get("tag_confidence") == 1.0]
    else:
        confident_runs = runs

    print(f"Total dataset records: {len(runs)}")
    print(f"Confident evaluation records (tag_confidence == 1.0): {len(confident_runs)}")

    # --- Holdout split or Specific Run IDs file ---
    eval_runs = confident_runs
    if getattr(args, "run_ids_file", None):
        target_path = Path(args.run_ids_file)
        if not target_path.exists():
            print(f"ERROR: --run-ids-file '{args.run_ids_file}' does not exist.", file=sys.stderr)
            sys.exit(1)

        requested_ids = []
        for line in target_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                requested_ids.append(line)

        dataset_by_id = {r["metadata"]["run_id"]: r for r in eval_runs}
        filtered_runs = []
        for rid in requested_ids:
            if rid in dataset_by_id:
                filtered_runs.append(dataset_by_id[rid])
            else:
                print(f"WARNING: run_id '{rid}' from {args.run_ids_file} not found in dataset. Skipping.")

        eval_runs = filtered_runs
        print(f"\n=== SPECIFIED RUN IDs FILE ({args.run_ids_file}, n={len(eval_runs)}) ===")
        for i, r in enumerate(eval_runs, 1):
            rid = r["metadata"]["run_id"]
            gt_cat = r["metadata"].get("root_cause_error_type", "?")
            gt_step = r["metadata"].get("root_cause_step_index", "?")
            print(f"  [{i:3d}] {rid}  (GT: {gt_cat} @ step {gt_step})")
        print("=" * 60 + "\n")
    elif args.holdout_only:
        # Exclude tuning cases to prevent data leakage
        if args.exclude_tuning_cases:
            tuning_ids = _load_tuning_case_ids()
            before = len(eval_runs)
            eval_runs = [r for r in eval_runs if r["metadata"]["run_id"] not in tuning_ids]
            excluded = before - len(eval_runs)
            print(f"Excluded {excluded} tuning cases ({len(tuning_ids)} IDs in tuning_case_run_ids.txt)")

        # Deterministic sample using a seeded RNG (independent of global state)
        rng = _random_module.Random(args.seed)
        if args.holdout_size >= len(eval_runs):
            print(f"Holdout size ({args.holdout_size}) >= available records ({len(eval_runs)}), using all")
        else:
            eval_runs = rng.sample(eval_runs, args.holdout_size)

        print(f"\n=== HOLDOUT SAMPLE (seed={args.seed}, n={len(eval_runs)}) ===")
        for i, r in enumerate(eval_runs, 1):
            rid = r["metadata"]["run_id"]
            gt_cat = r["metadata"].get("root_cause_error_type", "?")
            gt_step = r["metadata"].get("root_cause_step_index", "?")
            print(f"  [{i:3d}] {rid}  (GT: {gt_cat} @ step {gt_step})")
        print("=" * 60 + "\n")
    else:
        print("Running full evaluation (no holdout split).")

    full_traj = getattr(args, "full_trajectory", False)
    if full_traj:
        print("Full trajectory mode enabled (no step slicing, no text truncation).")

    if getattr(args, "sort_by_length", False):
        eval_runs.sort(key=lambda r: len(json.dumps(_prepare_trajectory_payload(r, full_trajectory=full_traj))))
        print(f"Sorted {len(eval_runs)} evaluation records by character length (ascending: shortest -> longest).")

    # Load cached judgments if present
    judged_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                judged_map = existing_data.get("judgments", {})
            print(f"Loaded {len(judged_map)} existing cached judgments from {args.output}")
        except Exception as e:
            print(f"Could not load existing output: {e}")

    # Rate-limited execution (stay strictly under 15 RPM limit -> ~4.2s per request)
    semaphore = asyncio.Semaphore(1)
    pending_runs = [r for r in eval_runs if r["metadata"]["run_id"] not in judged_map]
    print(f"Pending runs to judge: {len(pending_runs)}/{len(eval_runs)}")

    for idx, r in enumerate(pending_runs, 1):
        run_id = r["metadata"]["run_id"]
        payload = _prepare_trajectory_payload(r, full_trajectory=full_traj)
        
        print(f"[{idx}/{len(pending_runs)}] Judging run {run_id} ({payload.get('task_id')})...", end="", flush=True)
        prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nAnalyze this agent trajectory:\n{json.dumps(payload, indent=2)}\n\nOutput JSON:"
        res = await call_llm(prompt, args.provider, semaphore)
        
        if res:
            judged_map[run_id] = res
            print(f" Done. (Root cause: {res.get('root_cause_error_type')} @ Step {res.get('root_cause_step_index')})")
        else:
            print(" Failed.")

        # Save checkpoint periodically
        if idx % 5 == 0 or idx == len(pending_runs):
            metrics = calculate_metrics(eval_runs, judged_map)
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({
                    "metrics": metrics,
                    "judgments": judged_map
                }, f, indent=2, ensure_ascii=False)

        # Enforce rate limit delay between requests (~4.2s = 14 RPM)
        if idx < len(pending_runs):
            await asyncio.sleep(4.2)

    # Calculate final metrics
    metrics = calculate_metrics(eval_runs, judged_map)

    # Output final results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "judgments": judged_map
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("           LLM-AS-JUDGE PHASE 3 CALIBRATION RESULTS           ")
    print("=" * 60)
    print(f"Total Evaluated                 : {metrics['total_evaluated']}")
    print(f"Overall Category Exact Match Rate: {metrics['overall_category_accuracy'] * 100:.2f}%")
    print(f"Root-Cause Step Exact Match Rate: {metrics['step_exact_accuracy'] * 100:.2f}%")
    print(f"Root-Cause Step Within-1 Match  : {metrics['step_within1_accuracy'] * 100:.2f}%")

    print("\n--- Per-Category Match Rates ---")
    for cat, data in metrics["per_category"].items():
        print(f"  {cat:20s}: {data['match_rate']*100:6.2f}% ({data['correct']}/{data['total']})")

    print("\n--- Confusion Matrix (True Row -> Predicted Col) ---")
    header = f"{'True \\ Pred':20s} " + " ".join([f"{c[:8]:>8s}" for c in DEFAULT_TAGS])
    print(header)
    print("-" * len(header))
    for true_cat in DEFAULT_TAGS:
        row_str = f"{true_cat:20s} "
        for pred_cat in DEFAULT_TAGS:
            count = metrics["confusion_matrix"].get(true_cat, {}).get(pred_cat, 0)
            row_str += f"{count:8d} "
        print(row_str)

    # Find worst performing category
    cats_with_data = [c for c in DEFAULT_TAGS if metrics["per_category"][c]["total"] > 0]
    if cats_with_data:
        worst_cat = min(cats_with_data, key=lambda c: metrics["per_category"][c]["match_rate"])
        worst_rate = metrics["per_category"][worst_cat]["match_rate"] * 100

        print(f"\n--- Worst Performing Category: {worst_cat} ({worst_rate:.2f}% match) ---")
        worst_disagreements = [d for d in metrics["disagreements"] if d["true_category"] == worst_cat]
        print(f"Found {len(worst_disagreements)} disagreement examples for '{worst_cat}':")

        for idx, d in enumerate(worst_disagreements[:5], 1):
            print(f"\nExample {idx}:")
            print(f"  Run ID       : {d['run_id']}")
            print(f"  True Tag     : {d['true_category']} (original subtype: {d['original_subtype']})")
            print(f"  Predicted Tag: {d['pred_category']}")
            print(f"  True Step    : {d['true_step']} | Pred Step: {d['pred_step']}")
            print(f"  Judge Justification: {d['justification']}")
            print(f"  Grader Notes : {d['grader_notes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 LLM-as-judge tagger calibration")
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["cloudflare", "nvidia", "groq", "gemini", "openrouter", "tokenrouter"],
        help="Explicit provider used for every judge call.",
    )
    parser.add_argument("--data-dir", default="findings/agenterrorbench_converted.json", help="Input converted dataset")
    parser.add_argument("--output", default="findings/agenterrorbench_judged.json", help="Output judgment file")
    parser.add_argument("--confident-only", action="store_true", default=True, help="Evaluate on tag_confidence == 1.0 only")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for holdout sampling (default: 42)")
    parser.add_argument("--holdout-size", type=int, default=30, help="Number of records to sample for holdout (default: 30)")
    parser.add_argument("--holdout-only", action="store_true", default=False,
                        help="Run judge on a holdout sample only (not the full dataset)")
    parser.add_argument("--exclude-tuning-cases", action="store_true", default=True,
                        help="Exclude run_ids listed in findings/tuning_case_run_ids.txt from holdout pool")
    parser.add_argument("--sort-by-length", action="store_true", default=False,
                        help="Sort evaluation records by character length (ascending: shortest -> longest)")
    parser.add_argument(
        "--run-ids-file",
        type=str,
        default=None,
        help="Path to a text file containing run_ids to evaluate in exact order (overrides --holdout-only/--seed sampling)",
    )
    parser.add_argument(
        "--full-trajectory",
        action="store_true",
        default=False,
        help="Send full uncompressed trajectory to the judge (no step slicing, no text truncation)",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

