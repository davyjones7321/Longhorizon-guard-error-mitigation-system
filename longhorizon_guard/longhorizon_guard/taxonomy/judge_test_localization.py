#!/usr/bin/env python3
"""
Automated LLM-as-Judge Tagger Test Copy (Phase 3) for longhorizon_guard.
Testing Rule 9 Localization Rule against Cloudflare (@cf/meta/llama-3.3-70b-instruct-fp8-fast).
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
   - Exception: see Rule 9 for cases where a later step had the opportunity to catch/correct the same omitted constraint and failed to — those cases defer to Rule 9's memory_error attribution instead of tool_use_error.
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
9. Root-cause step localization for constraint/requirement gaps:
   - This rule is a scoped exception to Rule 4: Rule 4's tool_use_error mandate applies only when no later step had the opportunity to catch or correct the same gap. When a later step did have that opportunity and failed to use it, this rule's step-localization and memory_error attribution take precedence over Rule 4 for that case.
   - When an early step (e.g. an initial search query or plan) omits a stated task constraint or requirement, do NOT automatically attribute the root cause to that first occurrence. Trace forward through the rest of the trajectory:
     - If the same missing constraint is never revisited, corrected, or re-checked in any later step, and the agent had no further opportunity to catch it, the first step MAY be the root cause.
     - If a later step had the information available (from the task description or the agent's own prior observations) and still failed to apply or correct the gap, attribute the root cause to that LATER step instead — this is typically memory_error (memory_retrieval_failure or over_simplification), not planning_error.
     - Do not treat 'the first action wasn't exhaustive' as itself the root cause. An imperfect first attempt is normal; the root cause is wherever the trajectory failed to notice or correct the gap despite having the chance to.
   - The test: find the LATEST step where the same underlying gap could still have been caught and fixed. Treat that step as the root cause unless the trajectory truly never had another opportunity to fix it.

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


def _truncate_field(text: Optional[str], limit: int) -> Optional[str]:
    """Truncate text to limit if needed to fit within Cloudflare's 100KB payload limit."""
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "..."


def _prepare_trajectory_payload(run: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare payload of a trajectory for the LLM judge within Cloudflare payload limits."""
    meta = run.get("metadata", {}) or {}
    traj = run.get("trajectory", {}) or {}
    steps = traj.get("steps", [])

    formatted_steps = []
    for s in steps:
        raw_args = s.get("action_args", {})
        clean_args = {}
        if isinstance(raw_args, dict):
            for k, v in raw_args.items():
                clean_args[k] = _truncate_field(str(v), 500) if v is not None else None
        else:
            clean_args = {"args": _truncate_field(str(raw_args), 500)}

        formatted_steps.append({
            "step_index": s.get("step_index"),
            "reasoning": _truncate_field(s.get("reasoning") or "", 800),
            "action_name": s.get("action_name") or "none",
            "action_args": clean_args,
            "tool_response": _truncate_field(s.get("tool_response"), 800) if s.get("tool_response") else None,
        })

    payload = {
        "run_id": meta.get("run_id"),
        "task_id": meta.get("task_id"),
        "task_description": _truncate_field(meta.get("task_description") or meta.get("task_id") or "", 500),
        "final_status": meta.get("final_status"),
        "grader_notes": _truncate_field(meta.get("grader_notes") or "", 500),
        "total_steps": meta.get("total_steps_taken", len(steps)),
        "steps": formatted_steps,
    }
    return payload


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


async def _call_cloudflare_ai(prompt: str, cf_key: str, cf_account_id: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    req_data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {cf_key}", "Content-Type": "application/json"}

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=60)
            body = json.loads(resp.read().decode("utf-8"))
            result = body.get("result", {})
            choices = result.get("choices", [])
            text_content = ""
            if choices:
                text_content = choices[0]["message"]["content"]
            elif "response" in result:
                text_content = result["response"]

            if text_content:
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            print(f"Cloudflare AI attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            await asyncio.sleep(3.0)
    return None


async def call_gemini_judge(payload: Dict[str, Any], semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
    """Invoke LLM judge via Cloudflare Workers AI ONLY (@cf/meta/llama-3.3-70b-instruct-fp8-fast)."""
    cf_key = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_KEY")
    cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

    if not cf_key or not cf_account_id:
        print("ERROR: CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID not found in environment", file=sys.stderr)
        return None

    prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nAnalyze this agent trajectory:\n{json.dumps(payload, indent=2)}\n\nOutput JSON:"

    async with semaphore:
        res = await _call_cloudflare_ai(prompt, cf_key, cf_account_id)
        if res and isinstance(res, dict) and "root_cause_error_type" in res:
            res["provider_used"] = "Cloudflare Workers AI (@cf/meta/llama-3.3-70b-instruct-fp8-fast)"
            return res

    return None


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

        total += 1
        per_cat_total[true_cat] += 1
        confusion_matrix[true_cat][pred_cat] += 1

        cat_match = (pred_cat == true_cat)
        if cat_match:
            cat_correct += 1
            per_cat_correct[true_cat] += 1

        step_exact = (pred_step == true_step)
        if step_exact:
            step_exact_correct += 1

        step_w1 = (pred_step is not None and true_step is not None and abs(pred_step - true_step) <= 1)
        if step_w1:
            step_within1_correct += 1

        if not cat_match:
            disagreements.append({
                "run_id": run_id,
                "true_category": true_cat,
                "pred_category": pred_cat,
                "true_step": true_step,
                "pred_step": pred_step,
                "justification": judged.get("root_cause_justification", ""),
                "grader_notes": meta.get("grader_notes", ""),
                "original_subtype": meta.get("original_subtype", "N/A"),
            })

    per_cat_metrics = {}
    for cat in DEFAULT_TAGS:
        t_cnt = per_cat_total[cat]
        c_cnt = per_cat_correct[cat]
        per_cat_metrics[cat] = {
            "total": t_cnt,
            "correct": c_cnt,
            "match_rate": (c_cnt / t_cnt) if t_cnt > 0 else 0.0,
        }

    return {
        "total_evaluated": total,
        "category_match_rate": (cat_correct / total) if total > 0 else 0.0,
        "exact_step_match_rate": (step_exact_correct / total) if total > 0 else 0.0,
        "step_within1_match_rate": (step_within1_correct / total) if total > 0 else 0.0,
        "per_category": per_cat_metrics,
        "confusion_matrix": confusion_matrix,
        "disagreements": disagreements,
    }


def _load_tuning_case_ids() -> Set[str]:
    """Load tuning case run_ids from findings/tuning_case_run_ids.txt."""
    path = Path("findings/tuning_case_run_ids.txt")
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


async def main_async(args: argparse.Namespace) -> None:
    print(f"Loading dataset from {args.data_dir}...")
    runs = load_dataset(args.data_dir)

    # Filter to confident records only
    if args.confident_only:
        confident_runs = [r for r in runs if r["metadata"].get("tag_confidence") == 1.0]
    else:
        confident_runs = runs

    print(f"Total dataset records: {len(runs)}")
    print(f"Confident evaluation records (tag_confidence == 1.0): {len(confident_runs)}")

    # --- Holdout split ---
    eval_runs = confident_runs
    if args.holdout_only:
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

    if getattr(args, "sort_by_length", False):
        eval_runs.sort(key=lambda r: len(json.dumps(_prepare_trajectory_payload(r))))
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

    # Rate-limited execution
    semaphore = asyncio.Semaphore(1)
    pending_runs = [r for r in eval_runs if r["metadata"]["run_id"] not in judged_map]
    print(f"Pending runs to judge: {len(pending_runs)}/{len(eval_runs)}")

    for idx, r in enumerate(pending_runs, 1):
        run_id = r["metadata"]["run_id"]
        payload = _prepare_trajectory_payload(r)

        print(f"[{idx}/{len(pending_runs)}] Judging run {run_id} ({payload.get('task_id')})...", end="", flush=True)
        res = await call_gemini_judge(payload, semaphore)

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

        # Rate limiting sleep between requests
        if idx < len(pending_runs):
            await asyncio.sleep(4.0)

    # Calculate final metrics
    metrics = calculate_metrics(eval_runs, judged_map)

    print("\n" + "=" * 60)
    print("=== SUMMARY METRICS ===")
    print(f"Total Evaluated Records     : {metrics['total_evaluated']}")
    print(f"Category Match Accuracy     : {metrics['category_match_rate']*100:.2f}%")
    print(f"Exact Step Match Accuracy   : {metrics['exact_step_match_rate']*100:.2f}%")
    print(f"Step Within-1 Match Accuracy: {metrics['step_within1_match_rate']*100:.2f}%")

    print("\n--- Per-Category Accuracy ---")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 LLM-as-judge tagger calibration test script")
    parser.add_argument("--data-dir", default="findings/agenterrorbench_converted.json", help="Input converted dataset")
    parser.add_argument("--output", default="findings/test_localization_judged.json", help="Output judgment file")
    parser.add_argument("--confident-only", action="store_true", default=True, help="Evaluate on tag_confidence == 1.0 only")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for holdout sampling (default: 42)")
    parser.add_argument("--holdout-size", type=int, default=30, help="Number of records to sample for holdout (default: 30)")
    parser.add_argument("--holdout-only", action="store_true", default=False,
                        help="Run judge on a holdout sample only (not the full dataset)")
    parser.add_argument("--exclude-tuning-cases", action="store_true", default=True,
                        help="Exclude run_ids listed in findings/tuning_case_run_ids.txt from holdout pool")
    parser.add_argument("--sort-by-length", action="store_true", default=False,
                        help="Sort evaluation records by character length (ascending: shortest -> longest)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
