"""Propagation analysis module — local, read-only analysis over judged data.

No LLM API calls are made anywhere in this file. Everything here operates
on data already on disk (judged output JSONs + the converted dataset).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# PART 1 — Multi-file loader
# ---------------------------------------------------------------------------

# Stopwords that sometimes appear as action_name due to parser artifacts
# (e.g. Llama 3.3 freeform text parsed as structured tool calls).
_STOPWORD_ACTIONS = {"the", "i", "no", "to", "in", "at", "a", "an", "of", "is"}


def _load_converted_dataset(
    path: str = "findings/agenterrorbench_converted.json",
) -> Dict[str, Dict[str, Any]]:
    """Load the converted dataset and index it by run_id for O(1) lookups."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data if isinstance(data, list) else data.get("runs", [])
    return {r["metadata"]["run_id"]: r for r in runs}


def load_judged_records(
    paths: List[str],
    converted_path: str = "findings/agenterrorbench_converted.json",
) -> List[Dict[str, Any]]:
    """Load and merge judged-output JSONs, attaching ground-truth metadata.

    Each input file must have the standard format: {"judgments": {run_id: {...}}}.
    Duplicates across files are resolved by keeping the FIRST occurrence.
    Ground-truth fields are attached from the converted dataset so downstream
    analysis can compare predictions against labels.

    Also flags records where the ground-truth root-cause step has a stopword
    action_name (suspected_corrupted_action = True/False).
    """
    dataset_by_id = _load_converted_dataset(converted_path)

    merged: Dict[str, Dict[str, Any]] = {}
    for filepath in paths:
        if not os.path.exists(filepath):
            print(f"WARNING: File '{filepath}' does not exist. Skipping.", file=sys.stderr)
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        judgments = data.get("judgments", {})
        for run_id, judgment in judgments.items():
            if run_id in merged:
                print(
                    f"WARNING: Duplicate run_id '{run_id}' found in '{filepath}' — "
                    f"skipping (already loaded from an earlier file).",
                    file=sys.stderr,
                )
                continue
            merged[run_id] = dict(judgment)
            merged[run_id]["run_id"] = run_id

    # Cross-reference ground-truth metadata from converted dataset
    records = []
    for run_id, judgment in merged.items():
        gt_run = dataset_by_id.get(run_id)
        if gt_run:
            meta = gt_run.get("metadata", {})
            judgment["gt_root_cause_error_type"] = meta.get("root_cause_error_type")
            judgment["gt_root_cause_step_index"] = meta.get("root_cause_step_index")
            judgment["benchmark"] = meta.get("task_id", "unknown")
            judgment["total_steps"] = meta.get("total_steps_taken") or len(
                gt_run.get("trajectory", {}).get("steps", [])
            )

            # Flag if the GT root-cause step has a stopword action_name
            gt_step_idx = judgment["gt_root_cause_step_index"]
            steps = gt_run.get("trajectory", {}).get("steps", [])
            corrupted = False
            if gt_step_idx is not None:
                for s in steps:
                    if s.get("step_index") == gt_step_idx:
                        action = (s.get("action_name") or "").strip().lower()
                        if action in _STOPWORD_ACTIONS:
                            corrupted = True
                        break
            judgment["suspected_corrupted_action"] = corrupted
        else:
            judgment["gt_root_cause_error_type"] = None
            judgment["gt_root_cause_step_index"] = None
            judgment["benchmark"] = "unknown"
            judgment["total_steps"] = 0
            judgment["suspected_corrupted_action"] = False
            print(
                f"WARNING: run_id '{run_id}' not found in converted dataset — "
                f"ground-truth fields will be null.",
                file=sys.stderr,
            )
        records.append(judgment)

    return records


# ---------------------------------------------------------------------------
# PART 2 — Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute category distribution, step-position buckets, and accuracy.

    All analysis is local — no API calls.
    """
    # --- Category distribution ---
    overall_cat_counts: Counter = Counter()
    per_benchmark_cat_counts: Dict[str, Counter] = defaultdict(Counter)

    for r in records:
        pred_cat = r.get("root_cause_error_type", "unknown")
        benchmark = r.get("benchmark", "unknown")
        overall_cat_counts[pred_cat] += 1
        per_benchmark_cat_counts[benchmark][pred_cat] += 1

    # --- Step-position distribution ---
    # Bucket root_cause_step_index / total_steps into early/mid/late
    position_by_category: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        pred_cat = r.get("root_cause_error_type", "unknown")
        step_idx = r.get("root_cause_step_index")
        total = r.get("total_steps", 0)
        if step_idx is not None and total and total > 0:
            ratio = step_idx / total
            if ratio < 0.33:
                bucket = "early"
            elif ratio <= 0.66:
                bucket = "mid"
            else:
                bucket = "late"
            position_by_category[pred_cat][bucket] += 1

    # --- Accuracy-by-category (computed twice: all records, then clean only) ---
    def _compute_accuracy(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return accuracy_by_category + overall_accuracy for a record subset."""
        cat_total: Counter = Counter()
        cat_correct: Counter = Counter()
        for r in subset:
            gt_cat = r.get("gt_root_cause_error_type")
            pred_cat = r.get("root_cause_error_type")
            if gt_cat is None:
                continue  # no ground truth available
            cat_total[gt_cat] += 1
            if pred_cat == gt_cat:
                cat_correct[gt_cat] += 1

        by_cat = {}
        for cat in sorted(cat_total.keys()):
            t = cat_total[cat]
            c = cat_correct[cat]
            by_cat[cat] = {
                "total": t,
                "correct": c,
                "accuracy": round(c / t, 4) if t > 0 else 0.0,
            }
        ot = sum(cat_total.values())
        oc = sum(cat_correct.values())
        return {
            "accuracy_by_category": by_cat,
            "overall_accuracy": {
                "total": ot,
                "correct": oc,
                "accuracy": round(oc / ot, 4) if ot > 0 else 0.0,
            },
        }

    clean_records = [r for r in records if not r.get("suspected_corrupted_action")]
    corrupted_count = len(records) - len(clean_records)

    return {
        "total_records": len(records),
        "corrupted_records": corrupted_count,
        "category_distribution": {
            "overall": dict(overall_cat_counts.most_common()),
            "per_benchmark": {
                bm: dict(counts.most_common())
                for bm, counts in sorted(per_benchmark_cat_counts.items())
            },
        },
        "step_position_distribution": {
            cat: dict(buckets) for cat, buckets in sorted(position_by_category.items())
        },
        "including_corrupted": _compute_accuracy(records),
        "excluding_corrupted": _compute_accuracy(clean_records),
    }


# ---------------------------------------------------------------------------
# PART 3 — Heuristic downstream-effect scanner
# ---------------------------------------------------------------------------

# IMPORTANT: This is a KEYWORD HEURISTIC, not a real per-step error
# classification. It approximates downstream symptoms by scanning for
# failure-signal keywords in tool_response and reasoning fields AFTER the
# identified root-cause step. This uses data already available on disk,
# unlike AgentDebug's cascading_effects analysis which requires a separate
# per-step LLM pass that our judge pipeline deliberately does not perform.

_SIGNAL_KEYWORDS = [
    "error", "fail", "failed", "cannot", "invalid",
    "not found", "nothing happened", "unsuccessful",
]


def scan_downstream_effects(
    record: Dict[str, Any],
    raw_trajectory: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Scan steps AFTER root_cause_step_index for failure-signal keywords.

    THIS IS A KEYWORD HEURISTIC — it approximates downstream symptoms using
    data already available, unlike AgentDebug's cascading_effects which
    requires a separate per-step LLM pass we deliberately are not doing in
    our own judge.

    Args:
        record: A single judged record (must have 'root_cause_step_index').
        raw_trajectory: The trajectory dict (must have 'steps' list).

    Returns:
        List of dicts, each with step_index, keyword, field, and excerpt.
    """
    root_step = record.get("root_cause_step_index")
    if root_step is None:
        return []

    steps = raw_trajectory.get("steps", [])
    effects = []

    for step in steps:
        step_idx = step.get("step_index")
        if step_idx is None or step_idx <= root_step:
            continue

        for field_name in ("tool_response", "reasoning"):
            field_value = str(step.get(field_name) or "")
            field_lower = field_value.lower()
            for keyword in _SIGNAL_KEYWORDS:
                if keyword in field_lower:
                    effects.append({
                        "step_index": step_idx,
                        "keyword": keyword,
                        "field": field_name,
                        "excerpt": field_value[:100],
                    })
                    break  # one match per field per step is enough

    return effects


# ---------------------------------------------------------------------------
# PART 4 — CLI entry point
# ---------------------------------------------------------------------------

def _print_summary(stats: Dict[str, Any]) -> None:
    """Print a concise human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print(f"PROPAGATION ANALYSIS REPORT  ({stats['total_records']} records, "
          f"{stats.get('corrupted_records', 0)} flagged as corrupted)")
    print("=" * 60)

    # Category distribution
    print("\n--- Category Distribution (overall) ---")
    for cat, count in stats["category_distribution"]["overall"].items():
        print(f"  {cat:30s}  {count:4d}")

    print("\n--- Category Distribution (per benchmark) ---")
    for bm, cats in stats["category_distribution"]["per_benchmark"].items():
        print(f"  [{bm}]")
        for cat, count in cats.items():
            print(f"    {cat:28s}  {count:4d}")

    # Step-position buckets
    print("\n--- Step-Position Distribution (early/mid/late) ---")
    print(f"  {'Category':30s}  {'Early':>6s}  {'Mid':>6s}  {'Late':>6s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}")
    for cat, buckets in stats["step_position_distribution"].items():
        e = buckets.get("early", 0)
        m = buckets.get("mid", 0)
        l = buckets.get("late", 0)
        print(f"  {cat:30s}  {e:6d}  {m:6d}  {l:6d}")

    # Accuracy — print both views
    for label, key in [
        ("INCLUDING corrupted records", "including_corrupted"),
        ("EXCLUDING corrupted records", "excluding_corrupted"),
    ]:
        section = stats.get(key, {})
        if not section:
            continue
        print(f"\n--- Accuracy by Category — {label} ---")
        for cat, info in section["accuracy_by_category"].items():
            pct = info["accuracy"] * 100
            print(f"  {cat:30s}  {info['correct']:3d}/{info['total']:3d}  ({pct:5.1f}%)")

        oa = section["overall_accuracy"]
        pct = oa["accuracy"] * 100
        print(f"\n  {'OVERALL':30s}  {oa['correct']:3d}/{oa['total']:3d}  ({pct:5.1f}%)")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagation analysis — local analysis over judged data (no LLM calls).",
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="One or more judged-output JSON files to load and merge.",
    )
    parser.add_argument(
        "--output", "-o",
        default="findings/propagation_report.json",
        help="Path to write the combined JSON report.",
    )
    parser.add_argument(
        "--converted",
        default="findings/agenterrorbench_converted.json",
        help="Path to the converted dataset (for ground truth + trajectories).",
    )
    args = parser.parse_args()

    # Step 1: Load and merge judged records
    print(f"Loading {len(args.input)} judged file(s)...")
    records = load_judged_records(args.input, converted_path=args.converted)
    print(f"Loaded {len(records)} unique judged records.")

    # Step 2: Compute aggregate stats
    stats = compute_aggregate_stats(records)

    # Step 3: Scan downstream effects for every record
    dataset_by_id = _load_converted_dataset(args.converted)
    downstream_results = {}
    for rec in records:
        run_id = rec["run_id"]
        gt_run = dataset_by_id.get(run_id)
        if gt_run:
            raw_traj = gt_run.get("trajectory", {})
            effects = scan_downstream_effects(rec, raw_traj)
            if effects:
                downstream_results[run_id] = effects

    print(f"Downstream-effect scan: {len(downstream_results)} records have keyword matches.")

    # Step 4: Write combined report
    report = {
        "aggregate_stats": stats,
        "downstream_effects": downstream_results,
        "records_summary": [
            {
                "run_id": r["run_id"],
                "benchmark": r.get("benchmark"),
                "predicted_category": r.get("root_cause_error_type"),
                "predicted_step": r.get("root_cause_step_index"),
                "gt_category": r.get("gt_root_cause_error_type"),
                "gt_step": r.get("gt_root_cause_step_index"),
            }
            for r in records
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {args.output}")

    # Print human-readable summary
    _print_summary(stats)


if __name__ == "__main__":
    main()
