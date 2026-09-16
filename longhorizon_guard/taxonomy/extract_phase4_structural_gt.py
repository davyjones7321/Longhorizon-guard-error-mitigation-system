import sys
import os
import re
import json
from collections import Counter, defaultdict

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq

from longhorizon_guard.subgoals.tracker import (
    _detect_structural_outcome,
    _detect_tool_failure as tracker_detect_tool_failure,
)

# Custom benign patterns that also handle JSON-quoted `"error": null` / `"error": 0` / `'error': null`
_BENIGN_JSON_PATTERNS = [
    re.compile(r"\b0\s+errors?\b", re.IGNORECASE),
    re.compile(r"\bno\s+errors?\b", re.IGNORECASE),
    re.compile(r"['\"]?errors?['\"]?\s*:\s*(?:0|none|null)\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+errors?\b", re.IGNORECASE),
]

_MULTI_WORD_FAILURE_PATTERNS = [
    re.compile(r"\bcannot\s+find\b", re.IGNORECASE),
    re.compile(r"\bnothing\s+happens\b", re.IGNORECASE),
    re.compile(r"\binvalid\s+action\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+open\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+take\b", re.IGNORECASE),
    re.compile(r"\bsyntax\s+error\b", re.IGNORECASE),
]

_SINGLE_WORD_FAILURE_PATTERN = re.compile(r"\b(?:error|errors|failed|failure)\b", re.IGNORECASE)


def detect_tool_failure_calibrated(tool_resp: str):
    """Calibrated tool failure detector that correctly treats JSON `"error": null` as benign."""
    if not tool_resp or not str(tool_resp).strip():
        return None

    s = str(tool_resp)

    # 1. Multi-word failure expressions
    for pat in _MULTI_WORD_FAILURE_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(0)

    # 2. Strip benign patterns (including JSON "error": null)
    cleaned = s
    for benign in _BENIGN_JSON_PATTERNS:
        cleaned = benign.sub(" ", cleaned)

    # 3. Check single-word failure terms
    m = _SINGLE_WORD_FAILURE_PATTERN.search(cleaned)
    if m:
        return m.group(0)

    return None


def extract_dataset(path, name, is_alp=True, use_calibrated_failure=True):
    """Extract structural ground truth and compare against benchmark labels for a dataset."""
    t = pq.read_table(path)
    total = t.num_rows

    records = []

    for r in range(total):
        if is_alp:
            task_id = t.column("task_id")[r].as_py()
            bench_label = t.column("grade_pass")[r].as_py()
            conv = t.column("conversations")[r].as_py()
            tool_turns = [
                turn.get("value")
                for turn in conv
                if turn.get("from") in ("tool", "toolResult") and turn.get("value")
            ]
        else:
            task_id = t.column("instance_id")[r].as_py()
            bench_label = t.column("resolved")[r].as_py()
            if bench_label is None:
                bench_label = t.column("exit_status")[r].as_py() == "Submitted"
            msgs = t.column("messages")[r].as_py()
            tool_turns = [
                m.get("content")
                for m in msgs[2:]
                if m.get("role") == "user"
                and m.get("content")
                and ("<returncode>" in str(m.get("content")) or '{"ok"' in str(m.get("content")))
            ]

        last_tool = tool_turns[-1] if tool_turns else None

        if last_tool is None:
            structural_verdict = "ambiguous"
            structural_reason = "no_tool_response"
        else:
            struct_outcome = _detect_structural_outcome(last_tool)
            tool_str = str(last_tool).lower()
            if use_calibrated_failure:
                fail_term = detect_tool_failure_calibrated(tool_str)
            else:
                fail_term = tracker_detect_tool_failure(tool_str)

            if struct_outcome is True:
                if fail_term:
                    structural_verdict = "ambiguous"
                    structural_reason = f"ok_true_with_corroborating_failure_{fail_term}"
                else:
                    structural_verdict = "unambiguous_success"
                    structural_reason = "clean_returncode_0_or_ok_true"
            elif struct_outcome is False:
                structural_verdict = "unambiguous_failure"
                structural_reason = f"nonzero_returncode_or_ok_false_{fail_term or 'nonzero'}"
            else:
                structural_verdict = "ambiguous"
                structural_reason = "no_structural_tag_detected"

        # Determine match status
        if structural_verdict == "unambiguous_success":
            if bench_label is True:
                match_status = "agree"
            elif bench_label is False:
                match_status = "disagree"
            else:
                match_status = "unlabeled_benchmark"
        elif structural_verdict == "unambiguous_failure":
            if bench_label is False:
                match_status = "agree"
            elif bench_label is True:
                match_status = "disagree"
            else:
                match_status = "unlabeled_benchmark"
        else:
            match_status = "ambiguous"

        # Also compute reconciled status where None in ALP is treated as False (infra/verifier failure)
        bench_label_reconciled = False if bench_label is None else bench_label
        if structural_verdict == "unambiguous_success":
            match_status_reconciled = "agree" if bench_label_reconciled is True else "disagree"
        elif structural_verdict == "unambiguous_failure":
            match_status_reconciled = "agree" if bench_label_reconciled is False else "disagree"
        else:
            match_status_reconciled = "ambiguous"

        snippet = str(last_tool)[:150].replace("\n", " ") if last_tool else "NONE"
        records.append({
            "dataset": name,
            "task_id": task_id,
            "structural_verdict": structural_verdict,
            "structural_reason": structural_reason,
            "benchmark_label": bench_label,
            "match_status": match_status,
            "match_status_reconciled": match_status_reconciled,
            "snippet": snippet,
        })

    return records


def run_analysis():
    datasets = [
        ("terminal-bench-2", "new_dataset/agent_launch_pad/terminal-bench-2.parquet", True),
        ("scienceagentbench", "new_dataset/agent_launch_pad/scienceagentbench.parquet", True),
        ("swe_bench_test", "new_dataset/jetbrains_swe/swe_bench_test_trajectories.parquet", False),
        ("swe_smith", "new_dataset/jetbrains_swe/swe_smith_trajectories.parquet", False),
    ]

    print("================================================================================")
    print("PHASE 4 PART 1: FREE STRUCTURAL GROUND TRUTH EXTRACTION & BENCHMARK CROSS-CHECK")
    print("================================================================================")

    # 1. Run with calibrated detector (handling JSON "error": null correctly)
    print("\n--- RESULTS (Calibrated Rule S0: JSON 'error': null treated as benign) ---")
    all_records_calibrated = []
    summary_calibrated = {}

    for name, path, is_alp in datasets:
        recs = extract_dataset(path, name, is_alp=is_alp, use_calibrated_failure=True)
        all_records_calibrated.extend(recs)

        verdict_counts = Counter(r["structural_verdict"] for r in recs)
        match_counts = Counter(r["match_status"] for r in recs)
        match_reconciled = Counter(r["match_status_reconciled"] for r in recs)
        total = len(recs)
        unambiguous_total = verdict_counts["unambiguous_success"] + verdict_counts["unambiguous_failure"]

        summary_calibrated[name] = {
            "total": total,
            "unambiguous_total": unambiguous_total,
            "unambiguous_pct": (unambiguous_total / total) * 100,
            "unambiguous_success": verdict_counts["unambiguous_success"],
            "unambiguous_failure": verdict_counts["unambiguous_failure"],
            "ambiguous": verdict_counts["ambiguous"],
            "ambiguous_pct": (verdict_counts["ambiguous"] / total) * 100,
            "agree": match_counts["agree"],
            "disagree": match_counts["disagree"],
            "unlabeled": match_counts["unlabeled_benchmark"],
            "agree_reconciled": match_reconciled["agree"],
            "disagree_reconciled": match_reconciled["disagree"],
        }

    # Print summary table (Reconciled: benchmark None mapped to False)
    print(f"\n[TABLE A: FULLY RECONCILED (Agree + Disagree == Unambiguous)]")
    print(f"{'Dataset':<20} | {'Total':<6} | {'Unambiguous':<12} | {'Success':<8} | {'Failure':<8} | {'Ambiguous':<10} | {'Agree':<8} | {'Disagree':<8} | {'Check (A+D)':<11}")
    print("-" * 110)
    tot_all = sum(s["total"] for s in summary_calibrated.values())
    tot_unamb = sum(s["unambiguous_total"] for s in summary_calibrated.values())
    tot_succ = sum(s["unambiguous_success"] for s in summary_calibrated.values())
    tot_fail = sum(s["unambiguous_failure"] for s in summary_calibrated.values())
    tot_amb = sum(s["ambiguous"] for s in summary_calibrated.values())
    tot_agr_rec = sum(s["agree_reconciled"] for s in summary_calibrated.values())
    tot_dis_rec = sum(s["disagree_reconciled"] for s in summary_calibrated.values())

    for name, s in summary_calibrated.items():
        check_sum = s['agree_reconciled'] + s['disagree_reconciled']
        print(f"{name:<20} | {s['total']:<6} | {s['unambiguous_total']:<5} ({s['unambiguous_pct']:.1f}%) | {s['unambiguous_success']:<8} | {s['unambiguous_failure']:<8} | {s['ambiguous']:<5} ({s['ambiguous_pct']:.1f}%) | {s['agree_reconciled']:<8} | {s['disagree_reconciled']:<8} | {check_sum:<11}")
    print("-" * 110)
    print(f"{'TOTAL':<20} | {tot_all:<6} | {tot_unamb:<5} ({tot_unamb/tot_all*100:.1f}%) | {tot_succ:<8} | {tot_fail:<8} | {tot_amb:<5} ({tot_amb/tot_all*100:.1f}%) | {tot_agr_rec:<8} | {tot_dis_rec:<8} | {tot_agr_rec + tot_dis_rec:<11}")

    # Print explicit accounting table showing the un-evaluated benchmark gap
    print(f"\n[TABLE B: EXPLICIT UNLABELED BENCHMARK AUDIT (Showing exactly where the 173 and 11 came from)]")
    print(f"{'Dataset':<20} | {'Unambiguous':<12} | {'Agree (Pass/Fail)':<18} | {'Disagree':<10} | {'Unlabeled (None)':<18} | {'Sum (Agree+Dis+Unl)':<20}")
    print("-" * 110)
    for name, s in summary_calibrated.items():
        sum_check = s['agree'] + s['disagree'] + s['unlabeled']
        print(f"{name:<20} | {s['unambiguous_total']:<12} | {s['agree']:<18} | {s['disagree']:<10} | {s['unlabeled']:<18} | {sum_check:<20}")
    print("-" * 110)

    # 2. Also run with strict un-calibrated tracker._detect_tool_failure for comparison
    print("\n--- SENSITIVITY CHECK: Strict Tracker Regex (where JSON '\"error\": null' triggers failure) ---")
    summary_strict = {}
    for name, path, is_alp in datasets:
        recs = extract_dataset(path, name, is_alp=is_alp, use_calibrated_failure=False)
        verdict_counts = Counter(r["structural_verdict"] for r in recs)
        match_counts = Counter(r["match_status"] for r in recs)
        total = len(recs)
        unambiguous_total = verdict_counts["unambiguous_success"] + verdict_counts["unambiguous_failure"]
        summary_strict[name] = {
            "total": total,
            "unambiguous_total": unambiguous_total,
            "unambiguous_pct": (unambiguous_total / total) * 100,
            "unambiguous_success": verdict_counts["unambiguous_success"],
            "unambiguous_failure": verdict_counts["unambiguous_failure"],
            "ambiguous": verdict_counts["ambiguous"],
            "agree": match_counts["agree"],
            "disagree": match_counts["disagree"],
        }
    print(f"{'Dataset':<20} | {'Total':<6} | {'Unambiguous':<12} | {'Success':<8} | {'Failure':<8} | {'Ambiguous':<10} | {'Agree':<6} | {'Disagree':<8}")
    print("-" * 95)
    s_tot_unamb = sum(s["unambiguous_total"] for s in summary_strict.values())
    s_tot_succ = sum(s["unambiguous_success"] for s in summary_strict.values())
    s_tot_fail = sum(s["unambiguous_failure"] for s in summary_strict.values())
    s_tot_amb = sum(s["ambiguous"] for s in summary_strict.values())
    s_tot_agree = sum(s["agree"] for s in summary_strict.values())
    s_tot_disagree = sum(s["disagree"] for s in summary_strict.values())
    for name, s in summary_strict.items():
        print(f"{name:<20} | {s['total']:<6} | {s['unambiguous_total']:<5} ({s['unambiguous_pct']:.1f}%) | {s['unambiguous_success']:<8} | {s['unambiguous_failure']:<8} | {s['ambiguous']:<10} | {s['agree']:<6} | {s['disagree']:<8}")
    print("-" * 95)
    print(f"{'TOTAL':<20} | {tot_all:<6} | {s_tot_unamb:<5} ({s_tot_unamb/tot_all*100:.1f}%) | {s_tot_succ:<8} | {s_tot_fail:<8} | {s_tot_amb:<10} | {s_tot_agree:<6} | {s_tot_disagree:<8}")

    # 3. Select 15 representative spot-check examples
    # Diversity: across all 4 datasets, covering:
    # - Agree (structural success & bench True)
    # - Agree (structural failure & bench False)
    # - Disagree (structural success & bench False - false positive)
    # - Disagree (structural failure & bench True - false negative)
    # - Ambiguous (corroborating failure keywords or no tool)
    print("\n================================================================================")
    print("SPOT-CHECK SAMPLE (15 REPRESENTATIVE TRAJECTORIES)")
    print("================================================================================")

    # Pick samples intentionally
    spot_checks = []
    # 1. ALP Agreement: Success + True
    for r in all_records_calibrated:
        if r["dataset"] == "terminal-bench-2" and r["match_status"] == "agree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 2. ALP Agreement: Failure + False
    for r in all_records_calibrated:
        if r["dataset"] == "terminal-bench-2" and r["match_status"] == "agree" and r["structural_verdict"] == "unambiguous_failure":
            spot_checks.append(r)
            break
    # 3. ALP Disagreement: Structural Success, but Benchmark False (e.g. tool ok, test failed)
    for r in all_records_calibrated:
        if r["dataset"] == "terminal-bench-2" and r["match_status"] == "disagree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 4. ALP Disagreement: Structural Failure, but Benchmark True
    for r in all_records_calibrated:
        if r["dataset"] == "terminal-bench-2" and r["match_status"] == "disagree" and r["structural_verdict"] == "unambiguous_failure":
            spot_checks.append(r)
            break
    # 5. ALP Ambiguous: no tool response (agent failed to respond)
    for r in all_records_calibrated:
        if r["dataset"] == "terminal-bench-2" and r["structural_reason"] == "no_tool_response":
            spot_checks.append(r)
            break
    # 6. ALP Ambiguous: ok: true but corroborating failure keyword
    for r in all_records_calibrated:
        if r["dataset"] == "terminal-bench-2" and "ok_true_with_corroborating_failure" in r["structural_reason"]:
            spot_checks.append(r)
            break
    # 7. ScienceAgentBench Agreement: Success + True
    for r in all_records_calibrated:
        if r["dataset"] == "scienceagentbench" and r["match_status"] == "agree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 8. ScienceAgentBench Disagreement: Structural Success, Benchmark False
    for r in all_records_calibrated:
        if r["dataset"] == "scienceagentbench" and r["match_status"] == "disagree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 9. ScienceAgentBench Ambiguous: ok: true with corroborating failure
    for r in all_records_calibrated:
        if r["dataset"] == "scienceagentbench" and "ok_true_with_corroborating_failure" in r["structural_reason"]:
            spot_checks.append(r)
            break
    # 10. SWE-bench-test: Structural Success, Exit Submitted
    for r in all_records_calibrated:
        if r["dataset"] == "swe_bench_test" and r["match_status"] == "agree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 11. SWE-bench-test: Structural Success, Exit LimitsExceeded (disagreement)
    for r in all_records_calibrated:
        if r["dataset"] == "swe_bench_test" and r["match_status"] == "disagree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 12. SWE-bench-test: Structural Failure (nonzero returncode)
    for r in all_records_calibrated:
        if r["dataset"] == "swe_bench_test" and r["structural_verdict"] == "unambiguous_failure":
            spot_checks.append(r)
            break
    # 13. SWE-Smith: Agreement Success + Resolved True
    for r in all_records_calibrated:
        if r["dataset"] == "swe_smith" and r["match_status"] == "agree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 14. SWE-Smith: Disagreement Success + Resolved False
    for r in all_records_calibrated:
        if r["dataset"] == "swe_smith" and r["match_status"] == "disagree" and r["structural_verdict"] == "unambiguous_success":
            spot_checks.append(r)
            break
    # 15. SWE-Smith: Agreement Failure + Resolved False
    for r in all_records_calibrated:
        if r["dataset"] == "swe_smith" and r["match_status"] == "agree" and r["structural_verdict"] == "unambiguous_failure":
            spot_checks.append(r)
            break

    # Format spot check output
    for idx, sample in enumerate(spot_checks, 1):
        print(f"\n--- Example {idx}: [{sample['dataset']}] {sample['task_id']} ---")
        print(f"  Structural Verdict: {sample['structural_verdict']} ({sample['structural_reason']})")
        print(f"  Benchmark Label:    {sample['benchmark_label']}")
        print(f"  Match Status:       {sample['match_status']}")
        print(f"  Final Tool Snippet: {sample['snippet']}")

    # Save full results to JSON in scratch for reference
    output_path = "scratch/phase4_structural_ground_truth.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary_calibrated": summary_calibrated,
            "summary_strict": summary_strict,
            "spot_checks": spot_checks,
        }, f, indent=2)
    print(f"\nSaved full results to {output_path}")

if __name__ == "__main__":
    run_analysis()
