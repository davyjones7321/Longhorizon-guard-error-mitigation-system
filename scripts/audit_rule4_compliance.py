#!/usr/bin/env python3
"""Read-only Rule 4 compliance audit for existing LLM-as-judge outputs.

Scans every *judged*.json file below findings/, joins each prediction to its
source trajectory by run_id, and reports only clear Rule 4 triggers at the
model-predicted root-cause step. The script reads JSON and prints to stdout;
it makes no API calls and writes no files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINDINGS_DIR = REPO_ROOT / "findings"
GROUND_TRUTH_CANDIDATES = (
    REPO_ROOT / "findings" / "agenterrorbench_converted.json",
    REPO_ROOT / "sorted.json",
    REPO_ROOT / "parsed-data" / "sorted.json",
)

# These pairs are intentionally narrow. They are only used when both literal
# word families appear in the same action parameter value.
CONTRADICTORY_TERM_PAIRS = (
    ("men", (r"\bmen(?:'s)?\b", r"\bmens\b"), "women", (r"\bwomen(?:'s)?\b", r"\bwomens\b")),
    ("male", (r"\bmale\b",), "female", (r"\bfemale\b",)),
    ("boys", (r"\bboys?\b",), "girls", (r"\bgirls?\b",)),
)

# Missing-term detection is restricted to explicit gender requirements in an
# available task description. This avoids inferring arbitrary requirements.
EXPLICIT_REQUIRED_TERMS = (
    ("men", (r"\bmen(?:'s)?\b", r"\bmens\b")),
    ("women", (r"\bwomen(?:'s)?\b", r"\bwomens\b")),
    ("male", (r"\bmale\b",)),
    ("female", (r"\bfemale\b",)),
    ("boys", (r"\bboys?\b",)),
    ("girls", (r"\bgirls?\b",)),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def runs_from_dataset(data: Any) -> List[Dict[str, Any]]:
    """Match judge.py's dataset shape handling: accept {"runs": [...]} or a list."""
    if isinstance(data, dict):
        runs = data.get("runs", [])
    elif isinstance(data, list):
        runs = data
    else:
        runs = []
    return [run for run in runs if isinstance(run, dict)]


def load_ground_truth() -> Tuple[Dict[str, Dict[str, Any]], List[Path]]:
    """Build the run_id lookup used by judge.py's metric calculation."""
    run_map: Dict[str, Dict[str, Any]] = {}
    loaded_paths: List[Path] = []

    for path in GROUND_TRUTH_CANDIDATES:
        if not path.exists():
            continue
        for run in runs_from_dataset(load_json(path)):
            metadata = run.get("metadata", {})
            run_id = metadata.get("run_id")
            if isinstance(run_id, str) and run_id not in run_map:
                run_map[run_id] = run
        loaded_paths.append(path)

    return run_map, loaded_paths


def iter_judgments(data: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (run_id, judgment) from judge.py outputs and provider list outputs."""
    if isinstance(data, dict) and isinstance(data.get("judgments"), dict):
        for run_id, judgment in data["judgments"].items():
            if isinstance(run_id, str) and isinstance(judgment, dict):
                yield run_id, judgment
        return

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {})
            run_id = item.get("run_id") or metadata.get("run_id")
            if isinstance(run_id, str):
                yield run_id, item


def find_step(run: Dict[str, Any], step_index: Any) -> Optional[Dict[str, Any]]:
    try:
        target_index = int(step_index)
    except (TypeError, ValueError):
        return None

    steps = run.get("trajectory", {}).get("steps", [])
    for step in steps:
        if isinstance(step, dict) and step.get("step_index") == target_index:
            return step
    return None


def task_description_for(run: Dict[str, Any]) -> str:
    """Return a genuine task description when one is available, never task_id."""
    metadata = run.get("metadata", {})
    for container in (metadata, run):
        for key in ("task_description", "description"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def action_parameter_text(action_args: Any) -> str:
    if isinstance(action_args, dict):
        return "\n".join(str(value) for value in action_args.values() if value is not None)
    return str(action_args or "")


def is_search_step(step: Dict[str, Any]) -> bool:
    return "search" in str(step.get("action_name", "")).lower()


def has_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def query_values_from_step(step: Dict[str, Any]) -> List[str]:
    """Extract explicit query parameters and explicitly labeled queries in reasoning."""
    values: List[str] = []
    action_args = step.get("action_args", {})
    if isinstance(action_args, dict):
        for key, value in action_args.items():
            if "query" in str(key).lower() or "search" in str(key).lower():
                values.append(str(value or "").strip())

    reasoning = str(step.get("reasoning") or "")
    for match in re.finditer(r"\b(?:search\s*query|searchquery|query)\s*[:=]\s*([^\n<]{1,500})", reasoning, flags=re.IGNORECASE):
        values.append(match.group(1).strip())
    return list(dict.fromkeys(values))


def detect_rule4_triggers(run: Dict[str, Any], step: Dict[str, Any]) -> List[str]:
    """Return only literal, explainable Rule 4 trigger descriptions."""
    action_args = step.get("action_args", {})
    parameter_text = action_parameter_text(action_args)
    normalized = parameter_text.lower()
    triggers: List[str] = []

    for left_name, left_patterns, right_name, right_patterns in CONTRADICTORY_TERM_PAIRS:
        if has_any_pattern(normalized, left_patterns) and has_any_pattern(normalized, right_patterns):
            triggers.append(
                f"contradictory parameter terms: both '{left_name}' and '{right_name}' appear literally"
            )

    if not is_search_step(step):
        return triggers

    query_values = query_values_from_step(step)
    query_text = "\n".join(query_values).lower()
    description = task_description_for(run).lower()
    if description and query_values:
        for term_name, term_patterns in EXPLICIT_REQUIRED_TERMS:
            if has_any_pattern(description, term_patterns) and not has_any_pattern(query_text, term_patterns):
                triggers.append(
                    f"task description explicitly mentions '{term_name}', but search parameters omit it"
                )

    for query in query_values:
        if not query:
            triggers.append("empty search query parameter")
        elif not re.search(r"[a-z0-9]", query, flags=re.IGNORECASE):
            triggers.append("search query contains no alphanumeric keyword")
        elif re.search(r"<[^>]+>|\{\{|\}\}|\b(?:action|action-input|thought)\s*:", query, flags=re.IGNORECASE):
            triggers.append("search query visibly contains action/template syntax rather than keywords")

    return list(dict.fromkeys(triggers))


def audit(findings_dir: Path) -> int:
    ground_truth, source_paths = load_ground_truth()
    judged_paths = sorted(findings_dir.rglob("*judged*.json"))

    print("=" * 88)
    print("RULE 4 ABSOLUTE-PRIORITY COMPLIANCE AUDIT")
    print("Read-only: reads existing JSON only; no API calls and no output files are created.")
    print(f"Ground-truth sources: {', '.join(str(path.relative_to(REPO_ROOT)) for path in source_paths) or '[none]'}")
    print(f"Judged output files discovered: {len(judged_paths)}")
    print("=" * 88)

    total_predictions = 0
    missing_ground_truth = 0
    missing_predicted_step = 0
    flagged = 0
    compliant = 0
    violations = 0

    for path in judged_paths:
        relative_path = path.relative_to(REPO_ROOT)
        try:
            judgments = list(iter_judgments(load_json(path)))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[SKIP] {relative_path}: could not read JSON ({exc})")
            continue

        print(f"\n--- {relative_path} ({len(judgments)} judgment record(s)) ---")
        for run_id, judgment in judgments:
            total_predictions += 1
            run = ground_truth.get(run_id)
            if run is None:
                missing_ground_truth += 1
                continue

            predicted_step = judgment.get("root_cause_step_index")
            step = find_step(run, predicted_step)
            if step is None:
                missing_predicted_step += 1
                continue

            triggers = detect_rule4_triggers(run, step)
            if not triggers:
                continue

            flagged += 1
            predicted_category = judgment.get("root_cause_error_type", "[missing]")
            is_compliant = predicted_category == "tool_use_error"
            if is_compliant:
                compliant += 1
            else:
                violations += 1

            print(f"Run ID: {run_id}")
            print(f"Predicted root cause: {predicted_category} @ step {predicted_step}")
            print(f"Action name: {step.get('action_name', '[missing]')}")
            print(f"Rule 4 triggers: {'; '.join(triggers)}")
            print("Action args:")
            print(json.dumps(step.get("action_args", {}), ensure_ascii=False, indent=2))
            print(f"Rule 4 result: {'COMPLIANT (tool_use_error)' if is_compliant else 'VIOLATION (not tool_use_error)'}")
            print("-" * 88)

    compliance_pct = (compliant / flagged * 100.0) if flagged else 0.0
    violation_pct = (violations / flagged * 100.0) if flagged else 0.0
    print("\n" + "=" * 88)
    print("SUMMARY")
    print(f"Total judged records scanned: {total_predictions}")
    print(f"Total flagged Rule 4 trigger cases found: {flagged}")
    print(f"Tagged tool_use_error (compliant): {compliant} ({compliance_pct:.1f}%)")
    print(f"Tagged something else (violation): {violations} ({violation_pct:.1f}%)")
    print(f"Skipped for missing ground truth: {missing_ground_truth}")
    print(f"Skipped because predicted step was absent: {missing_predicted_step}")
    print("=" * 88)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Rule 4 compliance audit for judged outputs.")
    parser.add_argument(
        "--findings-dir",
        type=Path,
        default=DEFAULT_FINDINGS_DIR,
        help="Directory recursively scanned for *judged*.json files.",
    )
    args = parser.parse_args()
    return audit(args.findings_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
