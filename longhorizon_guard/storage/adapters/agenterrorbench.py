#!/usr/bin/env python3
"""
Adapter: AgentErrorBench → longhorizon_guard SCHEMA.md v1.0 format.

One-way converter that reads the AgentErrorBench dataset (HuggingFace
davide221/agenterrorbench, 200 annotated failure trajectories) and produces
a consolidated {"runs": [...]} JSON file loadable by load_dataset().

Does NOT modify reader.py, SCHEMA.md, or core taxonomy files.

Taxonomy mapping (approved by project lead 2026-08-26):
  Memory   subtypes  → memory_error
  Reflection subtypes → reflection_error
  Planning subtypes   → planning_error
  Action   subtypes   → tool_use_error  (incl. misalignment)
  System   subtypes   → external_error  (incl. step_limit, with progress flag)
  Empty    subtypes   → other           (with original_subtype_missing flag)
  hallucination       → disambiguated via failure_modules field

Usage:
    python -m longhorizon_guard.storage.adapters.agenterrorbench \\
        --output findings/agenterrorbench_converted.json
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Normalization tables — fix data-quality inconsistencies in the source
# ---------------------------------------------------------------------------

_MODULE_NORMALIZE = {
    "plan": "planning",
    "planning": "planning",
    "memory": "memory",
    "reflection": "reflection",
    "action": "action",
    "system": "system",
    "others": "others",
}

_SUBTYPE_NORMALIZE = {
    "plan_inefficient": "inefficient_plan",
    # Case / whitespace variants handled by lowercase + strip below.
}

# ---------------------------------------------------------------------------
# (normalized_module, normalized_subtype) → our 7-category taxonomy
# ---------------------------------------------------------------------------

TAXONOMY_MAP: Dict[tuple, str] = {
    # Memory module
    ("memory", "hallucination"):            "memory_error",
    ("memory", "memory_retrieval_failure"):  "memory_error",
    ("memory", "over_simplification"):       "memory_error",
    # Reflection module
    ("reflection", "progress_misjudge"):         "reflection_error",
    ("reflection", "outcome_misinterpretation"): "reflection_error",
    ("reflection", "causal_misattribution"):     "reflection_error",
    ("reflection", "hallucination"):             "reflection_error",
    # Planning module
    ("planning", "constraint_ignorance"): "planning_error",
    ("planning", "impossible_action"):    "planning_error",
    ("planning", "inefficient_plan"):     "planning_error",
    # Action module  (misalignment = bad execution of good plan → tool_use_error)
    ("action", "misalignment"):   "tool_use_error",
    ("action", "invalid_action"): "tool_use_error",
    ("action", "format_error"):   "tool_use_error",
    ("action", "parameter_error"): "tool_use_error",
    # System module  (step_limit gets extra progress flag, all map to external_error)
    ("system", "step_limit"):           "external_error",
    ("system", "tool_execution_error"): "external_error",
    ("system", "llm_limit"):           "external_error",
    ("system", "environment_error"):   "external_error",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_module(raw: str) -> str:
    """Normalize module name to a canonical form."""
    key = raw.strip().lower()
    return _MODULE_NORMALIZE.get(key, key)


def _normalize_subtype(raw: str) -> str:
    """Normalize failure subtype name to a canonical form."""
    key = raw.strip().lower()
    return _SUBTYPE_NORMALIZE.get(key, key)


def _map_category(module: str, subtype: str) -> str:
    """Map an AgentErrorBench (module, subtype) pair to our 7-category taxonomy."""
    norm_mod = _normalize_module(module)
    norm_sub = _normalize_subtype(subtype)
    if not norm_sub:
        return "other"
    key = (norm_mod, norm_sub)
    if key in TAXONOMY_MAP:
        return TAXONOMY_MAP[key]
    logger.warning("Unmapped (module=%r, subtype=%r) → 'other'", norm_mod, norm_sub)
    return "other"


def _detect_slow_progress(messages: List[Dict[str, Any]]) -> bool:
    """Heuristic: was the agent making diverse (but slow) progress?

    Compares the last 10 assistant actions.  If more than half are unique
    (first 200 chars), the agent was exploring, not stuck in a loop.
    """
    actions = [
        (m.get("content") or "")[:200]
        for m in messages
        if m.get("role") == "assistant"
    ]
    if len(actions) < 5:
        return True  # Too few steps to call it a loop
    window = actions[-10:]
    unique_ratio = len(set(window)) / len(window)
    return unique_ratio > 0.5


def _build_step_error_map(step_annotations_str: str) -> Dict[int, str]:
    """Parse step_annotations JSON and return {1-based-step: our_category}."""
    try:
        annotations = json.loads(step_annotations_str)
    except (json.JSONDecodeError, TypeError):
        return {}

    error_map: Dict[int, str] = {}
    module_keys = ("memory", "reflection", "planning", "plan", "action",
                   "system", "others")

    for ann in annotations:
        step_num = ann.get("step")
        if step_num is None:
            continue
        for key in module_keys:
            if key in ann and isinstance(ann[key], dict):
                ft = ann[key].get("failure_type", "")
                error_map[step_num] = _map_category(key, ft)
                break

    return error_map


def _parse_assistant_message(content: str, observation: Optional[str] = None) -> Tuple[str, str, Dict[str, Any], Optional[str]]:
    """Parse assistant message content into (reasoning, action_name, action_args, tool_response)."""
    content = content or ""
    reasoning = content
    action_str = content

    # XML tag parsing (ALFWorld / ReAct XML format)
    if "<action>" in content:
        action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL | re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).strip()
            reasoning = re.sub(r"<action>.*?</action>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    elif "Action:" in content:
        parts = content.split("Action:", 1)
        reasoning = parts[0].strip() or content
        action_str = parts[1].strip()
    elif "Thought:" in content:
        parts = content.split("Thought:", 1)
        reasoning = "Thought: " + parts[1].strip()

    action_name = "react_action"
    action_args: Dict[str, Any] = {"command": action_str}

    match_bracket = re.match(r"^([a-zA-Z0-9_\-\s]+)\[(.*)\]$", action_str.strip())
    if match_bracket:
        action_name = match_bracket.group(1).strip()
        action_args = {"query": match_bracket.group(2).strip()}
    else:
        # Strip any stray HTML/XML tags from action_str
        clean_action_str = re.sub(r"<[^>]+>", "", action_str).strip()
        words = clean_action_str.split(maxsplit=1)
        if words and len(words[0]) <= 20:
            action_name = words[0].lower()
            if len(words) > 1:
                action_args = {"target": words[1]}
            else:
                action_args = {"target": clean_action_str}

    tool_response = observation if observation else None
    return reasoning, action_name, action_args, tool_response


def _convert_messages_to_steps(
    messages: List[Dict[str, Any]],
    step_error_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Convert user/assistant message pairs into SCHEMA.md step dicts."""
    steps: List[Dict[str, Any]] = []
    step_index = 0

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        # Next user message is the environment observation
        observation: Optional[str] = None
        if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
            observation = messages[i + 1].get("content")

        # step_annotations use 1-based numbering
        error_tag = step_error_map.get(step_index + 1)
        reasoning, action_name, action_args, tool_response = _parse_assistant_message(msg.get("content", ""), observation)

        steps.append({
            "step_index": step_index,
            "reasoning": reasoning,
            "action_name": action_name,
            "action_args": action_args,
            "timestamp": None,
            "tool_response": tool_response,
            "state_snapshot": None,
            "error_tag": error_tag,
        })
        step_index += 1

    return steps


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_record(record: Dict[str, Any], split_name: str) -> Dict[str, Any]:
    """Convert one AgentErrorBench record to a SCHEMA.md-compliant run dict."""

    trajectory_id = record["trajectory_id"]

    # --- Determine primary failure module + subtype ---
    failure_modules = record["failure_modules"]
    failure_types = record["failure_types"]
    critical_module = record["critical_failure_module"]

    primary_module = failure_modules[0] if failure_modules else critical_module
    primary_subtype = failure_types[0] if failure_types else ""

    norm_module = _normalize_module(primary_module)
    norm_subtype = _normalize_subtype(primary_subtype)
    our_category = _map_category(primary_module, primary_subtype)

    # --- Parse full trajectory once ---
    traj_data: Dict[str, Any] = {}
    try:
        traj_data = json.loads(record["full_trajectory"])
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse full_trajectory for %s: %s",
                       trajectory_id, exc)

    messages = traj_data.get("messages", [])

    # --- Build metadata ---
    failure_reasonings = record["failure_reasonings"]

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": trajectory_id,
        "task_id": record["task_type"],
        "trial_number": 1,
        "final_status": "fail",
        "total_steps_taken": record["num_steps"],
        "horizon_level": None,
        "duration_seconds": None,
        "grader_notes": failure_reasonings[0] if failure_reasonings else None,
        "created_at": None,
        "root_cause_step_index": max(record["critical_failure_step"] - 1, 0),
        "root_cause_error_type": our_category,
        "tag_confidence": 1.0,
        "tag_source": "agenterrorbench_import",
        # --- Provenance fields ---
        "source_dataset": "agenterrorbench",
        "source_split": split_name,
        "source_llm_model": record["llm_model"],
        "original_module": norm_module,
        "original_subtype": norm_subtype if norm_subtype else None,
    }

    # Flag: missing original subtype (27 records in dataset)
    if not primary_subtype.strip():
        metadata["original_subtype_missing"] = True
        metadata["tag_confidence"] = None  # fallback label, not a real annotation
        logger.debug("Record %s: original subtype is empty → 'other' "
                      "(tag_confidence set to null)", trajectory_id)

    # Flag: step_limit progress heuristic
    if norm_subtype == "step_limit":
        slow = _detect_slow_progress(messages)
        metadata["likely_root_cause_was_inefficient_planning"] = slow

    # --- Convert trajectory ---
    step_error_map = _build_step_error_map(record["step_annotations"])
    steps = _convert_messages_to_steps(messages, step_error_map)
    trajectory = {"steps": steps} if steps else None

    return {"metadata": metadata, "trajectory": trajectory}


def run_conversion(output_path: str) -> Dict[str, Any]:
    """Load AgentErrorBench, convert all records, write consolidated JSON.

    Returns a summary dict for logging / validation.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        logger.error("Install the 'datasets' package:  pip install datasets")
        sys.exit(1)

    logger.info("Loading AgentErrorBench from HuggingFace…")
    ds = hf_load("davide221/agenterrorbench")

    runs: List[Dict[str, Any]] = []
    skipped = 0
    flags = {
        "original_subtype_missing": 0,
        "step_limit_slow_progress": 0,
        "step_limit_stuck": 0,
    }

    for split_name in ds:
        logger.info("  Processing split '%s' (%d records)…",
                     split_name, len(ds[split_name]))
        for record in ds[split_name]:
            try:
                converted = convert_record(record, split_name)
                runs.append(converted)

                meta = converted["metadata"]
                if meta.get("original_subtype_missing"):
                    flags["original_subtype_missing"] += 1
                prog = meta.get("likely_root_cause_was_inefficient_planning")
                if prog is True:
                    flags["step_limit_slow_progress"] += 1
                elif prog is False:
                    flags["step_limit_stuck"] += 1

            except Exception as exc:
                tid = record.get("trajectory_id", "?")
                logger.warning("Skipped record %s: %s", tid, exc)
                skipped += 1

    # --- Write output ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, ensure_ascii=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # --- Category distribution ---
    cat_dist = Counter(r["metadata"]["root_cause_error_type"] for r in runs)

    summary = {
        "total_converted": len(runs),
        "skipped": skipped,
        "output_path": output_path,
        "output_size_mb": round(file_size_mb, 1),
        "flags": flags,
        "category_distribution": dict(sorted(cat_dist.items(),
                                             key=lambda x: -x[1])),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AgentErrorBench → longhorizon_guard SCHEMA.md v1.0",
    )
    parser.add_argument(
        "--output", "-o",
        default="findings/agenterrorbench_converted.json",
        help="Output path for consolidated JSON "
             "(default: findings/agenterrorbench_converted.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    summary = run_conversion(args.output)

    logger.info("=== Conversion Summary ===")
    logger.info("  Records converted : %d", summary["total_converted"])
    logger.info("  Records skipped   : %d", summary["skipped"])
    logger.info("  Output file       : %s (%.1f MB)",
                summary["output_path"], summary["output_size_mb"])
    logger.info("  Flags:")
    for flag, count in summary["flags"].items():
        logger.info("    %-40s %d", flag, count)
    logger.info("  Category distribution:")
    for cat, count in summary["category_distribution"].items():
        logger.info("    %-25s %d", cat, count)


if __name__ == "__main__":
    main()
