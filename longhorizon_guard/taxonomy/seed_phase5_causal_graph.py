"""Automated Phase 5 Seeding Script for LongHorizon Guard Memory.

Seeds CausalErrorGraph and updates pattern_library.json using Phase 3
empirical comparison judgments (25 records from new_dataset/comparison_judgments.jsonl),
completely replacing legacy AlfWorld/WebShop embodied patterns.
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from longhorizon_guard.memory.causal_graph import CausalErrorGraph
from longhorizon_guard.memory.seed_loader import bootstrap_memory_graph
from longhorizon_guard.pattern_library.schema import PatternEntry, TrajectorySnippet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_phase5")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGMENTS_PATH = os.path.join(REPO_ROOT, "new_dataset", "comparison_judgments.jsonl")
PAIRS_PATH = os.path.join(REPO_ROOT, "new_dataset", "divergence_pairs.jsonl")
FINDINGS_LIB_PATH = os.path.join(REPO_ROOT, "findings", "pattern_library.json")
DATA_LIB_PATH = os.path.join(REPO_ROOT, "longhorizon_guard", "data", "pattern_library.json")
CAUSAL_GRAPH_PATH = os.path.join(REPO_ROOT, "findings", "memory", "causal_graph.json")

# Curated, descriptive trigger descriptions derived from chain_of_thought
TRIGGER_DESCRIPTIONS = {
    "pytorch-model-recovery": (
        "ignored a ModuleNotFoundError for PyTorch during initial environment inspection, "
        "failing to install the missing library before proceeding with dependent modeling steps"
    ),
    "torch-tensor-parallelism": (
        "inspected the implementation file but failed to notice the missing RowParallelLinear class, "
        "proceeding with syntax verification on an incomplete solution"
    ),
    "qemu-startup": (
        "configured QEMU with network port-forwarding rather than exposing the VM's serial console "
        "on telnet, failing to establish the interactive login prompt"
    ),
    "git-multibranch": (
        "invoked the todo checklist tool instead of the terminal execution tool, "
        "failing to install required server packages"
    ),
    "kv-store-grpc": (
        "invoked the todo tracking tool instead of executing pip install via the terminal, "
        "leaving required gRPC dependencies uninstalled"
    ),
    "hf-model-inference": (
        "recorded items with the todo tool instead of executing directory setup and package "
        "installations through the terminal"
    ),
    "git-leak-recovery": (
        "searched reachable git commit logs and grep history instead of checking dangling objects "
        "with git fsck to recover an excised secret"
    ),
    "large-scale-text-editing": (
        "applied an overly broad global whitespace substitution macro that stripped mandatory "
        "leading indentation and column alignment"
    ),
    "tune-mjcf": (
        "halted immediately after receiving the task prompt without formulating a plan or "
        "issuing tool calls to inspect and optimize the simulation model"
    ),
    "adaptive-rejection-sampler": (
        "implemented a simplified sampler algorithm that omitted essential log-concavity "
        "verification and mandatory input validation checks"
    ),
    "nginx-request-logging": (
        "called the todo checklist tool instead of running terminal commands to install and "
        "provision the Nginx service"
    ),
    "headless-terminal": (
        "called search_files omitting the mandatory directory path argument, causing an empty "
        "search result and missing the base class definition"
    ),
    "cancel-async-tasks": (
        "assumed task completion immediately after initial file creation without verifying whether "
        "asynchronous task cancellation and cleanup executed properly"
    ),
    "pypi-server": (
        "supplied an incorrect nested file path to write_file, preventing the module from being "
        "exposed at the expected package root"
    ),
    "regex-log": (
        "constructed a rigid regex pattern enforcing strict ordering between IP address and timestamp, "
        "rejecting conforming log entries where the timestamp preceded the IP"
    ),
    "cobol-modernization": (
        "invoked read_file on a binary data file instead of using terminal inspection tools like "
        "hexdump, failing to parse the binary record format"
    ),
    "polyglot-c-py": (
        "emitted reasoning thoughts without issuing a file-writing tool call to materialize the "
        "required polyglot source code"
    ),
    "log-summary-date-ranges": (
        "relied on an unvalidated execution success message without inspecting the output CSV file "
        "or confirming that input logs were processed"
    ),
    "openssl-selfsigned-cert": (
        "formulated an incomplete security workflow that terminated after certificate creation, "
        "omitting mandatory fingerprint extraction and verification output steps"
    ),
    "fix-git": (
        "executed git commands in the initial working directory without first locating or entering "
        "the target repository directory"
    ),
    "model-extraction-relu-logits": (
        "used the code execution sandbox instead of the container terminal tool to run a script "
        "requiring pre-installed scientific libraries, causing a ModuleNotFoundError"
    ),
    "multi-source-data-merger": (
        "invoked the todo tool instead of read_file to retrieve source data, leaving subsequent "
        "data merger operations without input records"
    ),
    "constraints-scheduling": (
        "planned only passive calendar file reads without algorithmic constraint-solving logic "
        "to resolve the optimal meeting slot"
    ),
    "sparql-university": (
        "overwrote the query file without reading back the contents to inspect whether the filter "
        "logic satisfied the threshold condition"
    ),
    "sab_3": (
        "stopped after generating reasoning thoughts without executing tool calls to inspect the "
        "training dataset or write the predictive model script"
    ),
}


def parse_trajectory_steps(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parses a ShareGPT conversation turn list into structured step dictionaries."""
    steps: List[Dict[str, Any]] = []
    i = 0
    step_counter = 0
    n = len(conversations)

    while i < n:
        turn = conversations[i]
        role = turn.get("from") or turn.get("role")
        val = str(turn.get("value") or turn.get("content") or "")

        if role == "gpt":
            tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", val, re.DOTALL)
            think_match = re.search(r"<think>(.*?)</think>", val, re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else ""

            if tool_calls:
                tc_str = tool_calls[0]
                act_name = ""
                act_args: Dict[str, Any] = {}
                try:
                    parsed = json.loads(tc_str.strip())
                    act_name = parsed.get("name", "")
                    raw_args = parsed.get("args", {})
                    if isinstance(raw_args, dict):
                        act_args = raw_args
                    else:
                        act_args = {"args": str(raw_args)}
                except Exception:
                    act_name = "unknown"
                    act_args = {"raw": tc_str.strip()[:100]}

                # Check if next turn is a gpt turn with reasoning
                j = i + 1
                if j < n:
                    next_role = conversations[j].get("from") or conversations[j].get("role")
                    next_val = str(conversations[j].get("value") or conversations[j].get("content") or "")
                    if next_role == "gpt" and "<think>" in next_val:
                        next_think = re.search(r"<think>(.*?)</think>", next_val, re.DOTALL)
                        if next_think and not reasoning:
                            reasoning = next_think.group(1).strip()
                        j += 1

                # Check if turn at j is tool response
                tool_resp: Optional[str] = None
                if j < n:
                    resp_role = conversations[j].get("from") or conversations[j].get("role")
                    resp_val = str(conversations[j].get("value") or conversations[j].get("content") or "")
                    if resp_role in ("tool", "toolResult"):
                        m = re.search(r"<tool_response>(.*?)</tool_response>", resp_val, re.DOTALL)
                        tool_resp = m.group(1).strip() if m else resp_val.strip()
                        i = j
                    else:
                        i = j - 1
                else:
                    i = j - 1

                steps.append({
                    "turn_index": i,
                    "step_index": step_counter,
                    "reasoning": reasoning[:300],
                    "action_name": act_name,
                    "action_args": act_args,
                    "tool_response": tool_resp[:400] if tool_resp else None,
                })
                step_counter += 1
            else:
                steps.append({
                    "turn_index": i,
                    "step_index": step_counter,
                    "reasoning": reasoning[:300],
                    "action_name": "none",
                    "action_args": {},
                    "tool_response": None,
                })
                step_counter += 1
        i += 1

    return steps


def extract_snippet_window(
    steps: List[Dict[str, Any]],
    divergence_step: Optional[int],
) -> List[TrajectorySnippet]:
    """Extracts a focused 2-4 step window around the confirmed divergence step."""
    if not steps:
        return [
            TrajectorySnippet(
                step_index=0,
                reasoning="Agent stopped without issuing any tool calls or executing any plan.",
                action_name="none",
                action_args={},
                tool_response=None,
            )
        ]

    target_turn = divergence_step if divergence_step is not None else 0
    closest_idx = min(range(len(steps)), key=lambda k: abs(steps[k]["turn_index"] - target_turn))
    w_start = max(0, closest_idx - 1)
    w_end = min(len(steps), closest_idx + 3)
    window_steps = steps[w_start:w_end]

    snippets: List[TrajectorySnippet] = []
    for s in window_steps:
        snippets.append(
            TrajectorySnippet(
                step_index=s["step_index"],
                reasoning=s["reasoning"],
                action_name=s["action_name"],
                action_args=s["action_args"],
                tool_response=s["tool_response"],
            )
        )
    return snippets


def build_phase5_patterns() -> List[PatternEntry]:
    """Loads Phase 3 comparison judgments and divergence pairs to build 25 patterns."""
    with open(JUDGMENTS_PATH, "r", encoding="utf-8") as f:
        judgments = [json.loads(line) for line in f if line.strip()]

    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        pairs = {json.loads(line)["task_id"]: json.loads(line) for line in f if line.strip()}

    patterns: List[PatternEntry] = []
    category_counters: Dict[str, int] = {}

    for j in judgments:
        tid = j["task_id"]
        cat = j["root_cause_error_type"]
        confidence = float(j.get("confidence", 1.0))
        safe_alt = j.get("what_succeeded_did_differently", "").strip()
        f_step = j.get("confirmed_divergence_step_failure")

        # Category pattern numbering: pattern_<category>_<NNN>
        category_counters[cat] = category_counters.get(cat, 0) + 1
        pat_id = f"pattern_{cat}_{category_counters[cat]:03d}"

        trigger_desc = TRIGGER_DESCRIPTIONS.get(
            tid,
            j.get("chain_of_thought", "")[:200].strip(),
        )

        pair = pairs.get(tid, {})
        fail_conv = pair.get("failure", {}).get("conversations", [])
        parsed_steps = parse_trajectory_steps(fail_conv)
        snippets = extract_snippet_window(parsed_steps, f_step)

        entry = PatternEntry(
            pattern_id=pat_id,
            category=cat,
            trigger_description=trigger_desc,
            example_snippet=snippets,
            safe_alternative=safe_alt,
            source_run_ids=[tid],
            confidence=confidence,
        )
        patterns.append(entry)

    return patterns


def main() -> None:
    logger.info("=== Phase 5 Causal Graph Seeding: Starting ===")

    # 1. Baseline summary before re-seeding
    logger.info("Checking baseline CausalErrorGraph before re-seeding...")
    if os.path.exists(CAUSAL_GRAPH_PATH):
        baseline_graph = CausalErrorGraph(storage_path=CAUSAL_GRAPH_PATH)
        baseline_summary = baseline_graph.summary()
    else:
        baseline_summary = {}

    print("\n--- BASELINE CausalErrorGraph SUMMARY (BEFORE) ---")
    print(json.dumps(baseline_summary, indent=2))

    # 2. Build 25 patterns from Phase 3 data
    patterns = build_phase5_patterns()
    logger.info("Built %d new clean patterns from Phase 3 judgments.", len(patterns))

    # Category breakdown
    cat_counts: Dict[str, int] = {}
    for p in patterns:
        cat_counts[p.category] = cat_counts.get(p.category, 0) + 1
    logger.info("Category breakdown: %s", cat_counts)

    # 3. Save to both pattern library locations
    payload = {"patterns": [p.to_dict() for p in patterns]}

    for path in (FINDINGS_LIB_PATH, DATA_LIB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Successfully wrote %d patterns to: %s", len(patterns), path)

    # 4. Re-seed CausalErrorGraph (clean rebuild from scratch, no stale legacy nodes)
    logger.info("Re-seeding CausalErrorGraph from scratch via bootstrap_memory_graph...")
    if os.path.exists(CAUSAL_GRAPH_PATH):
        os.remove(CAUSAL_GRAPH_PATH)
    new_graph = bootstrap_memory_graph(
        storage_path=CAUSAL_GRAPH_PATH,
        pattern_library_path=FINDINGS_LIB_PATH,
    )
    after_summary = new_graph.summary()

    print("\n--- UPDATED CausalErrorGraph SUMMARY (AFTER) ---")
    print(json.dumps(after_summary, indent=2))
    logger.info("=== Phase 5 Causal Graph Seeding: Complete ===")


if __name__ == "__main__":
    main()
