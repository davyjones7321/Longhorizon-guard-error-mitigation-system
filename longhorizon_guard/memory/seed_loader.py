"""Historical knowledge and failure pattern seed loader for LongHorizon Guard Memory.

Populates the CausalErrorGraph with known failure patterns, canonical tool anti-patterns,
and error cascade dynamics derived from empirical evaluations (excluding batch3 records).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from longhorizon_guard.memory.causal_graph import CausalErrorGraph, _generate_id
from longhorizon_guard.memory.schema import (
    ActionPatternNode,
    ErrorSignatureNode,
    NodeType,
    RecoveryNode,
    RemediedByEdge,
    SubgoalNode,
    TriggersErrorEdge,
)

logger = logging.getLogger("longhorizon_guard.memory.seed_loader")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATTERN_LIB = os.path.join(REPO_ROOT, "findings", "pattern_library.json")


def seed_from_pattern_library(
    causal_graph: CausalErrorGraph,
    pattern_library_path: Optional[str] = None,
) -> int:
    """Ingest patterns from pattern_library.json into the causal graph."""
    path = pattern_library_path or DEFAULT_PATTERN_LIB
    if not os.path.exists(path):
        logger.warning("Pattern library file not found at: %s", path)
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to load pattern library: %s", exc)
        return 0

    patterns = data.get("patterns", [])
    count = 0

    for pat in patterns:
        pat_id = pat.get("pattern_id") or _generate_id("pat", pat.get("trigger_description", ""))
        category = pat.get("category", "other")
        desc = pat.get("trigger_description", "")
        safe_alt = pat.get("safe_alternative", "")
        confidence = float(pat.get("confidence", 0.85))

        err_node = ErrorSignatureNode(
            error_id=pat_id,
            category=category,
            pattern_regex=desc[:120],
            description=desc,
            severity="high" if confidence >= 0.90 else "medium",
            metadata={"confidence": confidence, "source": "pattern_library"},
        )
        causal_graph.add_node(err_node)
        count += 1

        # Link recovery action if available
        if safe_alt:
            rec_id = _generate_id("rec", safe_alt)
            causal_graph.add_node(RecoveryNode(
                recovery_id=rec_id,
                action_description=safe_alt,
                safe_alternative=safe_alt,
            ))
            causal_graph.add_edge(RemediedByEdge(
                source_id=pat_id,
                target_id=rec_id,
                success_rate=confidence,
            ))

        # Ingest snippet action steps
        snippets = pat.get("example_snippet", [])
        if isinstance(snippets, list):
            for step in snippets:
                if not isinstance(step, dict):
                    continue
                act_name = str(step.get("action_name", "")).strip().lstrip("'\"")
                act_args = step.get("action_args", {})
                if act_name:
                    sig = f"{act_name}:{json.dumps(act_args, sort_keys=True)}"
                    act_id = _generate_id("act", sig)
                    if act_id not in causal_graph.graph:
                        causal_graph.add_node(ActionPatternNode(
                            action_id=act_id,
                            tool_name=act_name,
                            argument_pattern=str(act_args),
                            normalized_signature=sig,
                        ))
                    causal_graph.add_edge(TriggersErrorEdge(
                        source_id=act_id,
                        target_id=pat_id,
                        confidence=confidence,
                    ))

    return count


def seed_canonical_anti_patterns(causal_graph: CausalErrorGraph) -> int:
    """Seed high-confidence tool anti-patterns and known recovery paths."""
    anti_patterns = [
        {
            "tool": "bash",
            "pattern": "rm -rf /",
            "error_category": "environment_error",
            "error_desc": "Attempted destructive recursive deletion on root path",
            "recovery": "Target specific subdirectory or use safe trash utility",
        },
        {
            "tool": "bash",
            "pattern": "git push --force origin main",
            "error_category": "external_error",
            "error_desc": "Destructive force push to protected main branch",
            "recovery": "Pull and rebase latest commits instead of force pushing",
        },
        {
            "tool": "json_parser",
            "pattern": "json.loads without try-except",
            "error_category": "tool_use_error",
            "error_desc": "JSONDecodeError on malformed tool payload",
            "recovery": "Wrap JSON parsing in try-except block and sanitize escape characters",
        },
        {
            "tool": "llm_client",
            "pattern": "max_tokens_exceeded",
            "error_category": "context_length_error",
            "error_desc": "Context window exceeded token limit",
            "recovery": "Truncate historical conversation messages or use summary prompt",
        },
    ]

    for item in anti_patterns:
        causal_graph.record_action_failure(
            tool_name=item["tool"],
            argument_pattern=item["pattern"],
            error_category=item["error_category"],
            error_text=item["error_desc"],
            recovery_action=item["recovery"],
        )

    # Seed canonical subgoal prerequisites
    subgoals = [
        ("build_project", "Build source artifacts", []),
        ("run_tests", "Execute automated test suite", ["build_project"]),
        ("deploy_service", "Deploy service to production", ["build_project", "run_tests"]),
        ("backup_database", "Create pre-migration database snapshot", []),
        ("migrate_database", "Execute database schema migration", ["backup_database"]),
    ]

    for sg_id, desc, prereqs in subgoals:
        causal_graph.add_node(SubgoalNode(
            subgoal_id=sg_id,
            name=sg_id.replace("_", " ").title(),
            description=desc,
            required_preconditions=prereqs,
        ))
        for p in prereqs:
            causal_graph.add_subgoal_dependency(p, sg_id, is_strict=True)

    return len(anti_patterns) + len(subgoals)


def seed_error_cascades(causal_graph: CausalErrorGraph) -> int:
    """Seed empirical error transition and cascade dynamics."""
    cascades = [
        ("planning_error", "tool_use_error", 2, 0.65),
        ("memory_error", "tool_use_error", 1, 0.85),
    ]

    for src, dst, lag, prob in cascades:
        causal_graph.record_error_cascade(src, dst, step_lag=lag, transition_prob=prob)

    return len(cascades)


def bootstrap_memory_graph(
    storage_path: Optional[str] = None,
    pattern_library_path: Optional[str] = None,
) -> CausalErrorGraph:
    """Orchestrate loading of all historical seeds into CausalErrorGraph."""
    graph = CausalErrorGraph(storage_path=storage_path)
    seed_from_pattern_library(graph, pattern_library_path)
    seed_canonical_anti_patterns(graph)
    seed_error_cascades(graph)
    graph.save()
    return graph
