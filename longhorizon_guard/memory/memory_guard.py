"""Unified Memory Guard Coordinator for LongHorizon Guard.

Integrates WorkingMemory, CausalErrorGraph, AssociativeMemoryEngine, and
LocalConceptIndex into a single high-level interface wired into GuardInterface.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from longhorizon_guard.memory.associative_engine import AssociativeMemoryEngine
from longhorizon_guard.memory.capability_classifier import (
    classify_action_capability,
    classify_subgoal_category,
)
from longhorizon_guard.memory.causal_graph import CausalErrorGraph, _generate_id
from longhorizon_guard.memory.schema import MemoryAdvisory
from longhorizon_guard.memory.seed_loader import bootstrap_memory_graph
from longhorizon_guard.memory.vector_index import LocalConceptIndex
from longhorizon_guard.memory.working_memory import WorkingMemory

logger = logging.getLogger("longhorizon_guard.memory.memory_guard")


class MemoryGuard:
    """Coordinates working memory, causal graph reasoning, and associative retrieval."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_bootstrap: bool = True,
    ) -> None:
        self.storage_path = storage_path
        is_memory_db = bool(storage_path == ":memory:")
        needs_bootstrap = auto_bootstrap and (not storage_path or is_memory_db or (not is_memory_db and not os.path.exists(storage_path)))
        if needs_bootstrap:
            self.causal_graph = bootstrap_memory_graph(storage_path=storage_path)
        else:
            self.causal_graph = CausalErrorGraph(storage_path=storage_path)

        self.associative_engine = AssociativeMemoryEngine(self.causal_graph)
        self.concept_index = LocalConceptIndex()
        self.working_memory = WorkingMemory()

    def on_plan_proposed(
        self,
        task_description: str,
        proposed_plan: str,
        declared_subgoals: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate proposed plan against prerequisite dependencies and past task memories."""
        self.working_memory.init_session(task_description, proposed_plan, metadata)
        advisories: List[MemoryAdvisory] = []
        subgoals = declared_subgoals or []

        # 1. Prerequisite Sequence Verification
        prior_subgoal_names: List[str] = []
        # If no subgoals were passed from tracker, extract from proposed_plan
        if not subgoals and proposed_plan:
            import re
            lines = [ln.strip() for ln in proposed_plan.splitlines() if ln.strip()]
            for idx, ln in enumerate(lines):
                cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", ln).strip()
                if cleaned:
                    subgoals.append({"subgoal_id": f"subgoal_{idx+1:03d}", "description": cleaned})

        for sg in subgoals:
            sg_id = str(sg.get("subgoal_id", "")).strip().lower()
            sg_desc = str(sg.get("description", "")).strip().lower()
            sg_cat = classify_subgoal_category(sg_desc)
            target_query = sg_cat if sg_cat else f"{sg_id} {sg_desc}".strip()

            # Check if this subgoal has unmet prerequisites in the causal graph
            is_valid, missing = self.causal_graph.check_subgoal_prerequisites(prior_subgoal_names, target_query)
            if not is_valid and missing:
                adv = MemoryAdvisory(
                    advisory_type="prerequisite_violation",
                    severity="high",
                    message=(
                        f"Subgoal '{sg_desc or sg_id}' is scheduled before its required "
                        f"prerequisite(s): {', '.join(missing)}"
                    ),
                    confidence=0.92,
                    source_node_id=missing[0],
                    target_node_id=sg_id,
                    recovery_suggestion=f"Schedule '{missing[0]}' before '{sg_desc or sg_id}' in the execution plan.",
                )
                advisories.append(adv)
                self.working_memory.add_advisory(adv)

            prior_subgoal_names.append(sg_id)
            if sg_cat:
                prior_subgoal_names.append(sg_cat)
            if sg_desc:
                prior_subgoal_names.append(sg_desc)

        # 2. Similar Task Concept Search
        similar_tasks = self.concept_index.search(task_description, top_k=2, min_similarity=0.35)
        for sim in similar_tasks:
            meta = sim.get("metadata", {})
            if meta.get("known_pitfall"):
                adv = MemoryAdvisory(
                    advisory_type="past_failure_pattern",
                    severity="medium",
                    message=f"Similar past task encountered pitfall: {meta['known_pitfall']}",
                    confidence=sim["similarity"],
                    recovery_suggestion=meta.get("recommended_strategy"),
                )
                advisories.append(adv)
                self.working_memory.add_advisory(adv)

        flags = [a.message for a in advisories]
        suggestions = [a.recovery_suggestion for a in advisories if a.recovery_suggestion]

        return {
            "flagged": len(flags) > 0,
            "flags": flags,
            "suggestions": suggestions,
            "advisories": [a.to_dict() for a in advisories],
        }

    def on_pre_step(
        self,
        step_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inspect planned action for immediate loops and associative causal error risks."""
        import json
        tool_name = str(step_record.get("action_name", "")).strip()
        tool_args = step_record.get("action_args", {})

        flags: List[str] = []
        suggestions: List[str] = []

        # 1. Check working memory action repetition count
        repeat_count = self.working_memory.get_action_repeat_count(tool_name, tool_args, failed_only=True)
        if repeat_count >= 2:
            flags.append(
                f"[memory:loop_risk] Action '{tool_name}' with identical arguments failed {repeat_count} times recently."
            )
            # Find recovery for this tool
            recoveries = self.causal_graph.find_recovery_paths(tool_name)
            if recoveries:
                suggestions.append(recoveries[0]["recovery_suggestion"])

        # 2. Associative Multi-Hop Error Risk (Personalized PageRank)
        matched_act_ids = set()
        candidate_sig = f"{tool_name}:{tool_args}"
        direct_id = _generate_id("act", candidate_sig)
        if direct_id in self.causal_graph.graph:
            matched_act_ids.add(direct_id)

        # Extract arg string representation and dict values
        arg_str = ""
        arg_values = []
        if isinstance(tool_args, dict):
            arg_str = json.dumps(tool_args)
            arg_values = [str(v) for v in tool_args.values()]
        elif isinstance(tool_args, str):
            arg_str = tool_args
            arg_values = [tool_args]
        else:
            arg_str = str(tool_args)

        from longhorizon_guard.memory.schema import NodeType
        tool_clean = tool_name.strip().lower()
        tool_cap = classify_action_capability(tool_name)
        for nid, data in self.causal_graph.graph.nodes(data=True):
            if data.get("node_type") == NodeType.ACTION_PATTERN.value:
                node_tool = str(data.get("tool_name", "")).strip().lower()
                node_cap = classify_action_capability(node_tool)
                if tool_cap is not None and node_cap is not None:
                    tool_matches = (tool_cap == node_cap)
                else:
                    tool_matches = (node_tool == tool_clean)

                if tool_matches:
                    node_pat = str(data.get("argument_pattern", "")).strip().lower()
                    if node_pat:
                        if (
                            node_pat in arg_str.lower()
                            or any(node_pat in v.lower() for v in arg_values)
                            or any(v.lower() in node_pat for v in arg_values)
                        ):
                            matched_act_ids.add(nid)

        top_risk_cat = None
        top_risk_score = 0.85
        if matched_act_ids:
            risks = self.associative_engine.get_associative_risks(list(matched_act_ids), top_k=2, min_score=0.01)
            if risks:
                top_risk = risks[0]
                top_risk_cat = top_risk.get("category", "tool_use_error")
                top_risk_score = top_risk.get("associative_score", 0.85)
                flags.append(
                    f"[memory:associative_risk] Action '{tool_name}' has high associative causal link "
                    f"to '{top_risk['category']}' error: {top_risk['description']} (PPR={top_risk['associative_score']:.2f})"
                )
            recs = self.associative_engine.get_associative_recoveries(list(matched_act_ids), top_k=2)
            for rec in recs:
                if rec.get("safe_alternative") and rec["safe_alternative"] not in suggestions:
                    suggestions.append(rec["safe_alternative"])

        return {
            "flagged": len(flags) > 0,
            "flags": flags,
            "suggestions": suggestions,
            "category": top_risk_cat or "tool_use_error",
            "confidence": top_risk_score,
        }

    def on_post_step(
        self,
        step_record: Dict[str, Any],
        step_result: Dict[str, Any],
        drift_score: float = 0.0,
        velocity: float = 0.0,
        acceleration: float = 0.0,
    ) -> None:
        """Update working memory and record newly observed failures into the Causal Graph."""
        has_error = bool(step_result.get("flagged")) or any(
            err in str(step_record.get("tool_response", "")).lower()
            for err in ("error", "failed", "traceback", "exception")
        )
        self.working_memory.record_step(step_record, has_error=has_error)
        step_idx = int(step_record.get("step_index", len(self.working_memory.history)))
        self.working_memory.record_drift(step_idx, drift_score, velocity, acceleration)

        # If step experienced a confirmed failure, learn live into Causal Graph
        if has_error and step_result.get("category"):
            tool_name = str(step_record.get("action_name", "tool"))
            tool_args = str(step_record.get("action_args", {}))
            err_cat = str(step_result.get("category"))
            err_text = str(step_record.get("tool_response", ""))[:120]
            safe_alt = step_result.get("suggestions", [None])[0] if step_result.get("suggestions") else None

            self.causal_graph.record_action_failure(
                tool_name=tool_name,
                argument_pattern=tool_args,
                error_category=err_cat,
                error_text=err_text,
                recovery_action=safe_alt,
            )

    def on_run_end(
        self,
        run_summary: Dict[str, Any],
    ) -> None:
        """Consolidate session outcome, error cascades, and persist updated graph."""
        root_cause_type = run_summary.get("root_cause_error_type")
        root_cause_step = run_summary.get("root_cause_step_index")

        # If trajectory suffered a root cause error, record cascade relation
        if root_cause_type and root_cause_step is not None:
            # Check if there was downstream drift or tool error
            if len(self.working_memory.drift_trajectory) > 0:
                latest_drift = self.working_memory.get_latest_drift()
                if latest_drift and latest_drift.get("drift_score", 0.0) >= 0.50:
                    self.causal_graph.record_error_cascade(
                        root_error_id_or_cat=root_cause_type,
                        downstream_error_id_or_cat="drift",
                        step_lag=max(1, len(self.working_memory.history) - root_cause_step),
                    )

        # Index task description in concept index
        task_desc = self.working_memory.task_description
        if task_desc:
            cid = _generate_id("task", task_desc)
            self.concept_index.add_concept(cid, task_desc, metadata={"run_summary": run_summary})

        # Persist graph if a real path was specified
        if self.causal_graph.storage_path and self.causal_graph.storage_path != ":memory:":
            try:
                self.causal_graph.save()
            except Exception as exc:
                logger.debug("Failed to auto-save causal graph: %s", exc)
