"""Causal Error Knowledge Graph Engine for LongHorizon Guard.

Utilizes NetworkX DiGraph to model causal relationships between tools, errors,
subgoals, and recovery actions with local JSON/SQLite persistence.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx

from longhorizon_guard.memory.schema import (
    ActionPatternNode,
    EdgeType,
    ErrorSignatureNode,
    NodeType,
    PrerequisiteEdge,
    PropagatesToEdge,
    RecoveryNode,
    RemediedByEdge,
    SubgoalNode,
    TaskConceptNode,
    TriggersErrorEdge,
)

logger = logging.getLogger("longhorizon_guard.memory.causal_graph")

DEFAULT_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "findings",
    "memory",
)
DEFAULT_GRAPH_FILE = os.path.join(DEFAULT_MEMORY_DIR, "causal_graph.json")


def _generate_id(prefix: str, content: str) -> str:
    """Generate a stable, deterministic node/edge identifier."""
    h = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


class CausalErrorGraph:
    """Directed Knowledge Graph modeling causal failure dynamics and recovery paths."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self.storage_path = storage_path or DEFAULT_GRAPH_FILE
        self.graph: nx.DiGraph = nx.DiGraph()
        if self.storage_path != ":memory:" and os.path.exists(self.storage_path):
            try:
                self.load(self.storage_path)
            except Exception as exc:
                logger.warning("Failed to load existing graph from %s: %s", self.storage_path, exc)

    def add_node(
        self,
        node: Union[
            TaskConceptNode,
            SubgoalNode,
            ActionPatternNode,
            ErrorSignatureNode,
            RecoveryNode,
            Dict[str, Any],
        ],
    ) -> str:
        """Add a typed node to the graph and return its ID."""
        if hasattr(node, "to_dict"):
            data = node.to_dict()
        else:
            data = dict(node)

        node_id = (
            data.get("concept_id")
            or data.get("subgoal_id")
            or data.get("action_id")
            or data.get("error_id")
            or data.get("recovery_id")
            or data.get("id")
        )
        if not node_id:
            raise ValueError(f"Node missing identifiable ID field: {data}")

        self.graph.add_node(node_id, **data)
        return node_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node attributes by ID."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def add_edge(
        self,
        edge: Union[
            PrerequisiteEdge,
            TriggersErrorEdge,
            PropagatesToEdge,
            RemediedByEdge,
            Dict[str, Any],
        ],
    ) -> Tuple[str, str]:
        """Add a typed relational edge between two existing or newly created nodes."""
        if hasattr(edge, "to_dict"):
            data = edge.to_dict()
        else:
            data = dict(edge)

        source = data.get("source_id")
        target = data.get("target_id")
        if not source or not target:
            raise ValueError(f"Edge missing source_id or target_id: {data}")

        # Ensure source and target exist as generic nodes if not already added
        if source not in self.graph:
            self.graph.add_node(source, id=source, node_type="unknown")
        if target not in self.graph:
            self.graph.add_node(target, id=target, node_type="unknown")

        self.graph.add_edge(source, target, **data)
        return source, target

    def add_subgoal_dependency(
        self,
        from_subgoal_id: str,
        to_subgoal_id: str,
        is_strict: bool = True,
        description: str = "",
    ) -> None:
        """Record that to_subgoal_id requires from_subgoal_id as a prerequisite."""
        edge = PrerequisiteEdge(
            source_id=from_subgoal_id,
            target_id=to_subgoal_id,
            is_strict=is_strict,
            description=description,
        )
        self.add_edge(edge)

    def check_subgoal_prerequisites(
        self,
        completed_or_active_subgoals: List[str],
        proposed_subgoal_id: str,
    ) -> Tuple[bool, List[str]]:
        """Verify if all strict prerequisite subgoals are satisfied.

        Returns:
            (is_valid, list_of_missing_prerequisites)
        """
        target_node_id = None
        if proposed_subgoal_id in self.graph:
            target_node_id = proposed_subgoal_id
        else:
            p_clean = proposed_subgoal_id.lower().replace("_", " ")
            for n, data in self.graph.nodes(data=True):
                if data.get("node_type") == NodeType.SUBGOAL.value:
                    n_clean = str(n).lower().replace("_", " ")
                    name_clean = str(data.get("name", "")).lower().replace("_", " ")
                    if (n_clean and n_clean in p_clean) or (name_clean and name_clean in p_clean):
                        target_node_id = n
                        break

        if not target_node_id or target_node_id not in self.graph:
            return True, []

        completed_set = {s.lower().strip() for s in completed_or_active_subgoals}
        missing: List[str] = []

        # Find in-edges that are prerequisites pointing to target_node_id
        for u, v, data in self.graph.in_edges(target_node_id, data=True):
            if data.get("edge_type") == EdgeType.PREREQUISITE_OF.value:
                if data.get("is_strict", True):
                    u_node = self.graph.nodes.get(u, {})
                    u_id = str(u).lower().strip()
                    u_name = str(u_node.get("name", "")).lower().strip()
                    u_desc = str(u_node.get("description", "")).lower().strip()

                    satisfied = (
                        u_id in completed_set
                        or (u_name and u_name in completed_set)
                        or (u_desc and any(u_desc in c or c in u_desc for c in completed_set))
                    )
                    if not satisfied:
                        missing.append(u_node.get("name") or u)

        return len(missing) == 0, missing

    def record_action_failure(
        self,
        tool_name: str,
        argument_pattern: str,
        error_category: str,
        error_text: str,
        recovery_action: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Record an observed tool failure and causal error edge."""
        tool_clean = tool_name.strip()
        arg_clean = argument_pattern.strip()
        action_sig = f"{tool_clean}:{arg_clean}"
        act_id = _generate_id("act", action_sig)

        if act_id not in self.graph:
            self.add_node(ActionPatternNode(
                action_id=act_id,
                tool_name=tool_clean,
                argument_pattern=arg_clean,
                normalized_signature=action_sig,
            ))

        err_id = _generate_id("err", f"{error_category}:{error_text[:60]}")
        if err_id not in self.graph:
            self.add_node(ErrorSignatureNode(
                error_id=err_id,
                category=error_category,
                pattern_regex=error_text[:100],
                description=error_text[:200],
            ))

        # Check existing edge or create new one
        if self.graph.has_edge(act_id, err_id):
            edge_data = self.graph[act_id][err_id]
            edge_data["frequency"] = edge_data.get("frequency", 1) + 1
            if error_text not in edge_data.get("sample_errors", []):
                edge_data.setdefault("sample_errors", []).append(error_text)
        else:
            self.add_edge(TriggersErrorEdge(
                source_id=act_id,
                target_id=err_id,
                frequency=1,
                sample_errors=[error_text],
            ))

        # Connect recovery action if available
        if recovery_action:
            rec_id = _generate_id("rec", recovery_action)
            if rec_id not in self.graph:
                self.add_node(RecoveryNode(
                    recovery_id=rec_id,
                    action_description=recovery_action,
                    safe_alternative=recovery_action,
                ))

            if self.graph.has_edge(err_id, rec_id):
                r_data = self.graph[err_id][rec_id]
                r_data["times_applied"] = r_data.get("times_applied", 1) + 1
            else:
                self.add_edge(RemediedByEdge(
                    source_id=err_id,
                    target_id=rec_id,
                    success_rate=1.0,
                    times_applied=1,
                ))

        return act_id, err_id

    def record_error_cascade(
        self,
        root_error_id_or_cat: str,
        downstream_error_id_or_cat: str,
        step_lag: int = 1,
        transition_prob: float = 0.5,
    ) -> Tuple[str, str]:
        """Record an observed error transition or cascade."""
        u = _generate_id("err_cat", root_error_id_or_cat)
        v = _generate_id("err_cat", downstream_error_id_or_cat)

        if u not in self.graph:
            self.add_node(ErrorSignatureNode(
                error_id=u,
                category=root_error_id_or_cat,
                pattern_regex=root_error_id_or_cat,
                description=f"Error category {root_error_id_or_cat}",
            ))
        if v not in self.graph:
            self.add_node(ErrorSignatureNode(
                error_id=v,
                category=downstream_error_id_or_cat,
                pattern_regex=downstream_error_id_or_cat,
                description=f"Error category {downstream_error_id_or_cat}",
            ))

        if self.graph.has_edge(u, v):
            e_data = self.graph[u][v]
            e_data["transition_prob"] = min(0.99, e_data.get("transition_prob", 0.5) + 0.05)
        else:
            self.add_edge(PropagatesToEdge(
                source_id=u,
                target_id=v,
                step_lag=step_lag,
                transition_prob=transition_prob,
            ))

        return u, v

    def find_recovery_paths(self, tool_name: str, error_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover direct and 2-hop recovery paths for a tool or error category."""
        paths = []
        tool_clean = tool_name.strip().lower()

        # Iterate over matching action nodes
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.ACTION_PATTERN.value:
                if str(data.get("tool_name", "")).lower() == tool_clean:
                    # Look for triggers_error out-edges
                    for _, err_id, err_edge in self.graph.out_edges(node_id, data=True):
                        err_node = self.graph.nodes.get(err_id, {})
                        if error_category and err_node.get("category") != error_category:
                            continue
                        # Look for remedied_by out-edges from error
                        for _, rec_id, rec_edge in self.graph.out_edges(err_id, data=True):
                            rec_node = self.graph.nodes.get(rec_id, {})
                            paths.append({
                                "action_id": node_id,
                                "tool_name": data.get("tool_name"),
                                "error_category": err_node.get("category"),
                                "error_pattern": err_node.get("pattern_regex"),
                                "recovery_suggestion": rec_node.get("safe_alternative"),
                                "success_rate": rec_edge.get("success_rate", 1.0),
                            })
        return paths

    def detect_cycles(self) -> List[List[str]]:
        """Check for cycles in prerequisite relationships."""
        prereq_subgraph = nx.DiGraph([
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == EdgeType.PREREQUISITE_OF.value
        ])
        try:
            return list(nx.simple_cycles(prereq_subgraph))
        except Exception:
            return []

    def save(self, path: Optional[str] = None) -> str:
        """Persist graph structure and attributes to disk."""
        target = path or self.storage_path
        if not target or target == ":memory:":
            return ":memory:"

        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)

        data = {
            "version": "1.0",
            "nodes": [
                {"id": n, **attrs} for n, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **attrs} for u, v, attrs in self.graph.edges(data=True)
            ],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target

    def load(self, path: Optional[str] = None) -> None:
        """Load graph structure from disk."""
        source = path or self.storage_path
        if not source or source == ":memory:" or not os.path.exists(source):
            return

        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        for n in data.get("nodes", []):
            nid = n.pop("id", None)
            if nid:
                self.graph.add_node(nid, **n)

        for e in data.get("edges", []):
            src = e.pop("source", None)
            tgt = e.pop("target", None)
            if src and tgt:
                self.graph.add_edge(src, tgt, **e)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of node and edge distributions."""
        node_counts: Dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            nt = d.get("node_type", "unknown")
            node_counts[nt] = node_counts.get(nt, 0) + 1

        edge_counts: Dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_counts[et] = edge_counts.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts,
        }
