"""Unit tests for Phase 2: Causal Error Knowledge Graph Engine."""

import os
import tempfile
import pytest

from longhorizon_guard.memory.causal_graph import CausalErrorGraph
from longhorizon_guard.memory.schema import (
    ActionPatternNode,
    EdgeType,
    ErrorSignatureNode,
    NodeType,
    SubgoalNode,
)


class TestCausalErrorGraph:
    """Validate graph creation, traversal, cycle checking, and persistence."""

    def test_node_and_edge_addition(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        sub_a = SubgoalNode(subgoal_id="sub_a", name="Backup DB", description="Create backup")
        sub_b = SubgoalNode(subgoal_id="sub_b", name="Migrate DB", description="Run schema migration")

        aid = graph.add_node(sub_a)
        bid = graph.add_node(sub_b)
        assert aid == "sub_a"
        assert bid == "sub_b"

        graph.add_subgoal_dependency("sub_a", "sub_b", is_strict=True, description="Backup required before migration")
        assert graph.graph.has_edge("sub_a", "sub_b")

        summary = graph.summary()
        assert summary["total_nodes"] == 2
        assert summary["total_edges"] == 1
        assert summary["nodes_by_type"][NodeType.SUBGOAL.value] == 2
        assert summary["edges_by_type"][EdgeType.PREREQUISITE_OF.value] == 1

    def test_subgoal_prerequisite_validation(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        graph.add_node(SubgoalNode(subgoal_id="prep_env", name="Setup Environment", description="Install deps"))
        graph.add_node(SubgoalNode(subgoal_id="run_tests", name="Run Test Suite", description="Execute pytest"))
        graph.add_subgoal_dependency("prep_env", "run_tests", is_strict=True)

        # 1. Prerequisite missing -> fails
        valid, missing = graph.check_subgoal_prerequisites([], "run_tests")
        assert valid is False
        assert "prep_env" in missing or "Setup Environment" in missing

        # 2. Prerequisite satisfied -> passes
        valid, missing = graph.check_subgoal_prerequisites(["prep_env"], "run_tests")
        assert valid is True
        assert len(missing) == 0

    def test_record_action_failure_and_recovery(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        act_id, err_id = graph.record_action_failure(
            tool_name="bash",
            argument_pattern="grep 'test' file.txt",
            error_category="tool_use_error",
            error_text="grep: unrecognized option on windows powershell",
            recovery_action="Use Select-String instead of grep on Windows",
        )

        assert act_id.startswith("act_")
        assert err_id.startswith("err_")

        # Query recovery paths
        recoveries = graph.find_recovery_paths("bash", error_category="tool_use_error")
        assert len(recoveries) >= 1
        rec = recoveries[0]
        assert rec["tool_name"] == "bash"
        assert rec["error_category"] == "tool_use_error"
        assert "Select-String" in rec["recovery_suggestion"]

    def test_error_cascade_recording(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        u, v = graph.record_error_cascade("planning_error", "tool_use_error", step_lag=3, transition_prob=0.65)
        assert graph.graph.has_edge(u, v)

        # Re-recording increments transition probability
        graph.record_error_cascade("planning_error", "tool_use_error")
        prob = graph.graph[u][v]["transition_prob"]
        assert prob >= 0.70

    def test_cycle_detection(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        graph.add_subgoal_dependency("step_1", "step_2")
        graph.add_subgoal_dependency("step_2", "step_3")
        assert len(graph.detect_cycles()) == 0

        # Create cycle
        graph.add_subgoal_dependency("step_3", "step_1")
        cycles = graph.detect_cycles()
        assert len(cycles) > 0

    def test_graph_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            g1 = CausalErrorGraph(storage_path=temp_path)
            g1.add_node(SubgoalNode(subgoal_id="sg1", name="Init", description="Desc 1"))
            g1.add_node(SubgoalNode(subgoal_id="sg2", name="Exec", description="Desc 2"))
            g1.add_subgoal_dependency("sg1", "sg2")
            g1.save()

            # Load into fresh graph
            g2 = CausalErrorGraph(storage_path=temp_path)
            assert g2.graph.number_of_nodes() == 2
            assert g2.graph.number_of_edges() == 1
            assert g2.graph.has_edge("sg1", "sg2")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
