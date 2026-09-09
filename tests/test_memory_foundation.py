"""Unit tests for Phase 1: Memory Foundation & Working Memory."""

import pytest

from longhorizon_guard.memory.schema import (
    ActionPatternNode,
    EdgeType,
    ErrorSignatureNode,
    MemoryAdvisory,
    NodeType,
    PrerequisiteEdge,
    PropagatesToEdge,
    RecoveryNode,
    RemediedByEdge,
    SubgoalNode,
    TaskConceptNode,
    TriggersErrorEdge,
)
from longhorizon_guard.memory.working_memory import WorkingMemory


class TestMemorySchema:
    """Validate typed schema data contracts and conversions."""

    def test_schema_node_and_edge_instantiation(self):
        # 1. Nodes
        task_node = TaskConceptNode("task_01", "Database Setup", "Initialize database", tags=["db", "sql"])
        subgoal_node = SubgoalNode("sub_01", "Create Schema", "Run migrations", required_preconditions=["db_running"])
        action_node = ActionPatternNode("act_01", "bash", "npm test", "bash:npm test")
        error_node = ErrorSignatureNode("err_01", "tool_use_error", r"command not found", "Tool missing")
        recovery_node = RecoveryNode("rec_01", "install tool", "Run apt install")

        assert task_node.node_type == NodeType.TASK_CONCEPT.value
        assert subgoal_node.node_type == NodeType.SUBGOAL.value
        assert action_node.node_type == NodeType.ACTION_PATTERN.value
        assert error_node.node_type == NodeType.ERROR_SIGNATURE.value
        assert recovery_node.node_type == NodeType.RECOVERY_ACTION.value

        # Serialization to dict
        t_dict = task_node.to_dict()
        assert t_dict["concept_id"] == "task_01"
        assert t_dict["tags"] == ["db", "sql"]

        # 2. Edges
        prereq = PrerequisiteEdge("sub_01", "sub_02", is_strict=True)
        triggers = TriggersErrorEdge("act_01", "err_01", frequency=3)
        propagates = PropagatesToEdge("err_01", "err_02", step_lag=4)
        remedied = RemediedByEdge("err_01", "rec_01", success_rate=0.95)

        assert prereq.edge_type == EdgeType.PREREQUISITE_OF.value
        assert triggers.edge_type == EdgeType.TRIGGERS_ERROR.value
        assert propagates.edge_type == EdgeType.PROPAGATES_TO.value
        assert remedied.edge_type == EdgeType.REMEDIED_BY.value
        assert remedied.to_dict()["success_rate"] == 0.95

    def test_memory_advisory_contract(self):
        advisory = MemoryAdvisory(
            advisory_type="predicted_error",
            severity="high",
            message="Tool 'bash' is known to fail with 'command not found'",
            confidence=0.88,
            source_node_id="act_01",
            target_node_id="err_01",
            recovery_suggestion="Verify PATH before calling bash",
        )
        data = advisory.to_dict()
        assert data["advisory_type"] == "predicted_error"
        assert data["severity"] == "high"
        assert data["recovery_suggestion"] == "Verify PATH before calling bash"


class TestWorkingMemory:
    """Validate session working memory state tracking and sliding windows."""

    def test_working_memory_lifecycle(self):
        wm = WorkingMemory(session_id="sess_123", max_window=5)
        wm.init_session("Deploy app", "1. Build\n2. Test", metadata={"run_id": "test_run"})

        assert wm.session_id == "sess_123"
        assert wm.task_description == "Deploy app"
        assert len(wm.history) == 0

        # Record clean step
        wm.record_step({"step_index": 0, "action_name": "git", "action_args": {"cmd": "status"}}, has_error=False)
        assert len(wm.history) == 1
        assert wm.get_action_repeat_count("git", {"cmd": "status"}) == 1
        assert wm.get_action_repeat_count("git", {"cmd": "status"}, failed_only=True) == 0

        # Record failing step
        wm.record_step({"step_index": 1, "action_name": "run", "action_args": {"cmd": "build"}}, has_error=True)
        assert len(wm.history) == 2
        assert wm.tool_failure_counts["run"] == 1
        assert wm.get_action_repeat_count("run", {"cmd": "build"}, failed_only=True) == 1

        # Re-initialize cleans up working state
        wm.init_session("New Task")
        assert len(wm.history) == 0
        assert len(wm.recent_actions) == 0
        assert len(wm.tool_failure_counts) == 0

    def test_sliding_window_max_size(self):
        wm = WorkingMemory(session_id="window_test", max_window=3)
        for i in range(5):
            wm.record_step({"step_index": i, "action_name": "read", "action_args": {"file": f"{i}.txt"}})

        assert len(wm.recent_actions) == 3
        # Should retain items 2, 3, 4
        assert wm.get_action_repeat_count("read", {"file": "0.txt"}) == 0
        assert wm.get_action_repeat_count("read", {"file": "4.txt"}) == 1

    def test_drift_and_advisories_tracking(self):
        wm = WorkingMemory(session_id="drift_test")
        wm.record_drift(step_index=0, drift_score=0.15, velocity=0.05, acceleration=0.01)
        wm.record_drift(step_index=1, drift_score=0.45, velocity=0.30, acceleration=0.25)

        latest = wm.get_latest_drift()
        assert latest is not None
        assert latest["step_index"] == 1.0
        assert latest["drift_score"] == 0.45
        assert latest["velocity"] == 0.30

        adv = MemoryAdvisory(
            advisory_type="prerequisite_violation",
            severity="critical",
            message="Prerequisite missing",
            confidence=0.95,
        )
        wm.add_advisory(adv)
        assert len(wm.advisories) == 1

        summary = wm.to_dict()
        assert summary["advisories_count"] == 1
        assert summary["total_steps"] == 0
        assert len(summary["drift_trajectory"]) == 2
