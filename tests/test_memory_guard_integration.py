"""Integration tests for Phase 5: GuardInterface with Causal Memory Engine."""

import pytest

from longhorizon_guard.config import GuardConfig
from longhorizon_guard.interface import GuardInterface


class TestMemoryGuardIntegration:
    """Validate GuardInterface wired with the Causal Error Memory Graph."""

    def test_memory_disabled_by_default_is_backward_compatible(self):
        """Default GuardConfig maintains enable_memory=False and zero memory overhead."""
        guard = GuardInterface()
        assert guard.enable_memory is False
        assert guard.memory_guard is None

        # Standard plan check works without memory
        res = guard.on_plan_proposed("Standard Task", "1. Step A\n2. Step B")
        assert res["approved"] is True
        assert res["flagged"] is False

    def test_memory_enabled_initialization(self):
        """GuardConfig with enable_memory=True instantiates active MemoryGuard."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:")
        guard = GuardInterface(config=cfg)
        assert guard.enable_memory is True
        assert guard.memory_guard is not None
        assert guard.memory_guard.causal_graph.graph.number_of_nodes() > 10

    def test_plan_proposed_catches_prerequisite_violation(self):
        """on_plan_proposed flags missing prerequisite subgoals using causal memory graph."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:")
        guard = GuardInterface(config=cfg)

        # Plan targets deployment directly without running tests or build
        bad_plan = "1. Deploy service to production directly"
        res = guard.on_plan_proposed("Production deployment task", bad_plan)

        assert res["flagged"] is True
        assert any("prerequisite" in f.lower() for f in res["flags"])
        assert len(res["suggestions"]) >= 1

    def test_on_step_memory_associative_enrichment(self):
        """on_step associative memory provides recovery suggestion for known risk actions."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:")
        guard = GuardInterface(config=cfg)

        # Step invokes known anti-pattern tool
        step_rec = {
            "step_index": 0,
            "action_name": "bash",
            "action_args": {"cmd": "rm -rf /"},
            "tool_response": "Permission denied: rm: cannot remove '/'",
        }
        res = guard.on_step(step_rec, history=[])
        assert res["flagged"] is True
        # Verify recovery suggestion attached
        assert any("safe trash" in s.lower() or "target" in s.lower() for s in res.get("suggestions", []))

    def test_on_run_end_live_memory_consolidation(self):
        """on_run_end records newly observed error and cascade into Causal Graph."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:")
        guard = GuardInterface(config=cfg)

        step_0 = {
            "step_index": 0,
            "action_name": "custom_api_tool",
            "action_args": {"endpoint": "/v1/bad"},
            "tool_response": "500 Internal Server Error",
        }
        guard.on_step(step_0, history=[])

        # Run end summary
        summary = guard.on_run_end(
            metadata={"run_id": "mem_learn_test", "task_description": "API Integration"},
            trajectory={"steps": [step_0]},
        )
        assert summary is not None

        # Verify new action node was dynamically added to Causal Graph
        mg = guard.memory_guard
        assert mg is not None
        summary_g = mg.causal_graph.summary()
        assert summary_g["total_nodes"] > 10

    def test_memory_flag_does_not_suppress_other_detection_layers(self):
        """Regression test: verify memory flags do not cause KeyError or suppress matcher/drift/reflector."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:", drift_threshold=0.01)
        guard = GuardInterface(config=cfg)
        guard.on_plan_proposed("task", "1. step a\n2. step b")

        step_rec = {
            "action_name": "bash",
            "action_args": {"cmd": "rm -rf /"},
            "tool_response": "Permission denied: rm: cannot remove '/'",
            "reasoning": "trying again after previous failure, stuck in a loop, repeating same action",
        }
        res = guard.on_step(step_rec, history=[])

        assert res.get("memory_flagged") is True
        assert res.get("match_details") is not None
        assert res.get("drift_assessment") is not None
        assert res.get("reflection_result") is not None
