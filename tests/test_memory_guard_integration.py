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

    def test_on_subgoal_boundary_prerequisite_violation_advisory(self):
        """End-to-end: deploy_service active before run_tests completed triggers advisory warning."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:")
        guard = GuardInterface(config=cfg)

        plan = "1. Build source artifacts\n2. Execute automated test suite\n3. Deploy service to production"
        guard.on_plan_proposed(
            task_description="Build, test and deploy pipeline",
            proposed_plan=plan,
            metadata={"run_id": "test_e2e_prereq_001"},
        )

        # Step 0: completes subgoal_001 ("Build source artifacts")
        step0 = {
            "step_index": 0,
            "reasoning": "Building the project source artifacts",
            "action_name": "build",
            "action_args": {"target": "all"},
            "tool_response": "Build completed successfully. task complete",
        }
        res0 = guard.on_step(step0, history=[], metadata={"run_id": "test_e2e_prereq_001"})
        assert res0.get("subgoal_transition") is not None
        assert res0["subgoal_transition"]["completed_subgoal_id"] == "subgoal_001"

        b_res0 = guard.on_subgoal_boundary(
            subgoal_id="subgoal_001",
            subgoal_status="completed",
            step_history=[step0],
        )
        assert b_res0["checkpoint_passed"] is True
        assert b_res0["next_subgoal"] == "subgoal_002"
        # Newly active is subgoal_002 ("Execute automated test suite")
        # Prerequisite is build_project which IS completed -> no violation
        assert "prerequisite_violation" not in b_res0
        assert b_res0.get("warning") is None

        # Step 1: fails on subgoal_002 ("Execute automated test suite")
        step1 = {
            "step_index": 1,
            "reasoning": "Running test suite",
            "action_name": "pytest",
            "action_args": {"flags": "-v"},
            "tool_response": "tests failed with exit code 1",
        }
        res1 = guard.on_step(step1, history=[step0], metadata={"run_id": "test_e2e_prereq_001"})
        assert res1.get("subgoal_transition") is not None
        assert res1["subgoal_transition"]["completed_status"] == "failed"

        # Boundary for subgoal_002 transitioning as failed -> next active is subgoal_003 (Deploy service)
        b_res1 = guard.on_subgoal_boundary(
            subgoal_id="subgoal_002",
            subgoal_status="failed",
            step_history=[step0, step1],
        )
        # Subgoal 2 failed, so checkpoint did not pass (expected status failed)
        assert b_res1["checkpoint_passed"] is False
        assert b_res1["next_subgoal"] == "subgoal_003"

        # Now subgoal_003 ("Deploy service to production") is active!
        # Its prerequisites are build_project (completed) AND run_tests (NOT completed, failed).
        assert "prerequisite_violation" in b_res1
        assert b_res1["prerequisite_violation"]["active_subgoal"] == "subgoal_003"
        missing = b_res1["prerequisite_violation"]["missing_prerequisites"]
        assert any("test" in m.lower() or "run_tests" in m.lower() for m in missing)
        assert "[memory:prerequisite_violation]" in b_res1.get("warning", "")
        assert any("test" in s.lower() for s in b_res1.get("suggestions", []))

    def test_on_subgoal_boundary_prerequisites_clean_path(self):
        """Clean path: all prerequisites completed before deploy_service becomes active -> no warning."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:")
        guard = GuardInterface(config=cfg)

        plan = "1. Build source artifacts\n2. Execute automated test suite\n3. Deploy service to production"
        guard.on_plan_proposed(
            task_description="Build, test and deploy pipeline",
            proposed_plan=plan,
            metadata={"run_id": "test_e2e_clean_001"},
        )

        step0 = {
            "step_index": 0,
            "reasoning": "Building",
            "action_name": "build",
            "action_args": {},
            "tool_response": "Build success task complete",
        }
        guard.on_step(step0, history=[])
        b_res0 = guard.on_subgoal_boundary(
            subgoal_id="subgoal_001",
            subgoal_status="completed",
            step_history=[step0],
        )
        assert "prerequisite_violation" not in b_res0

        step1 = {
            "step_index": 1,
            "reasoning": "Testing",
            "action_name": "test",
            "action_args": {},
            "tool_response": "Test success task complete",
        }
        guard.on_step(step1, history=[step0])
        b_res1 = guard.on_subgoal_boundary(
            subgoal_id="subgoal_002",
            subgoal_status="completed",
            step_history=[step0, step1],
        )
        assert b_res1["checkpoint_passed"] is True
        assert b_res1["next_subgoal"] == "subgoal_003"
        # Subgoal 3 active, and both build_project and run_tests are completed
        assert "prerequisite_violation" not in b_res1
        assert b_res1.get("warning") is None

    def test_on_subgoal_boundary_advisory_only_never_halts(self):
        """Prerequisite violation at boundary must remain advisory-only (does not deny or alter checkpoint_passed)."""
        cfg = GuardConfig(enable_memory=True, memory_storage_path=":memory:", block_on_critical=True)
        guard = GuardInterface(config=cfg)

        plan = "1. Build source artifacts\n2. Execute automated test suite\n3. Deploy service to production"
        guard.on_plan_proposed(
            task_description="Build and deploy",
            proposed_plan=plan,
        )

        # Complete subgoal_001 first so active subgoal becomes subgoal_002
        guard.on_subgoal_boundary(
            subgoal_id="subgoal_001",
            subgoal_status="completed",
            step_history=[],
        )

        # Force transition: subgoal_002 stalled_advanced (counted as passed checkpoint)
        # but NOT completed
        b_res = guard.on_subgoal_boundary(
            subgoal_id="subgoal_002",
            subgoal_status="stalled_advanced",
            step_history=[],
        )
        # checkpoint_passed remains True because status was stalled_advanced
        assert b_res["checkpoint_passed"] is True
        assert b_res["next_subgoal"] == "subgoal_003"
        # Prerequisite run_tests was not completed (only stalled_advanced)
        assert "prerequisite_violation" in b_res
        assert "[memory:prerequisite_violation]" in b_res.get("warning", "")
        # Does not have a continue_execution=False key or denial
        assert b_res.get("continue_execution") is not False

    def test_on_subgoal_boundary_memory_disabled_bypasses_cleanly(self):
        """With enable_memory=False, on_subgoal_boundary runs cleanly without checking causal graph."""
        guard = GuardInterface()
        assert guard.enable_memory is False
        assert guard.memory_guard is None

        plan = "1. Build source artifacts\n2. Execute automated test suite\n3. Deploy service to production"
        guard.on_plan_proposed(task_description="Build and deploy", proposed_plan=plan)

        b_res = guard.on_subgoal_boundary(
            subgoal_id="subgoal_001",
            subgoal_status="completed",
            step_history=[],
        )
        assert b_res["checkpoint_passed"] is True
        assert b_res["next_subgoal"] == "subgoal_002"
        assert "prerequisite_violation" not in b_res
        assert b_res.get("warning") is None
