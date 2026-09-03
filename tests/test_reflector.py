"""
Comprehensive test suite for longhorizon_guard.reflector (Phase 5 PlanReflector).

Test Cases:
  (a) Plan still valid — verify plan_still_valid=True, revision_suggested=False
  (b) Plan invalidated by failures/drift — verify plan_still_valid=False & clear reasoning
  (c) Fires correctly on subgoal boundary
  (d) Fires correctly on step-interval even without a boundary
  (e) Coincidence guard — does NOT double-fire when triggers coincide on same step
  (f) Missing drift_monitor data — verify graceful degradation
  (g) Internal exception — verify fail-open
  (h) Timeout — verify fails open cleanly
  (i) Full Integration Check — all 4 hooks aggregate 4 modules without key collisions
"""

import logging
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.reflector.schema import ReflectionResult
from longhorizon_guard.reflector.reflector import (
    PlanReflector,
    DEFAULT_STEP_INTERVAL,
)

PATTERN_LIB_PATH = str(REPO_ROOT / "findings" / "pattern_library.json")


# =========================================================================
# Test (a): Plan still clearly valid
# =========================================================================

class TestPlanStillValid:
    """Test that execution without drift/failures produces valid plan assessment."""

    def test_clean_execution_plan_valid(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Find apple", "1. Go to table\n2. Pick apple")

        subgoal_state = {
            "current_subgoal_id": "subgoal_001",
            "failed_subgoals_count": 0,
            "stalled_advanced_subgoals_count": 0,
            "subgoal_progress_ratio": 0.50,
        }
        drift_assessment = {"drift_detected": False, "severity_score": 0.10}
        step_record = {"step_index": 5, "reasoning": "Moving forward."}

        res = reflector.evaluate(
            step_record=step_record,
            history=[{"step_index": i} for i in range(5)],
            subgoal_state=subgoal_state,
            drift_assessment=drift_assessment,
            trigger_type="step_interval",
        )

        assert res.plan_still_valid is True
        assert res.revision_suggested is False
        assert "valid and achievable" in res.revision_reasoning
        assert res.trigger_type == "step_interval"

    def test_single_failed_subgoal_plan_remains_valid(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Open box", "1. Open drawer 1\n2. Open box")

        subgoal_state = {
            "current_subgoal_id": "subgoal_002",
            "failed_subgoals_count": 1,  # Only 1 failed subgoal
            "stalled_advanced_subgoals_count": 0,
            "subgoal_progress_ratio": 0.25,
        }
        step_record = {"step_index": 5, "reasoning": "Retrying drawer 2."}

        res = reflector.evaluate(
            step_record=step_record,
            history=[{"step_index": i} for i in range(5)],
            subgoal_state=subgoal_state,
            drift_assessment=None,
            trigger_type="step_interval",
        )

        # Single failed subgoal is recoverable — plan remains valid
        assert res.plan_still_valid is True
        assert res.revision_suggested is False


# =========================================================================
# Test (b): Plan invalidated by accumulated failures / drift
# =========================================================================

class TestPlanInvalidated:
    """Test plan invalidation when subgoals fail or drift severity is high."""

    def test_two_failed_subgoals_invalidates_plan(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Open box", "1. Open drawer 1\n2. Open drawer 2\n3. Open box")

        subgoal_state = {
            "current_subgoal_id": "subgoal_003",
            "failed_subgoals_count": 2,  # 2 failed subgoals
            "stalled_advanced_subgoals_count": 0,
            "subgoal_progress_ratio": 0.0,
        }
        step_record = {"step_index": 5, "reasoning": "Drawer 1 & 2 both failed."}

        res = reflector.evaluate(
            step_record=step_record,
            history=[{"step_index": i} for i in range(5)],
            subgoal_state=subgoal_state,
            drift_assessment=None,
            trigger_type="step_interval",
        )

        assert res.plan_still_valid is False
        assert res.revision_suggested is True
        assert "subgoals failed" in res.revision_reasoning
        assert res.confidence >= 0.70

    def test_high_drift_severity_invalidates_plan(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Search items", "1. Search A\n2. Search B")

        subgoal_state = {
            "current_subgoal_id": "subgoal_001",
            "stalled_advanced_subgoals_count": 2,
            "failed_subgoals_count": 0,
        }
        drift_assessment = {
            "drift_detected": True,
            "severity_score": 0.75,
            "triggered_signals": ["REPEATED_STALLED_SUBGOALS", "SLOW_PROGRESS_RATIO"],
        }
        step_record = {"step_index": 5, "reasoning": "Still stuck."}

        res = reflector.evaluate(
            step_record=step_record,
            history=[{"step_index": i} for i in range(5)],
            subgoal_state=subgoal_state,
            drift_assessment=drift_assessment,
            trigger_type="step_interval",
        )

        assert res.plan_still_valid is False
        assert res.revision_suggested is True
        assert "stalled-advanced" in res.revision_reasoning or "Drift monitor" in res.revision_reasoning


# =========================================================================
# Test (c) & (d): Subgoal boundary & Step-interval triggers
# =========================================================================

class TestTriggers:
    """Test boundary and step-interval trigger logic."""

    def test_fires_on_step_interval(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Task", "Plan")

        # Step 4: not interval step -> no eval
        res_4 = reflector.evaluate({"step_index": 4}, history=[{"step_index": i} for i in range(4)])
        assert res_4.plan_still_valid is True
        assert res_4.revision_reasoning == ""  # skipped

        # Step 5: step-interval hit -> evaluates
        res_5 = reflector.evaluate({"step_index": 5}, history=[{"step_index": i} for i in range(5)])
        assert "step_interval" in res_5.trigger_type
        assert res_5.step_index == 5

    def test_fires_on_subgoal_boundary(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Task", "Plan")

        res_boundary = reflector.evaluate(
            step_record={"step_index": 3},
            history=[{"step_index": i} for i in range(3)],
            trigger_type="subgoal_boundary",
            force=True,
        )
        assert res_boundary.trigger_type == "subgoal_boundary"
        assert res_boundary.step_index == 3


# =========================================================================
# Test (e): Coincidence Guard
# =========================================================================

class TestCoincidenceGuard:
    """Test that boundary & step-interval on same step do not double-fire."""

    def test_coincidence_guard_prevents_double_eval(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Task", "Plan")

        # First eval on step 5 (step-interval)
        res1 = reflector.evaluate({"step_index": 5}, history=[{"step_index": i} for i in range(5)])
        assert res1.step_index == 5

        # Second eval on SAME step 5 (subgoal_boundary without force)
        res2 = reflector.evaluate(
            {"step_index": 5},
            history=[{"step_index": i} for i in range(5)],
            trigger_type="subgoal_boundary",
            force=False,
        )

        # Must skip second evaluation to prevent double-firing
        assert res2.revision_reasoning == ""


# =========================================================================
# Test (f): Missing drift_monitor data — Graceful degradation
# =========================================================================

class TestGracefulDegradation:
    """Test reflector works when drift_assessment is None."""

    def test_missing_drift_monitor_data_works(self):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Task", "Plan")

        res = reflector.evaluate(
            step_record={"step_index": 5},
            history=[{"step_index": i} for i in range(5)],
            subgoal_state={"current_subgoal_id": "subgoal_001"},
            drift_assessment=None,  # Missing
            match_details=None,
        )

        assert res.plan_still_valid is True
        assert "subgoals" in res.evidence_sources
        assert "drift_monitor" not in res.evidence_sources


# =========================================================================
# Test (g) & (h): Internal Exception & Timeout Fail-Open
# =========================================================================

class TestFailOpenAndTimeout:
    """Test fail-open exception and timeout safety."""

    def test_internal_exception_fails_open(self, caplog):
        reflector = PlanReflector(step_interval=5)
        reflector.init_plan("Task", "Plan")

        with patch.object(reflector, "_evaluate_impl", side_effect=RuntimeError("reflector boom")):
            with caplog.at_level(logging.ERROR, logger="longhorizon_guard.reflector"):
                res = reflector.evaluate({"step_index": 5}, history=[{"step_index": i} for i in range(5)])

        assert res.plan_still_valid is True
        assert res.revision_suggested is False

    def test_timeout_fails_open(self):
        import time as _time

        reflector = PlanReflector(step_interval=5, timeout_seconds=0.1)
        reflector.init_plan("Task", "Plan")

        def slow_impl(*args, **kwargs):
            _time.sleep(2.0)
            return ReflectionResult()

        reflector._evaluate_impl = slow_impl

        res = reflector.evaluate({"step_index": 5}, history=[{"step_index": i} for i in range(5)], force=True)
        assert res.plan_still_valid is True
        assert res.revision_suggested is False
        assert "timed out" in res.revision_reasoning


# =========================================================================
# Test (i): Full Integration Check across all 4 components & 4 hooks
# =========================================================================

class TestFullIntegrationAllFourModules:
    """Verify that GuardInterface, subgoals, drift_monitor, and reflector
    aggregate outputs cleanly across all 4 hooks without key collision."""

    def test_all_four_hooks_integrate_all_four_modules(self):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)

        # Hook 1: on_plan_proposed
        plan_res = guard.on_plan_proposed(
            task_description="GAIA multi-step web research task",
            proposed_plan=(
                "1. Search web for dress shirts\n"
                "2. Extract shirt sizes\n"
                "3. Select men's dress shirt under $50\n"
                "4. Complete checkout"
            ),
            metadata={"run_id": "test_integration_all"},
        )
        assert plan_res["approved"] is True
        assert "subgoals" in plan_res
        assert plan_res["subgoals"]["total_subgoals_count"] == 4

        # Hook 2: on_step
        history = []
        step0 = {
            "step_index": 0,
            "reasoning": "Searching web for dress shirts.",
            "action_name": "search",
            "action_args": {"query": "dress shirts"},
            "tool_response": "Found dress shirt results.",
        }
        step_res = guard.on_step(step0, history=history, metadata={"run_id": "test_integration_all"})

        # Check all module keys in on_step
        assert "continue_execution" in step_res
        assert "drift_detected" in step_res
        assert "subgoal_state" in step_res
        assert "drift_assessment" in step_res
        assert "reflection_result" in step_res

        history.append(step0)

        # Hook 3: on_subgoal_boundary
        b_res = guard.on_subgoal_boundary(
            subgoal_id="subgoal_001",
            subgoal_status="completed",
            step_history=history,
        )
        assert "checkpoint_passed" in b_res
        assert "subgoal_state" in b_res
        assert "drift_assessment" in b_res
        assert "reflection_result" in b_res
        assert b_res["reflection_result"]["trigger_type"] == "subgoal_boundary"

        # Hook 4: on_run_end
        run_metadata = {"run_id": "test_integration_all"}
        trajectory = {"steps": history}
        run_res = guard.on_run_end(run_metadata, trajectory)

        assert "processed" in run_res
        assert "flags_summary" in run_res
        assert "subgoals_summary" in run_res
        assert "drift_summary" in run_res
        assert "reflection_summary" in run_res

        summary = run_res["subgoals_summary"]
        assert summary["total_subgoals"] == 4
