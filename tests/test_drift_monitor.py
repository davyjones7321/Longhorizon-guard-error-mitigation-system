"""
Comprehensive test suite for longhorizon_guard.drift_monitor.

Test Cases:
  (a) Clean run, no drift signals — verify drift_detected=False
  (b) Run with 2+ stalled_advanced subgoals — verify drift_detected=True & REPEATED_STALLED_SUBGOALS
  (c) Run with slow progress_ratio relative to step count — verify SLOW_PROGRESS_RATIO
  (d) Exception/malformed input — verify fail-open
  (e) Single stalled_advanced (1 alone) — verify NO over-triggering of drift on its own
  (f) GuardInterface integration — verify drift assessment in on_step, on_subgoal_boundary, on_run_end
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.drift_monitor.schema import DriftAssessment
from longhorizon_guard.drift_monitor.monitor import (
    DriftMonitor,
    SIGNAL_REPEATED_STALLED,
    SIGNAL_SLOW_PROGRESS,
    SIGNAL_ACCUMULATED_FAILURES,
    SIGNAL_PATTERN_REPETITION,
)
from longhorizon_guard.subgoals.schema import SubgoalStatus

PATTERN_LIB_PATH = str(REPO_ROOT / "findings" / "pattern_library.json") if (REPO_ROOT / "findings" / "pattern_library.json").exists() else str(REPO_ROOT.parent / "findings" / "pattern_library.json")


# =========================================================================
# Test (a): Clean run, no drift signals
# =========================================================================

class TestCleanRunNoDrift:
    """Test that normal, progress-making steps produce no drift flags."""

    def test_clean_step_no_drift(self):
        monitor = DriftMonitor()
        subgoal_state = {
            "current_subgoal_id": "subgoal_001",
            "current_subgoal_description": "Search web",
            "status": "in_progress",
            "steps_in_current_subgoal": 2,
            "time_in_current_subgoal_seconds": 1.5,
            "completed_subgoals_count": 1,
            "stalled_advanced_subgoals_count": 0,
            "failed_subgoals_count": 0,
            "abandoned_subgoals_count": 0,
            "total_subgoals_count": 3,
            "subgoal_progress_ratio": 0.333,
            "active_subgoal_index": 1,
            "is_fallback": False,
        }
        step_record = {"step_index": 2, "reasoning": "Looking for items."}
        history = [{"step_index": 0}, {"step_index": 1}]

        assessment = monitor.evaluate_step(subgoal_state, step_record, history)

        assert assessment.drift_detected is False
        assert assessment.severity_level in ("none", "low")
        assert len(assessment.triggered_signals) == 0


# =========================================================================
# Test (b): 2+ stalled_advanced subgoals trigger REPEATED_STALLED_SUBGOALS
# =========================================================================

class TestRepeatedStalledSubgoals:
    """Test that >=2 stalled_advanced subgoals trigger drift assessment."""

    def test_two_stalled_subgoals_triggers_drift(self):
        monitor = DriftMonitor()
        subgoal_state = {
            "current_subgoal_id": "subgoal_003",
            "current_subgoal_description": "Open drawer",
            "status": "in_progress",
            "steps_in_current_subgoal": 1,
            "stalled_advanced_subgoals_count": 2,  # 2 stalled subgoals
            "completed_subgoals_count": 0,
            "failed_subgoals_count": 0,
            "total_subgoals_count": 4,
            "subgoal_progress_ratio": 0.50,
        }
        step_record = {"step_index": 12, "reasoning": "Still trying..."}
        history = [{"step_index": i} for i in range(12)]

        assessment = monitor.evaluate_step(subgoal_state, step_record, history)

        assert assessment.drift_detected is True
        assert SIGNAL_REPEATED_STALLED in assessment.triggered_signals
        assert assessment.severity_score >= 0.45
        assert assessment.severity_level in ("medium", "high", "critical")


# =========================================================================
# Test (e): Single stalled_advanced (1 alone) does NOT trigger drift on its own
# =========================================================================

class TestSingleStalledNoOverTrigger:
    """Verify that a single stalled subgoal alone does not trigger drift."""

    def test_single_stalled_subgoal_does_not_trigger_drift(self):
        monitor = DriftMonitor()
        subgoal_state = {
            "current_subgoal_id": "subgoal_002",
            "current_subgoal_description": "Find item",
            "status": "in_progress",
            "steps_in_current_subgoal": 2,
            "stalled_advanced_subgoals_count": 1,  # Only 1 stalled subgoal
            "completed_subgoals_count": 0,
            "failed_subgoals_count": 0,
            "total_subgoals_count": 4,
            "subgoal_progress_ratio": 0.25,
        }
        step_record = {"step_index": 6, "reasoning": "Searching..."}
        history = [{"step_index": i} for i in range(6)]

        assessment = monitor.evaluate_step(subgoal_state, step_record, history)

        # 1 stalled alone is NOT enough for REPEATED_STALLED_SUBGOALS
        assert SIGNAL_REPEATED_STALLED not in assessment.triggered_signals
        assert assessment.drift_detected is False


class TestSingleFailedNoOverTrigger:
    """Verify that a single failed subgoal alone does not trigger drift."""

    def test_single_failed_subgoal_does_not_trigger_drift(self):
        monitor = DriftMonitor()
        subgoal_state = {
            "current_subgoal_id": "subgoal_002",
            "current_subgoal_description": "Open drawer",
            "status": "in_progress",
            "steps_in_current_subgoal": 2,
            "stalled_advanced_subgoals_count": 0,
            "completed_subgoals_count": 1,
            "failed_subgoals_count": 1,  # Only 1 failed subgoal
            "total_subgoals_count": 4,
            "subgoal_progress_ratio": 0.25,
        }
        step_record = {"step_index": 5, "reasoning": "Retrying next drawer..."}
        history = [{"step_index": i} for i in range(5)]

        assessment = monitor.evaluate_step(subgoal_state, step_record, history)

        # 1 failed subgoal alone is recoverable agent behavior — must NOT trigger drift
        assert SIGNAL_ACCUMULATED_FAILURES not in assessment.triggered_signals
        assert assessment.drift_detected is False

    def test_two_failed_subgoals_triggers_drift(self):
        monitor = DriftMonitor()
        subgoal_state = {
            "current_subgoal_id": "subgoal_003",
            "current_subgoal_description": "Open cabinet",
            "status": "in_progress",
            "steps_in_current_subgoal": 2,
            "stalled_advanced_subgoals_count": 0,
            "completed_subgoals_count": 0,
            "failed_subgoals_count": 2,  # 2 failed subgoals
            "total_subgoals_count": 4,
            "subgoal_progress_ratio": 0.0,
        }
        step_record = {"step_index": 6, "reasoning": "Still failing..."}
        history = [{"step_index": i} for i in range(6)]

        assessment = monitor.evaluate_step(subgoal_state, step_record, history)

        assert assessment.drift_detected is True
        assert SIGNAL_ACCUMULATED_FAILURES in assessment.triggered_signals


class TestMissingPatternDataGracefulDegradation:
    """Verify graceful degradation when GuardInterface pattern flag data is absent."""

    def test_standalone_drift_monitor_without_pattern_data(self):
        monitor = DriftMonitor()
        subgoal_state = {
            "current_subgoal_id": "subgoal_001",
            "current_subgoal_description": "Search item",
            "status": "in_progress",
            "steps_in_current_subgoal": 2,
            "completed_subgoals_count": 0,
            "stalled_advanced_subgoals_count": 0,
            "failed_subgoals_count": 0,
            "total_subgoals_count": 3,
            "subgoal_progress_ratio": 0.0,
        }
        step_record = {"step_index": 1, "reasoning": "Normal search"}

        # Explicitly pass match_details=None (pattern flag data absent)
        assessment = monitor.evaluate_step(
            subgoal_state=subgoal_state,
            step_record=step_record,
            history=[{"step_index": 0}],
            match_details=None,
        )

        assert assessment.drift_detected is False
        assert SIGNAL_PATTERN_REPETITION not in assessment.triggered_signals
        assert assessment.severity_level in ("none", "low")


# =========================================================================
# Test (c): Slow progress_ratio relative to step count
# =========================================================================

class TestSlowProgressRatio:
    """Test drift detection when steps accumulate without plan progress."""

    def test_slow_progress_ratio_flagged(self):
        monitor = DriftMonitor()
        subgoal_state = {
            "current_subgoal_id": "subgoal_001",
            "current_subgoal_description": "Initial search",
            "status": "in_progress",
            "steps_in_current_subgoal": 9,  # 9 steps in single subgoal
            "stalled_advanced_subgoals_count": 0,
            "completed_subgoals_count": 0,
            "failed_subgoals_count": 0,
            "total_subgoals_count": 5,
            "subgoal_progress_ratio": 0.0,
        }
        step_record = {"step_index": 8, "reasoning": "Wandering around..."}
        history = [{"step_index": i} for i in range(8)]

        assessment = monitor.evaluate_step(subgoal_state, step_record, history)

        assert assessment.drift_detected is True
        assert SIGNAL_SLOW_PROGRESS in assessment.triggered_signals


# =========================================================================
# Test (d): Exception / Malformed input fail-open
# =========================================================================

class TestDriftMonitorFailOpen:
    """Test fail-open behavior on corrupted or unexpected input."""

    def test_malformed_subgoal_state_fails_open(self):
        monitor = DriftMonitor()
        # Invalid / missing state keys
        assessment = monitor.evaluate_step(
            subgoal_state={"corrupted": True},
            step_record={},
            history=[],
        )
        assert assessment.drift_detected is False
        assert assessment.severity_level in ("none", "low")

    def test_monitor_exception_fails_open(self, caplog):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)
        with patch.object(guard._drift_monitor, "evaluate_step", side_effect=ValueError("drift monitor error")):
            with caplog.at_level(logging.ERROR, logger="longhorizon_guard.guard"):
                step_res = guard.on_step({"step_index": 0}, history=[])

        assert step_res["continue_execution"] is True
        assert step_res["drift_detected"] is False


# =========================================================================
# Test (f): GuardInterface Integration
# =========================================================================

class TestGuardInterfaceDriftIntegration:
    """Test integrated drift monitoring in GuardInterface hooks."""

    def test_on_step_attaches_drift_assessment(self):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)
        guard.on_plan_proposed(
            task_description="Search task",
            proposed_plan="1. Subgoal A\n2. Subgoal B\n3. Subgoal C\n4. Subgoal D",
            metadata={"run_id": "test_gi_drift"},
        )

        step_res = guard.on_step(
            {
                "step_index": 0,
                "reasoning": "Step 0",
                "tool_response": "Result 0",
            },
            history=[],
            metadata={"run_id": "test_gi_drift"},
        )

        assert "drift_assessment" in step_res
        assert step_res["drift_assessment"]["drift_detected"] is False

    def test_on_run_end_attaches_drift_summary(self):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)
        guard.on_plan_proposed(
            task_description="Task",
            proposed_plan="1. A\n2. B",
            metadata={"run_id": "test_gi_run_end"},
        )

        trajectory = {"steps": [{"step_index": 0}, {"step_index": 1}]}
        run_res = guard.on_run_end(
            run_metadata={"run_id": "test_gi_run_end"},
            trajectory=trajectory,
        )

        assert "drift_summary" in run_res
        assert run_res["drift_summary"]["run_id"] == "test_gi_run_end"
        assert "final_drift_assessment" in run_res["drift_summary"]
