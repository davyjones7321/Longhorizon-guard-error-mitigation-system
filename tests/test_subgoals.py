"""
Comprehensive test suite for longhorizon_guard.subgoals (SubgoalTracker & schema).

Test Cases:
  (a) Plan with clear subgoal structure — verify parsing & tracking
  (b) Plan with no subgoal structure — verify single-subgoal fallback
  (c) Step completing a subgoal — verify transition event & on_subgoal_boundary
  (d) Step not mapping to any subgoal — verify no crash
  (e) Malformed/corrupted plan input — verify fail-open
  (f) Full trajectory run with multiple subgoals — verify on_run_end summary
  (g) State payload schema verification for drift_monitor consumption
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
from longhorizon_guard.subgoals.schema import (
    SubgoalRecord,
    SubgoalStatus,
    SubgoalStatePayload,
)
from longhorizon_guard.subgoals.tracker import (
    SubgoalTracker,
    parse_plan_subgoals,
)

PATTERN_LIB_PATH = str(REPO_ROOT / "findings" / "pattern_library.json")


# =========================================================================
# Test (a): Plan with clear subgoal structure
# =========================================================================

class TestSubgoalParsingAndTracking:
    """Test parsing and tracking on structured plans."""

    def test_parse_numbered_plan(self):
        plan = (
            "1. Search for red shirts on webshop\n"
            "2. Select size 3X-large tall\n"
            "3. Add item to cart and proceed to checkout"
        )
        records, is_fallback = parse_plan_subgoals(plan)

        assert is_fallback is False
        assert len(records) == 3
        assert records[0].subgoal_id == "subgoal_001"
        assert "Search for red shirts" in records[0].description
        assert records[1].subgoal_id == "subgoal_002"
        assert "Select size" in records[1].description
        assert records[2].subgoal_id == "subgoal_003"

    def test_parse_bullet_plan(self):
        plan = (
            "- Go to cabinet 1 and open it\n"
            "- Take the clean ladle\n"
            "- Go to diningtable 1 and place the ladle"
        )
        records, is_fallback = parse_plan_subgoals(plan)

        assert is_fallback is False
        assert len(records) == 3

    def test_parse_transition_words_plan(self):
        plan = (
            "First, search for YouTube video. "
            "Next, inspect video duration and release year. "
            "Finally, extract the channel name."
        )
        records, is_fallback = parse_plan_subgoals(plan)

        assert is_fallback is False
        assert len(records) == 3


# =========================================================================
# Test (b): Single-subgoal fallback for unformatted plans
# =========================================================================

class TestSingleSubgoalFallback:
    """Test fallback mechanism when plan text has no list structure."""

    def test_unformatted_plan_fallback(self):
        plan = "I will solve this task by trying different things until it works."
        records, is_fallback = parse_plan_subgoals(plan)

        assert is_fallback is True
        assert len(records) == 1
        assert records[0].subgoal_id == "subgoal_001"
        assert "Execute task:" in records[0].description

    def test_empty_plan_fallback(self):
        records, is_fallback = parse_plan_subgoals("")

        assert is_fallback is True
        assert len(records) == 1
        assert records[0].subgoal_id == "subgoal_001"


# =========================================================================
# Test (c): Step completion & on_subgoal_boundary transitions
# =========================================================================

class TestSubgoalTransitions:
    """Test step execution transitions and boundary event generation."""

    def test_step_completes_subgoal_and_fires_boundary(self):
        tracker = SubgoalTracker()
        tracker.init_plan(
            task_description="ALFWorld clean ladle task",
            proposed_plan="1. Open cabinet 1\n2. Take clean ladle\n3. Place on table",
            metadata={"run_id": "test_run_sub_001"},
        )

        step1 = {
            "step_index": 0,
            "reasoning": "Go to cabinet 1 and open it.",
            "action_name": "open",
            "action_args": {"target": "cabinet 1"},
            "tool_response": "You open cabinet 1. You see a clean ladle.",
        }

        res = tracker.process_step(step1, history=[], metadata={"run_id": "test_run_sub_001"})

        assert res["checkpoint_passed"] is True
        assert res["transition_event"] is not None
        trans = res["transition_event"]
        assert trans["completed_subgoal_id"] == "subgoal_001"
        assert trans["completed_status"] == SubgoalStatus.COMPLETED.value
        assert trans["next_subgoal_id"] == "subgoal_002"

        # Verify state payload
        state = res["state_payload"]
        assert state["completed_subgoals_count"] == 1
        assert state["current_subgoal_id"] == "subgoal_002"
        assert state["status"] == SubgoalStatus.IN_PROGRESS.value

    def test_guard_interface_on_subgoal_boundary_populated(self):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)
        guard.on_plan_proposed(
            task_description="Find red dress shirt",
            proposed_plan="1. Search webshop\n2. Select size 3X\n3. Checkout",
            metadata={"run_id": "test_boundary_001"},
        )

        # Step 0 triggers transition
        step0 = {
            "step_index": 0,
            "reasoning": "Search for red dress shirt.",
            "action_name": "search",
            "action_args": {"query": "red dress shirt"},
            "tool_response": "You see red dress shirt in results. Found items.",
        }
        step_res = guard.on_step(step0, history=[], metadata={"run_id": "test_boundary_001"})

        assert "subgoal_state" in step_res
        assert "subgoal_transition" in step_res
        assert step_res["subgoal_transition"]["completed_subgoal_id"] == "subgoal_001"

        # Call on_subgoal_boundary
        b_res = guard.on_subgoal_boundary(
            subgoal_id="subgoal_001",
            subgoal_status="completed",
            step_history=[step0],
        )
        assert b_res["checkpoint_passed"] is True
        assert b_res["next_subgoal"] == "subgoal_002"
        assert b_res["subgoal_state"] is not None


# =========================================================================
# Test Part 1: Stalled-Advanced Status Fidelity (Rule S3)
# =========================================================================

class TestStalledAdvancedStatus:
    """Verify that Rule S3 forced-advance produces stalled_advanced status."""

    def test_s3_forced_advance_produces_stalled_advanced(self):
        tracker = SubgoalTracker(max_subgoal_steps=6)
        tracker.init_plan(
            task_description="Long search task",
            proposed_plan="1. Search drawers for key\n2. Open locked box",
            metadata={"run_id": "test_s3_001"},
        )

        # Process 6 steps without success keyword or failure keyword (Rule S3 trigger)
        for i in range(6):
            step = {
                "step_index": i,
                "reasoning": "Searching...",
                "action_name": "open",
                "action_args": {"target": f"drawer {i+1}"},
                "tool_response": "The drawer is empty.",
            }
            res = tracker.process_step(step, history=[])

        # Step 6 (the 6th step) must force-advance with stalled_advanced status
        trans = res["transition_event"]
        assert trans is not None
        assert trans["completed_subgoal_id"] == "subgoal_001"
        assert trans["completed_status"] == SubgoalStatus.STALLED_ADVANCED.value
        assert "force advanced" in trans["trigger_reason"]

        # Check state payload
        state = res["state_payload"]
        assert state["stalled_advanced_subgoals_count"] == 1
        assert state["completed_subgoals_count"] == 0
        assert state["current_subgoal_id"] == "subgoal_002"

    def test_s2_genuine_completion_produces_completed(self):
        tracker = SubgoalTracker()
        tracker.init_plan(
            task_description="Search task",
            proposed_plan="1. Search for key\n2. Use key",
            metadata={"run_id": "test_s2_001"},
        )

        step = {
            "step_index": 0,
            "reasoning": "Search drawer.",
            "action_name": "open",
            "action_args": {"target": "drawer 1"},
            "tool_response": "You open drawer 1. Found key.",
        }
        res = tracker.process_step(step, history=[])

        trans = res["transition_event"]
        assert trans is not None
        assert trans["completed_status"] == SubgoalStatus.COMPLETED.value
        state = res["state_payload"]
        assert state["completed_subgoals_count"] == 1
        assert state["stalled_advanced_subgoals_count"] == 0

    def test_on_run_end_reports_stalled_advanced_bucket(self):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH, max_subgoal_steps=6)
        guard.on_plan_proposed(
            task_description="Task",
            proposed_plan="1. Subgoal A\n2. Subgoal B",
            metadata={"run_id": "test_s3_run_end"},
        )

        # Force advance subgoal A via 6 steps
        for i in range(6):
            guard.on_step(
                {
                    "step_index": i,
                    "reasoning": "Searching...",
                    "action_name": "look",
                    "tool_response": "The room is quiet.",
                },
                history=[],
                metadata={"run_id": "test_s3_run_end"},
            )

        trajectory = {"steps": [{"step_index": i} for i in range(6)]}
        run_res = guard.on_run_end(
            run_metadata={"run_id": "test_s3_run_end"},
            trajectory=trajectory,
        )

        summary = run_res["subgoals_summary"]
        assert summary is not None
        assert summary["stalled_advanced_count"] == 1
        assert summary["completed_count"] == 0



# =========================================================================
# Test (d): Unmapped step handling
# =========================================================================

class TestUnmappedSteps:
    """Test steps that do not map to any active subgoal."""

    def test_step_processed_without_crash_when_subgoals_depleted(self):
        tracker = SubgoalTracker()
        tracker.init_plan("task", "1. Single short action", metadata={"run_id": "test_unmapped"})

        # Finish subgoal 1
        step0 = {
            "step_index": 0,
            "reasoning": "Action",
            "action_name": "click",
            "tool_response": "You arrived at destination. Success.",
        }
        tracker.process_step(step0, history=[])

        # Extra step after all subgoals are completed
        extra_step = {
            "step_index": 1,
            "reasoning": "Extra idle step",
            "action_name": "look",
            "tool_response": "Nothing new.",
        }
        res = tracker.process_step(extra_step, history=[step0])

        assert res["checkpoint_passed"] is True
        assert res["transition_event"] is None


# =========================================================================
# Test (e): Malformed / Corrupted plan input
# =========================================================================

class TestMalformedPlanFailOpen:
    """Test fail-open behavior on corrupted plan inputs."""

    def test_corrupted_plan_fails_open(self):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)

        # None / invalid plan inputs
        res = guard.on_plan_proposed(
            task_description="Task",
            proposed_plan=None,  # type: ignore[arg-type]
            metadata={"run_id": "test_corrupt_plan"},
        )

        assert res["approved"] is True
        assert "subgoals" in res
        assert res["subgoals"]["is_fallback"] is True

    def test_tracker_exception_in_process_step_fails_open(self, caplog):
        tracker = SubgoalTracker()
        tracker.init_plan("task", "1. Do X")

        with patch.object(tracker, "get_state_payload", side_effect=RuntimeError("tracker error")):
            with caplog.at_level(logging.ERROR, logger="longhorizon_guard.guard"):
                res = tracker.process_step({"step_index": 0}, history=[])

        assert res["checkpoint_passed"] is True


# =========================================================================
# Test (f): Full trajectory run with multiple subgoals
# =========================================================================

class TestFullRunSubgoalSummary:
    """Test on_run_end summary across a full trajectory."""

    def test_full_run_subgoals_summary(self):
        guard = GuardInterface(pattern_library_path=PATTERN_LIB_PATH)

        guard.on_plan_proposed(
            task_description="GAIA research task",
            proposed_plan="1. Search web for video\n2. Extract channel name\n3. Format final answer",
            metadata={"run_id": "test_full_run_001"},
        )

        step0 = {
            "step_index": 0,
            "reasoning": "Search web for video.",
            "action_name": "search",
            "tool_response": "Found video URL.",
        }
        guard.on_step(step0, history=[])

        step1 = {
            "step_index": 1,
            "reasoning": "Extract channel name from page.",
            "action_name": "extract",
            "tool_response": "Channel name: National Geographic.",
        }
        guard.on_step(step1, history=[step0])

        trajectory = {"steps": [step0, step1]}
        run_res = guard.on_run_end(
            run_metadata={"run_id": "test_full_run_001"},
            trajectory=trajectory,
        )

        assert run_res["processed"] is True
        assert "subgoals_summary" in run_res
        summary = run_res["subgoals_summary"]
        assert summary is not None
        assert summary["total_subgoals"] == 3
        assert summary["completed_count"] >= 1
        assert len(summary["subgoals"]) == 3


# =========================================================================
# Test (g): State payload schema for drift_monitor consumption
# =========================================================================

class TestDriftMonitorPayloadSchema:
    """Verify that get_state_payload() matches the required contract."""

    def test_state_payload_fields(self):
        tracker = SubgoalTracker()
        tracker.init_plan("task", "1. Step A\n2. Step B")
        payload = tracker.get_state_payload()

        assert isinstance(payload, SubgoalStatePayload)
        d = payload.to_dict()

        required_keys = {
            "current_subgoal_id",
            "current_subgoal_description",
            "status",
            "steps_in_current_subgoal",
            "time_in_current_subgoal_seconds",
            "completed_subgoals_count",
            "failed_subgoals_count",
            "abandoned_subgoals_count",
            "total_subgoals_count",
            "subgoal_progress_ratio",
            "active_subgoal_index",
            "is_fallback",
        }
        assert required_keys.issubset(set(d.keys()))
        assert d["total_subgoals_count"] == 2
        assert d["active_subgoal_index"] == 0
        assert d["status"] == SubgoalStatus.IN_PROGRESS.value


class TestRuleF1WordBoundariesAndBenignPhrasing:
    """Verify Rule F1 word-boundary matching and benign negative-pattern handling (FIX F-07)."""

    def test_benign_error_mentions_do_not_trigger_f1(self):
        """Benign phrases mentioning error ('0 errors', 'no error', etc.) must NOT mark subgoal as failed."""
        benign_responses = [
            "Test suite passed with 0 errors and 0 warnings.",
            "Execution completed: no error detected during run.",
            "Diagnostic status: error: none.",
            "Completed linting: errors: 0.",
            "File transfer finished without error.",
        ]

        for idx, resp in enumerate(benign_responses):
            tracker = SubgoalTracker()
            tracker.init_plan("Task", "1. Execute benign task")
            res = tracker.process_step(
                {
                    "step_index": 0,
                    "reasoning": "Checking status",
                    "action_name": "check",
                    "tool_response": resp,
                },
                history=[],
            )
            state = res["state_payload"]
            assert state["status"] == SubgoalStatus.IN_PROGRESS.value, (
                f"Benign tool response incorrectly triggered failure: '{resp}'"
            )
            assert state["failed_subgoals_count"] == 0

    def test_real_failure_messages_trigger_f1(self):
        """Genuine failure messages must immediately mark subgoal as failed."""
        failure_responses = [
            "Execution error: connection timed out after 30s",
            "Error: unable to locate specified target file",
            "Process failed with exit status 1",
            "Command rejected: syntax error near unexpected token",
            "Action failed: invalid action parameters",
            "Agent could not proceed: cannot find requested item",
        ]

        for idx, resp in enumerate(failure_responses):
            tracker = SubgoalTracker()
            tracker.init_plan("Task", "1. Execute critical task")
            res = tracker.process_step(
                {
                    "step_index": 0,
                    "reasoning": "Attempting action",
                    "action_name": "run",
                    "tool_response": resp,
                },
                history=[],
            )
            trans = res["transition_event"]
            assert trans is not None, f"Expected failure transition for: '{resp}'"
            assert trans["completed_status"] == SubgoalStatus.FAILED.value
            assert "failure keyword" in trans["trigger_reason"]


class TestConfigurableMaxSubgoalSteps:
    """Verify configurable max_subgoal_steps threshold (FIX F-08)."""

    def test_custom_step_threshold_advances_at_configured_count(self):
        """Subgoal auto-advances at custom max_subgoal_steps limit."""
        tracker = SubgoalTracker(max_subgoal_steps=3)
        tracker.init_plan("Task", "1. Step A\n2. Step B")

        for i in range(2):
            res = tracker.process_step(
                {"step_index": i, "reasoning": "Work", "action_name": "step", "tool_response": "neutral"},
                history=[],
            )
            assert res["state_payload"]["status"] == SubgoalStatus.IN_PROGRESS.value

        # 3rd step reaches max_subgoal_steps=3 threshold
        res = tracker.process_step(
            {"step_index": 2, "reasoning": "Work", "action_name": "step", "tool_response": "neutral"},
            history=[],
        )
        assert res["transition_event"]["completed_status"] == SubgoalStatus.STALLED_ADVANCED.value
        assert res["state_payload"]["current_subgoal_id"] == "subgoal_002"

    def test_default_threshold_is_10(self):
        """Default max_subgoal_steps is 10 (uncalibrated heuristic baseline)."""
        tracker = SubgoalTracker()
        assert tracker.max_subgoal_steps == 10

        tracker.init_plan("Task", "1. Step A\n2. Step B")
        # Steps 0..8 (9 steps) must remain IN_PROGRESS
        for i in range(9):
            res = tracker.process_step(
                {"step_index": i, "reasoning": "Work", "action_name": "step", "tool_response": "neutral"},
                history=[],
            )
            assert res["state_payload"]["status"] == SubgoalStatus.IN_PROGRESS.value

        # Step 9 (the 10th step) must advance
        res = tracker.process_step(
            {"step_index": 9, "reasoning": "Work", "action_name": "step", "tool_response": "neutral"},
            history=[],
        )
        assert res["transition_event"]["completed_status"] == SubgoalStatus.STALLED_ADVANCED.value

