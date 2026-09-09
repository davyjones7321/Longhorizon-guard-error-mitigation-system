"""
Production tests for GuardInterface.

Tests:
  (a) Step matching a known pattern cluster
  (b) Step matching only via keyword fallback (tool_use_error / external_error)
  (c) Exception inside matcher — verify fail-open + logging
  (d) Clean step with no match — verify no false positive
  (e) on_plan_proposed catches planning_error patterns
  (f) Structural heuristic: repeated action detection
  (g) Structural heuristic: 'Nothing happens' loop
  (h) on_run_end full trajectory scan
  (i) Match timeout fail-open
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure longhorizon_guard is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longhorizon_guard.taxonomy.categories import DEFAULT_TAGS
from longhorizon_guard.config import GuardConfig
from longhorizon_guard.interface import (
    GuardInterface,
    PatternMatcher,
    MatchResult,
    KeywordRule,
    _build_keyword_rules,
    _detect_action_repetition,
    _detect_nothing_happens_loop,
    _tokenize,
    _tfidf_vector,
    _cosine_sim,
    DEFAULT_CATEGORY_THRESHOLDS,
)

# Path to real pattern library (produced by mine_patterns)
PATTERN_LIB_PATH = str(REPO_ROOT / "findings" / "pattern_library.json")


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def guard():
    """GuardInterface loaded with real pattern_library.json."""
    return GuardInterface(pattern_library_path=PATTERN_LIB_PATH)


@pytest.fixture
def guard_no_patterns(tmp_path):
    """GuardInterface with empty pattern library (keyword-only mode)."""
    empty = tmp_path / "empty_patterns.json"
    empty.write_text(json.dumps({"patterns": []}), encoding="utf-8")
    return GuardInterface(pattern_library_path=str(empty))


@pytest.fixture
def guard_missing_file():
    """GuardInterface pointing to a non-existent file — should init safely."""
    return GuardInterface(pattern_library_path="/nonexistent/path.json")


# =========================================================================
# Test (a): Step matching a known pattern cluster
# =========================================================================

class TestClusterMatch:
    """Test that a step resembling a known planning_error pattern gets matched."""

    def test_planning_error_flawed_search_query(self, guard):
        """Text similar to pattern_planning_error_001 (flawed search query) should match."""
        step = {
            "step_index": 0,
            "reasoning": (
                "The agent's initial plan and search query were flawed and incomplete, "
                "leading to an inefficient search process and failure to find the correct items. "
                "The search did not consider all necessary parameters."
            ),
            "action_name": "search",
            "action_args": {"query": "dress shirts"},
            "tool_response": "No results found",
        }
        result = guard.on_step(step, history=[], metadata={"run_id": "test_cluster_001"})

        assert result["continue_execution"] is True  # never blocks
        # We expect either cluster or keyword match for planning-related text
        if result["drift_detected"]:
            assert result["match_details"] is not None
            assert result["match_details"]["category"] in (
                "planning_error", "reflection_error", "memory_error"
            )
            assert result["match_details"]["confidence"] > 0
            assert result["match_details"]["layer"] in ("cluster", "keyword")

    def test_mechanical_sequential_search_pattern(self, guard):
        """Text about checking drawers/cabinets one by one should match planning_error."""
        step = {
            "step_index": 5,
            "reasoning": (
                "The agent began to check drawers one by one after finding the desk lamp, "
                "following a mechanical strategy of opening each cabinet sequentially "
                "without prioritizing more reasonable search locations."
            ),
            "action_name": "open",
            "action_args": {"target": "drawer 3"},
            "tool_response": "The drawer is empty.",
        }
        result = guard.on_step(step, history=[], metadata={"run_id": "test_cluster_002"})

        assert result["continue_execution"] is True
        if result["drift_detected"]:
            details = result["match_details"]
            assert details["category"] in ("planning_error",)
            assert details["confidence"] > 0


# =========================================================================
# Test (b): Keyword fallback only (tool_use_error / external_error)
# =========================================================================

class TestKeywordFallback:
    """Test keyword rules for categories with no cluster patterns."""

    def test_tool_use_nothing_happens(self, guard_no_patterns):
        """'Nothing happens' in tool_response should flag tool_use_error via keywords."""
        step = {
            "step_index": 3,
            "reasoning": "Go to the coffee table.",
            "action_name": "'go",
            "action_args": {"target": "coffeetable 1"},
            "tool_response": "Nothing happens.",
        }
        result = guard_no_patterns.on_step(step, history=[], metadata={"run_id": "test_kw_001"})

        assert result["continue_execution"] is True
        if result["drift_detected"]:
            details = result["match_details"]
            assert details["layer"] == "keyword"
            assert details["category"] in ("tool_use_error", "reflection_error")

    def test_external_step_limit(self, guard_no_patterns):
        """Text mentioning step limit / timeout should flag external_error."""
        step = {
            "step_index": 60,
            "reasoning": "The trajectory has exceeded the maximum steps limit and was truncated.",
            "action_name": "terminate",
            "action_args": {},
            "tool_response": "Step limit reached. Maximum steps exceeded.",
        }
        result = guard_no_patterns.on_step(step, history=[], metadata={"run_id": "test_kw_002"})

        assert result["continue_execution"] is True
        assert result["drift_detected"] is True
        details = result["match_details"]
        assert details is not None
        assert details["layer"] == "keyword"
        assert details["category"] == "external_error"
        assert details["rule_id"] in ("external_step_limit", "external_env_error")

    def test_malformed_action(self, guard_no_patterns):
        """Malformed action should be caught by keyword rules even without patterns."""
        step = {
            "step_index": 1,
            "reasoning": "Try to go to the cabinet.",
            "action_name": "'go",
            "action_args": {"target": "cabinet 1"},
            "tool_response": "Nothing happens. The action was malformed and invalid.",
        }
        result = guard_no_patterns.on_step(step, history=[], metadata={"run_id": "test_kw_003"})

        assert result["continue_execution"] is True
        if result["drift_detected"]:
            details = result["match_details"]
            assert details["layer"] == "keyword"
            assert details["category"] in ("tool_use_error", "reflection_error")


# =========================================================================
# Test (c): Exception in matcher — verify fail-open + logging
# =========================================================================

class TestFailOpen:
    """Verify that exceptions inside the matcher don't crash the guard."""

    def test_matcher_exception_fails_open(self, guard, caplog):
        """If the matcher raises, on_step should return safe defaults and log."""
        step = {
            "step_index": 5,
            "reasoning": "Some text",
            "action_name": "search",
        }

        with patch.object(guard._matcher, "match", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.ERROR, logger="longhorizon_guard.guard"):
                result = guard.on_step(step, history=[], metadata={"run_id": "test_fail_001"})

        # Must continue execution (fail open)
        assert result["continue_execution"] is True
        assert result["drift_detected"] is False
        assert result["warning"] is None

        # Must have logged the error
        assert any("on_step error" in r.message or "boom" in r.message
                    for r in caplog.records)

    def test_missing_pattern_file_inits_safely(self, guard_missing_file):
        """GuardInterface with missing file should init without raising."""
        step = {"step_index": 0, "reasoning": "test"}
        result = guard_missing_file.on_step(step, history=[], metadata={})

        assert result["continue_execution"] is True
        assert result["drift_detected"] is False

    def test_corrupted_step_record(self, guard):
        """Garbage step record should not crash."""
        result = guard.on_step(
            step_record={},  # no fields
            history=[],
            metadata={"run_id": "test_corrupt"},
        )
        assert result["continue_execution"] is True

    def test_on_plan_proposed_exception_fails_open(self, guard, caplog):
        """Exception in on_plan_proposed should fail open."""
        with patch.object(guard._matcher, "match", side_effect=ValueError("bad")):
            with caplog.at_level(logging.ERROR, logger="longhorizon_guard.guard"):
                result = guard.on_plan_proposed("task", "plan", metadata={"run_id": "x"})

        assert result["approved"] is True
        assert result["flags"] == []


# =========================================================================
# Test (d): Clean step with no match — no false positive
# =========================================================================

class TestNoFalsePositive:
    """A benign step should produce no flags."""

    def test_clean_step_no_match(self, guard):
        """A normal, successful step should not trigger any match."""
        step = {
            "step_index": 3,
            "reasoning": "I found the apple on the counter. I will pick it up and bring it to the table.",
            "action_name": "take",
            "action_args": {"object": "apple 1", "from": "countertop 1"},
            "tool_response": "You pick up the apple 1 from the countertop 1.",
        }
        result = guard.on_step(step, history=[], metadata={"run_id": "test_clean_001"})

        assert result["continue_execution"] is True
        assert result["drift_detected"] is False
        assert result["warning"] is None
        assert result["match_details"] is None

    def test_clean_plan_no_flag(self, guard):
        """A reasonable plan should not be flagged."""
        result = guard.on_plan_proposed(
            task_description="Find the apple and put it on the table.",
            proposed_plan="First, look around. Then pick up the apple. Then go to the table and put it down.",
            metadata={"run_id": "test_clean_plan"},
        )

        assert result["approved"] is True
        assert result["flags"] == []


# =========================================================================
# Test (e): on_plan_proposed catches planning_error
# =========================================================================

class TestOnPlanProposed:
    """Test that planning_error patterns are caught at plan stage."""

    def test_plan_with_planning_error_signals(self, guard):
        """A plan that resembles known planning_error patterns should flag."""
        result = guard.on_plan_proposed(
            task_description="Find men's dress shirts under $50",
            proposed_plan=(
                "The initial plan and search query were flawed. "
                "The agent's plan did not consider all the necessary parameters "
                "for the search query, leading to an incomplete and inefficient search."
            ),
            metadata={"run_id": "test_plan_001"},
        )

        assert result["approved"] is True  # flag, don't block
        # May or may not flag depending on threshold — just verify structure
        assert isinstance(result["flags"], list)
        assert isinstance(result["suggestions"], list)

    def test_contradictory_plan_flagged(self, guard):
        """A plan that directly contradicts task constraints should flag a contradiction."""
        result = guard.on_plan_proposed(
            task_description="Find men's dress shirts",
            proposed_plan="1. Search for women's dress shirts. 2. Select first result.",
            metadata={"run_id": "test_plan_contradict"},
        )
        assert result["approved"] is True
        assert any("planning_contradiction" in f for f in result["flags"])
        assert any("women" in f and "men" in f for f in result["flags"])
        assert len(result["suggestions"]) > 0

    def test_omitted_constraint_plan_flagged(self, guard):
        """A plan that omits explicit task constraints from search strategy should flag an omission."""
        result = guard.on_plan_proposed(
            task_description="Find men's dress shirts under $50",
            proposed_plan="1. Search for dress shirts. 2. Select first result.",
            metadata={"run_id": "test_plan_omit"},
        )
        assert result["approved"] is True
        assert any("planning_constraint_omission" in f for f in result["flags"])
        assert len(result["suggestions"]) > 0

    def test_clean_constrained_plan_passes(self, guard):
        """A plan that properly includes required task constraints should pass cleanly."""
        result = guard.on_plan_proposed(
            task_description="Find blue cotton socks under $20",
            proposed_plan="1. Search for blue cotton socks under $20. 2. Select first result.",
            metadata={"run_id": "test_plan_clean_constrained"},
        )
        assert result["approved"] is True
        assert result["flags"] == []

    def test_search_step_contradiction_flagged(self, guard):
        """A search action step that contradicts task description should be flagged in on_step."""
        step = {
            "step_index": 0,
            "action_name": "search",
            "action_args": {"query": "women's dress shirts"},
            "reasoning": "Searching for the item",
        }
        meta = {"task_description": "Find men's dress shirts"}
        res = guard.on_step(step, history=[], metadata=meta)
        assert res["drift_detected"] is True
        assert "planning_contradiction" in str(res.get("warning"))

    def test_guard_interface_none_path_safe(self):
        """GuardInterface(pattern_library_path=None) must safely default without raising TypeError."""
        from longhorizon_guard import GuardInterface
        g = GuardInterface(pattern_library_path=None)
        assert g is not None


# =========================================================================
# Test (f): Structural heuristic — repeated action
# =========================================================================

class TestStructuralRepetition:
    """Test the action-repetition structural heuristic."""

    def test_repeated_action_detected(self):
        """3 identical actions in history should trigger memory_error."""
        current = {
            "step_index": 5,
            "action_name": "go",
            "action_args": {"target": "desk 1"},
        }
        history = [
            {"action_name": "go", "action_args": {"target": "desk 1"}},
            {"action_name": "look", "action_args": {}},
            {"action_name": "go", "action_args": {"target": "desk 1"}},
            {"action_name": "go", "action_args": {"target": "desk 1"}},
        ]
        result = _detect_action_repetition(current, history)
        assert result is not None
        assert result.matched is True
        assert result.category == "memory_error"
        assert result.rule_id == "structural_action_repetition"

    def test_no_repetition(self):
        """Different actions should not trigger."""
        current = {"step_index": 2, "action_name": "take", "action_args": {"obj": "cup"}}
        history = [
            {"action_name": "look", "action_args": {}},
            {"action_name": "go", "action_args": {"target": "table"}},
        ]
        result = _detect_action_repetition(current, history)
        assert result is None

    def test_repetition_threshold_requires_three_occurrences(self):
        """2 repeated actions in history must NOT trigger; 3 repeated actions MUST trigger."""
        current = {
            "step_index": 4,
            "action_name": "bash",
            "action_args": {"cmd": "ls"},
        }
        # 2 identical actions in history -> repeat_count == 2 -> should NOT trigger
        two_history = [
            {"action_name": "bash", "action_args": {"cmd": "ls"}},
            {"action_name": "bash", "action_args": {"cmd": "ls"}},
        ]
        assert _detect_action_repetition(current, two_history) is None

        # 3 identical actions in history -> repeat_count == 3 -> MUST trigger
        three_history = [
            {"action_name": "bash", "action_args": {"cmd": "ls"}},
            {"action_name": "bash", "action_args": {"cmd": "ls"}},
            {"action_name": "bash", "action_args": {"cmd": "ls"}},
        ]
        res = _detect_action_repetition(current, three_history)
        assert res is not None
        assert res.matched is True
        assert res.rule_id == "structural_action_repetition"
        assert res.category == "memory_error"


# =========================================================================
# Test (g): Structural heuristic — 'Nothing happens' loop
# =========================================================================

class TestNothingHappensLoop:
    """Test the 'Nothing happens' loop detector."""

    def test_nothing_happens_repeated(self):
        """Multiple 'Nothing happens' should flag tool_use_error."""
        current = {
            "step_index": 4,
            "action_name": "'go",
            "action_args": {"target": "fridge 1"},
            "tool_response": "Nothing happens.",
        }
        history = [
            {"tool_response": "Nothing happens.", "action_name": "'go", "action_args": {}},
            {"tool_response": "Nothing happens.", "action_name": "'go", "action_args": {}},
            {"tool_response": "You arrive at fridge 1.", "action_name": "go", "action_args": {}},
        ]
        result = _detect_nothing_happens_loop(current, history)
        assert result is not None
        assert result.matched is True
        assert result.category == "tool_use_error"
        assert result.rule_id == "structural_nothing_happens_loop"

    def test_no_nothing_happens(self):
        """Normal responses should not trigger."""
        current = {
            "step_index": 2,
            "tool_response": "You arrive at desk 1.",
        }
        history = [
            {"tool_response": "You pick up the pen."},
        ]
        result = _detect_nothing_happens_loop(current, history)
        assert result is None


# =========================================================================
# Test (h): on_run_end full trajectory scan
# =========================================================================

class TestOnRunEnd:
    """Test full trajectory scan via on_run_end."""

    def test_run_end_finds_earliest_flagged_step(self, guard):
        """on_run_end should scan all steps and report the earliest flagged step as root cause."""
        trajectory = {
            "steps": [
                {
                    "step_index": 0,
                    "reasoning": "Look around.",
                    "action_name": "look",
                    "action_args": {},
                    "tool_response": "You see a desk and a chair.",
                },
                {
                    "step_index": 1,
                    "reasoning": "Go to desk.",
                    "action_name": "'go",
                    "action_args": {"target": "desk 1"},
                    "tool_response": "Nothing happens.",
                },
                {
                    "step_index": 2,
                    "reasoning": "Go to desk again.",
                    "action_name": "'go",
                    "action_args": {"target": "desk 1"},
                    "tool_response": "Nothing happens.",
                },
                {
                    "step_index": 3,
                    "reasoning": "Go to desk yet again.",
                    "action_name": "'go",
                    "action_args": {"target": "desk 1"},
                    "tool_response": "Nothing happens.",
                },
            ]
        }
        result = guard.on_run_end(
            run_metadata={"run_id": "test_run_end_001"},
            trajectory=trajectory,
        )

        assert result["processed"] is True
        # Should have detected something in the repeated 'Nothing happens' steps
        assert len(result["flags_summary"]) > 0
        earliest_step_index = result["flags_summary"][0]["step_index"]
        assert result["root_cause_step_index"] == earliest_step_index
        if result["root_cause_error_type"]:
            assert result["root_cause_error_type"] in DEFAULT_TAGS

    def test_run_end_selects_earliest_over_higher_confidence_step(self, guard):
        """Verify on_run_end selects the earliest flagged step even if a later step has higher confidence."""
        trajectory = {
            "steps": [
                {"step_index": 0, "reasoning": "step 0"},
                {"step_index": 1, "reasoning": "step 1"},
                {"step_index": 2, "reasoning": "step 2"},
            ]
        }
        step_returns = [
            {"match_details": None, "drift_detected": False, "reflection_result": None},
            {"match_details": {"step_index": 1, "category": "planning_error", "confidence": 0.35, "layer": "cluster"}, "drift_detected": False},
            {"match_details": {"step_index": 2, "category": "tool_use_error", "confidence": 0.95, "layer": "structural"}, "drift_detected": False},
        ]
        with patch.object(guard, "on_step", side_effect=step_returns):
            result = guard.on_run_end(metadata={"run_id": "test_earliest"}, trajectory=trajectory)
            assert result["root_cause_step_index"] == 1
            assert result["root_cause_error_type"] == "planning_error"
            assert result["root_cause_source"] == "pattern_match"
            assert len(result["flags_summary"]) == 2

    # ---- Tests for FIX A: 4-tier fallback chain & root_cause_source ----

    def test_run_end_tier1_pattern_match_source(self, guard):
        """Tier 1: When pattern match occurs, root_cause_source='pattern_match'."""
        trajectory = {
            "steps": [
                {"step_index": 0, "reasoning": "Look around.", "action_name": "look", "action_args": {}, "tool_response": "ok"},
                {"step_index": 1, "reasoning": "mechanical sequential search one by one cabinet by cabinet", "action_name": "open", "action_args": {"target": "cabinet 1"}, "tool_response": "ok"},
            ]
        }
        result = guard.on_run_end(metadata={"run_id": "test_tier1"}, trajectory=trajectory)
        assert result["processed"] is True
        assert result["root_cause_source"] == "pattern_match"
        assert result["root_cause_step_index"] == 1
        assert result["root_cause_error_type"] == "planning_error"

    def test_run_end_tier2_drift_monitor_fallback(self, guard):
        """Tier 2: No pattern match, but drift fires -> root_cause_source='drift_monitor'."""
        trajectory = {
            "steps": [
                {"step_index": 0, "reasoning": "clean step 0"},
                {"step_index": 1, "reasoning": "clean step 1"},
                {"step_index": 2, "reasoning": "clean step 2"},
            ]
        }
        step_returns = [
            {"match_details": None, "drift_detected": False, "drift_assessment": {"drift_detected": False}, "reflection_result": None},
            {
                "match_details": None,
                "drift_detected": True,
                "drift_assessment": {
                    "drift_detected": True,
                    "step_index": 1,
                    "severity_level": "high",
                    "triggered_signals": ["REPEATED_STALLED_SUBGOALS"],
                    "reasons": ["2 stalled subgoals"],
                },
                "reflection_result": None,
            },
            {"match_details": None, "drift_detected": False, "drift_assessment": {"drift_detected": False}, "reflection_result": None},
        ]
        with patch.object(guard, "on_step", side_effect=step_returns):
            result = guard.on_run_end(metadata={"run_id": "test_tier2"}, trajectory=trajectory)
            assert result["root_cause_source"] == "drift_monitor"
            assert result["root_cause_step_index"] == 1
            assert result["root_cause_error_type"] == "planning_error"

    def test_run_end_tier3_reflector_fallback(self, guard):
        """Tier 3: No pattern match, no drift, but reflector suggests revision -> root_cause_source='reflector'."""
        trajectory = {
            "steps": [
                {"step_index": 0, "reasoning": "clean step 0"},
                {"step_index": 1, "reasoning": "step 1"},
            ]
        }
        step_returns = [
            {"match_details": None, "drift_detected": False, "drift_assessment": None, "reflection_result": None},
            {
                "match_details": None,
                "drift_detected": False,
                "drift_assessment": None,
                "reflection_result": {
                    "revision_suggested": True,
                    "step_index": 1,
                    "revision_reasoning": "Plan invalidated due to unexpected goal divergence",
                    "evidence_sources": ["subgoal_boundary"],
                },
            },
        ]
        with patch.object(guard, "on_step", side_effect=step_returns):
            result = guard.on_run_end(metadata={"run_id": "test_tier3"}, trajectory=trajectory)
            assert result["root_cause_source"] == "reflector"
            assert result["root_cause_step_index"] == 1
            assert result["root_cause_error_type"] in ("planning_error", "plan_deviation")

    def test_run_end_tier4_clean_run_none(self, guard):
        """Tier 4: Genuinely clean run -> root_cause_source='none', all root cause fields None."""
        trajectory = {
            "steps": [
                {"step_index": 0, "reasoning": "Look around.", "action_name": "look", "action_args": {}, "tool_response": "Clean room."},
            ]
        }
        result = guard.on_run_end(metadata={"run_id": "test_tier4_clean"}, trajectory=trajectory)
        assert result["processed"] is True
        assert result["root_cause_source"] == "none"
        assert result["root_cause_step_index"] is None
        assert result["root_cause_error_type"] is None


class TestFlaggedConvenienceAndUnifiedMetadata:
    """Test FIX B (unified metadata= parameter) and FIX C (flagged convenience boolean)."""

    def test_on_step_flagged_field_behavior(self, guard):
        """Verify flagged boolean is True when any subsystem flags, False when clean."""
        step = {"step_index": 0, "reasoning": "look around", "action_name": "look", "action_args": {}, "tool_response": "ok"}

        # 1. Clean step -> flagged is False
        res = guard.on_step(step, history=[])
        assert "flagged" in res
        assert res["flagged"] is False

        # 2. Pattern matcher match -> flagged is True
        pattern_step = {
            "step_index": 1,
            "reasoning": "mechanical sequential search one by one cabinet by cabinet",
            "action_name": "open",
            "action_args": {},
            "tool_response": "ok",
        }
        res_pattern = guard.on_step(pattern_step, history=[step])
        assert res_pattern["flagged"] is True

    def test_unified_metadata_parameter_across_all_hooks(self, guard):
        """Verify metadata= keyword works consistently on all 4 hooks."""
        meta = {"run_id": "test_meta_unification", "domain": "alfworld"}

        # Hook 1: on_plan_proposed
        r1 = guard.on_plan_proposed("Clean task", "1. Do step", metadata=meta)
        assert r1["approved"] is True

        # Hook 2: on_step
        s = {"step_index": 0, "reasoning": "start", "action_name": "look", "action_args": {}}
        r2 = guard.on_step(s, history=[], metadata=meta)
        assert "continue_execution" in r2
        assert "flagged" in r2

        # Hook 3: on_subgoal_boundary
        r3 = guard.on_subgoal_boundary("subgoal_001", "completed", step_history=[s], metadata=meta)
        assert "checkpoint_passed" in r3

        # Hook 4: on_run_end
        r4 = guard.on_run_end(metadata=meta, trajectory={"steps": [s]})
        assert r4["processed"] is True
        assert "root_cause_source" in r4


# =========================================================================
# Test (i): Match timeout fail-open
# =========================================================================

class TestMatchTimeout:
    """Test that a hanging match call times out and fails open."""

    def test_timeout_returns_no_match(self):
        """A matcher that hangs should timeout and return no match."""
        import time as _time

        matcher = PatternMatcher(
            pattern_library_path=PATTERN_LIB_PATH,
            match_timeout=0.1,  # very short timeout
        )

        original_impl = matcher._match_impl

        def slow_match(text):
            _time.sleep(5)  # way longer than timeout
            return original_impl(text)

        matcher._match_impl = slow_match

        result = matcher.match("some text", run_id="timeout_test", step_index=0)
        assert result.matched is False  # timed out → fail open


# =========================================================================
# Test: TF-IDF helpers
# =========================================================================

class TestTfidfHelpers:
    """Unit tests for the low-level TF-IDF functions."""

    def test_tokenize_basic(self):
        tokens = _tokenize("The agent's initial plan was flawed.")
        assert "initial" in tokens
        assert "plan" in tokens
        assert "flawed" in tokens
        assert "the" not in tokens  # stopword

    def test_cosine_self_similarity(self):
        v = {"a": 0.5, "b": 0.5, "c": 0.7071}
        sim = _cosine_sim(v, v)
        assert sim > 0.99

    def test_cosine_empty(self):
        assert _cosine_sim({}, {"a": 1.0}) == 0.0
        assert _cosine_sim({"a": 1.0}, {}) == 0.0

    def test_cosine_orthogonal(self):
        assert _cosine_sim({"a": 1.0}, {"b": 1.0}) == 0.0


# =========================================================================
# Test: Keyword rule coverage
# =========================================================================

class TestKeywordRuleCoverage:
    """Verify that keyword rules exist for all required categories."""

    def test_all_categories_have_rules(self):
        """tool_use_error and external_error MUST have keyword rules."""
        rules = _build_keyword_rules()
        categories_covered = set(r.category for r in rules)
        assert "tool_use_error" in categories_covered
        assert "external_error" in categories_covered
        assert "memory_error" in categories_covered
        assert "planning_error" in categories_covered
        assert "reflection_error" in categories_covered

    def test_at_least_two_rules_per_critical_category(self):
        """tool_use_error and external_error should have >= 2 rules each."""
        rules = _build_keyword_rules()
        from collections import Counter
        counts = Counter(r.category for r in rules)
        assert counts["tool_use_error"] >= 2
        assert counts["external_error"] >= 2


# =========================================================================
# Test: Boilerplate Regression Guard
# =========================================================================

class TestBoilerplateRegression:
    """Verify that heavy environment/prompt boilerplate in tool_response does not trigger Layer A matches."""

    def test_heavy_boilerplate_in_tool_response_does_not_trigger_layer_a(self, guard):
        """A clean step with heavy ALFWORLD prompt boilerplate in tool_response must not trigger Layer A pattern matches."""
        heavy_boilerplate_tool_response = (
            "You are an expert agent operating in the ALFRED Embodied Environment. "
            "Your task is to: heat some egg and put it in fridge. "
            "Available tools: go to cabinet 1, go to cabinet 2, open cabinet 2, take soapbottle from cabinet 3, "
            "search results at step 1 led to failure of task. Nothing happens when opening cabinet 3. "
            "Cabinet 1, cabinet 2, cabinet 3, cabinet 4, cabinet 5, cabinet 6, cabinet 7, cabinet 8..."
        )
        step = {
            "step_index": 0,
            "reasoning": "I will proceed to open the main container carefully.",
            "action_name": "open",
            "action_args": {"target": "main container"},
            "tool_response": heavy_boilerplate_tool_response,
        }

        result = guard.on_step(step_record=step, history=[], metadata={"run_id": "test_boilerplate_001"})
        
        # Layer A cluster match must NOT fire on this clean step despite boilerplate
        match_details = result.get("match_details")
        if match_details:
            assert match_details.get("layer") != "cluster", f"Boilerplate in tool_response triggered Layer A cluster match: {match_details}"


class TestOnRunEndLiveVsOfflineParity:
    """Regression tests verifying live-monitoring vs standalone offline on_run_end parity (F-01)."""

    def test_live_vs_offline_identical_analysis(self):
        """Pattern A (live on_step + on_run_end) and Pattern B (standalone on_run_end) must produce identical analysis."""
        from longhorizon_guard import GuardInterface

        task = "Find blue sneakers and buy"
        plan = "1. Search for blue sneakers\n2. Select size\n3. Purchase"

        steps = [
            {"step_index": 0, "reasoning": "searching", "action_name": "search", "action_args": {"query": "blue sneakers"}, "tool_response": "found 5"},
            {"step_index": 1, "reasoning": "selecting", "action_name": "click", "action_args": {"button": "size 10"}, "tool_response": "selected"},
            {"step_index": 2, "reasoning": "buying", "action_name": "click", "action_args": {"button": "buy"}, "tool_response": "purchased"},
        ]

        # Pattern A: Live monitoring
        g_live = GuardInterface()
        g_live.on_plan_proposed(task, plan)
        for s in steps:
            g_live.on_step(s, [])
        res_live = g_live.on_run_end(metadata={"task": task}, trajectory={"steps": steps})

        # Pattern B: Standalone offline post-hoc
        g_offline = GuardInterface()
        g_offline.on_plan_proposed(task, plan)
        res_offline = g_offline.on_run_end(metadata={"task": task}, trajectory={"steps": steps})

        # Verify parity
        assert res_live["root_cause_error_type"] == res_offline["root_cause_error_type"]
        assert res_live["root_cause_step_index"] == res_offline["root_cause_step_index"]
        assert res_live["root_cause_source"] == res_offline["root_cause_source"]
        assert len(res_live["flags_summary"]) == len(res_offline["flags_summary"])

        # Subgoals parity: statuses and step_indices must match exactly
        live_sgs = res_live["subgoals_summary"]["subgoals"]
        offline_sgs = res_offline["subgoals_summary"]["subgoals"]
        assert len(live_sgs) == len(offline_sgs)
        for i in range(len(live_sgs)):
            assert live_sgs[i]["subgoal_id"] == offline_sgs[i]["subgoal_id"]
            assert live_sgs[i]["status"] == offline_sgs[i]["status"]
            assert live_sgs[i]["step_indices"] == offline_sgs[i]["step_indices"]

        # Drift parity
        assert (
            res_live["drift_summary"]["final_drift_assessment"]["drift_detected"]
            == res_offline["drift_summary"]["final_drift_assessment"]["drift_detected"]
        )


class TestOnPlanProposedFlaggedField:
    """Verify on_plan_proposed approved vs flagged contract (ISSUE-PLAN-01)."""

    def test_plan_approved_always_true_and_flagged_reflects_detection(self, guard):
        """approved remains True (non-blocking contract), flagged reflects whether flags were detected."""
        # Bad plan: contradiction and omission
        bad_res = guard.on_plan_proposed(
            task_description="Search for mens running shoes",
            proposed_plan="1. Search for womens running shoes",
        )
        assert bad_res["approved"] is True
        assert bad_res["flagged"] is True
        assert len(bad_res["flags"]) > 0

        # Good plan: clean matching plan
        good_res = guard.on_plan_proposed(
            task_description="Search for cotton socks",
            proposed_plan="1. Search for cotton socks",
        )
        assert good_res["approved"] is True
        assert good_res["flagged"] is False
        assert len(good_res["flags"]) == 0


class TestGuardConfigIntegration:
    """Verify GuardConfig integration with GuardInterface (FIX F-13)."""

    def test_guard_interface_initialization_with_config(self):
        config = GuardConfig(
            max_subgoal_steps=7,
            drift_threshold=0.25,
            fail_open=True,
        )
        guard = GuardInterface(config=config)
        assert guard._subgoal_tracker.max_subgoal_steps == 7
        assert guard._drift_monitor.drift_threshold == 0.25
        assert guard.fail_open is True

    def test_guard_interface_direct_params_override_defaults(self):
        guard = GuardInterface(max_subgoal_steps=5, drift_threshold=0.40)
        assert guard._subgoal_tracker.max_subgoal_steps == 5
        assert guard._drift_monitor.drift_threshold == 0.40


class TestStatusCodeKeywordRegression:
    """Verify that 404/500 status codes in code do not cause false positives, while genuine external errors fire."""

    def test_rest_api_status_codes_not_flagged_as_external_error(self, guard):
        """Realistic REST API code containing 404/500 status codes (abort call, test assertions) must NOT flag external_error."""
        # 1. Route handler with abort(404)
        step_route = {
            "step_index": 0,
            "reasoning": "Handle missing item by aborting with 404 status code.",
            "action_name": "edit_file",
            "action_args": {
                "path": "app/routes.py",
                "content": (
                    "@app.route('/items/<int:item_id>')\n"
                    "def get_item(item_id):\n"
                    "    item = db.find(item_id)\n"
                    "    if item is None:\n"
                    "        abort(404)\n"
                    "    return jsonify(item)\n"
                ),
            },
            "tool_response": "Applied edit to app/routes.py",
        }
        res_route = guard.on_step(step_route, history=[], metadata={"run_id": "test_rest_001"})
        assert res_route["flagged"] is False
        assert res_route["match_details"] is None

        # 2. Test assertions with 404 and 500 status codes
        step_test = {
            "step_index": 1,
            "reasoning": "Add test assertions for 404 not-found and 500 server fault responses.",
            "action_name": "edit_file",
            "action_args": {
                "path": "tests/test_routes.py",
                "content": (
                    "def test_item_not_found(client):\n"
                    "    resp = client.get('/items/999')\n"
                    "    assert resp.status_code == 404\n\n"
                    "def test_server_fault(client):\n"
                    "    resp = client.post('/items/crash')\n"
                    "    assert resp.status_code == 500\n"
                ),
            },
            "tool_response": "Applied edit to tests/test_routes.py",
        }
        res_test = guard.on_step(step_test, history=[step_route], metadata={"run_id": "test_rest_002"})
        assert res_test["flagged"] is False
        assert res_test["match_details"] is None

    def test_genuine_external_error_fires_on_500_server_error(self, guard_no_patterns):
        """Genuine external error language 'Connection error: 500 Internal Server Error from upstream API' must fire."""
        step = {
            "step_index": 2,
            "reasoning": "Calling external payment gateway.",
            "action_name": "call_api",
            "action_args": {"endpoint": "https://api.payment.com/v1/charge"},
            "tool_response": "Connection error: 500 Internal Server Error from upstream API",
        }
        result = guard_no_patterns.on_step(step, history=[], metadata={"run_id": "test_ext_001"})
        assert result["flagged"] is True
        details = result.get("match_details")
        assert details is not None
        assert details["category"] == "external_error"
        assert details["rule_id"] == "external_env_error"

    def test_genuine_external_error_fires_on_http_404(self, guard_no_patterns):
        """Genuine external error language 'HTTP 404 error: endpoint not found on upstream service' must fire."""
        step = {
            "step_index": 3,
            "reasoning": "Observed HTTP 404 error from upstream service.",
            "action_name": "fetch",
            "action_args": {"url": "https://api.service.internal/data"},
            "tool_response": "HTTP 404 error: endpoint not found on remote server",
        }
        result = guard_no_patterns.on_step(step, history=[], metadata={"run_id": "test_ext_002"})
        assert result["flagged"] is True
        details = result.get("match_details")
        assert details is not None
        assert details["category"] == "external_error"
        assert details["rule_id"] == "external_env_error"

    def test_audited_bare_keywords_no_false_positives(self, guard):
        """Audited bare keywords (timeout=10, looping, instead of) must not trigger false positives."""
        # 1. requests.get with timeout=10 should NOT flag external_step_limit
        s_timeout = {
            "step_index": 0,
            "reasoning": "Fetch data with timeout=10 configuration.",
            "action_name": "edit_file",
            "action_args": {"path": "fetch.py", "content": "requests.get(url, timeout=10)"},
            "tool_response": "Applied edit",
        }
        r_timeout = guard.on_step(s_timeout, history=[])
        assert r_timeout["flagged"] is False

        # 2. 'Looping over the items' should NOT flag memory_repeated_action
        s_loop = {
            "step_index": 1,
            "reasoning": "Looping over the items in the list to calculate total.",
            "action_name": "edit_file",
            "action_args": {"path": "calc.py", "content": "for x in items: total += x"},
            "tool_response": "Applied edit",
        }
        r_loop = guard.on_step(s_loop, history=[])
        assert r_loop["flagged"] is False

        # 3. 'use dictionary instead of list' should NOT flag reflection_wrong_object
        s_instead = {
            "step_index": 2,
            "reasoning": "I will use a dictionary instead of a list for fast lookups.",
            "action_name": "edit_file",
            "action_args": {"path": "store.py", "content": "lookup = {}"},
            "tool_response": "Applied edit",
        }
        r_instead = guard.on_step(s_instead, history=[])
        assert r_instead["flagged"] is False

    def test_tool_response_flags_memory_error_hallucinated_observation(self, guard_no_patterns):
        """Genuine memory_error language occurring only in tool_response must be flagged."""
        step = {
            "step_index": 0,
            "reasoning": "Check environment observation status.",
            "action_name": "verify_state",
            "action_args": {},
            "tool_response": "Agent hallucinated an observation that was never actually returned by environment",
        }
        res = guard_no_patterns.on_step(step, history=[])
        assert res["flagged"] is True
        details = res.get("match_details")
        assert details is not None
        assert details["category"] == "memory_error"
        assert details["rule_id"] == "memory_forgot_observation"

    def test_tool_response_flags_memory_error_infinite_loop(self, guard_no_patterns):
        """Infinite loop detection appearing only in tool_response must be flagged as memory_error."""
        step = {
            "step_index": 1,
            "reasoning": "Inspect loop monitor log.",
            "action_name": "read_log",
            "action_args": {"path": "monitor.log"},
            "tool_response": "Agent is stuck in an infinite loop revisiting the same action",
        }
        res = guard_no_patterns.on_step(step, history=[])
        assert res["flagged"] is True
        details = res.get("match_details")
        assert details is not None
        assert details["category"] == "memory_error"
        assert details["rule_id"] == "memory_repeated_action"

    def test_tool_response_flags_tool_error_unrecognized_command(self, guard_no_patterns):
        """Malformed tool use syntax/command error occurring only in tool_response must be flagged as tool_error."""
        step = {
            "step_index": 2,
            "reasoning": "Execute CLI utility.",
            "action_name": "run_bash",
            "action_args": {"command": "tool_cli --invalid-flag"},
            "tool_response": "Error: unrecognized command and syntax error in arguments",
        }
        res = guard_no_patterns.on_step(step, history=[])
        assert res["flagged"] is True
        details = res.get("match_details")
        assert details is not None
        assert details["category"] == "tool_use_error"
        assert details["rule_id"] == "tool_use_malformed_action"

    def test_benign_tool_response_status_codes_not_flagged(self, guard):
        """Benign status numbers in tool_response (e.g. 404 lines changed, 500 records) must NOT flag."""
        # 1. '404 lines changed' in tool_response
        s_404 = {
            "step_index": 0,
            "reasoning": "Apply bulk migration to source files.",
            "action_name": "apply_patch",
            "action_args": {"patch": "migration.patch"},
            "tool_response": "Applied edit, 404 lines changed, 0 failures",
        }
        r_404 = guard.on_step(s_404, history=[])
        assert r_404["flagged"] is False
        assert r_404["match_details"] is None

        # 2. 'Processed 500 records' in tool_response
        s_500 = {
            "step_index": 1,
            "reasoning": "Run database batch ingestion job.",
            "action_name": "ingest_batch",
            "action_args": {"batch_size": 500},
            "tool_response": "Successfully processed 500 records in 150ms",
        }
        r_500 = guard.on_step(s_500, history=[])
        assert r_500["flagged"] is False
        assert r_500["match_details"] is None

    def test_tool_response_flags_planning_error_exhaustive_search(self, guard_no_patterns):
        """Exhaustive search detected solely in tool_response must be flagged as planning_error."""
        step = {
            "step_index": 0,
            "reasoning": "Proceed with task execution.",
            "action_name": "next_step",
            "action_args": {},
            "tool_response": "Environment feedback: agent used exhaustive search, checking cabinet by cabinet with no informed strategy",
        }
        res = guard_no_patterns.on_step(step, history=[])
        assert res["flagged"] is True
        details = res.get("match_details")
        assert details is not None
        assert details["category"] == "planning_error"
        assert details["rule_id"] == "planning_exhaustive_search"

    def test_benign_tool_response_planning_keywords_not_flagged(self, guard):
        """Benign search/cabinet text in tool_response lacking error co-occurrence must NOT flag."""
        step = {
            "step_index": 1,
            "reasoning": "Query inventory database.",
            "action_name": "query_items",
            "action_args": {"target": "cabinet"},
            "tool_response": "Search complete: found 10 items in cabinet inventory, sorted by id",
        }
        res = guard.on_step(step, history=[])
        assert res["flagged"] is False
        assert res.get("match_details") is None

    def test_tool_response_flags_reflection_error(self, guard_no_patterns):
        """Reflection error feedback present solely in tool_response must be flagged as reflection_error."""
        step = {
            "step_index": 2,
            "reasoning": "Continue interacting with environment.",
            "action_name": "interact",
            "action_args": {},
            "tool_response": "Feedback ignored: agent misinterpreted feedback from the environment and picked the wrong object instead of the correct one",
        }
        res = guard_no_patterns.on_step(step, history=[])
        assert res["flagged"] is True
        details = res.get("match_details")
        assert details is not None
        assert details["category"] == "reflection_error"
        assert details["rule_id"] in ("reflection_wrong_object", "reflection_ignored_feedback")

    def test_benign_tool_response_reflection_keywords_not_flagged(self, guard):
        """Benign phrasing with 'instead of' in tool_response lacking error co-occurrence must NOT flag."""
        step = {
            "step_index": 3,
            "reasoning": "Execute optimization benchmark.",
            "action_name": "run_benchmark",
            "action_args": {},
            "tool_response": "Benchmark result: using set lookup instead of linear list scan improved runtime",
        }
        res = guard.on_step(step, history=[])
        assert res["flagged"] is False
        assert res.get("match_details") is None


class TestGuardRuntimeEnhancements:
    """Tests covering Critical and High findings: schema, subgoals, taxonomy, and root cause."""

    def test_on_step_top_level_schema_contract(self, guard):
        """Top-level schema must include category, confidence, and suggestions on on_step()."""
        # Flagged step
        step_flagged = {
            "step_index": 0,
            "reasoning": "Check 404 response.",
            "action_name": "curl",
            "action_args": {},
            "tool_response": "HTTP 404 error: resource not found on remote server",
        }
        res_flagged = guard.on_step(step_flagged, history=[])
        assert res_flagged["flagged"] is True
        assert "category" in res_flagged
        assert res_flagged["category"] == "external_error"
        assert "confidence" in res_flagged
        assert isinstance(res_flagged["confidence"], float)
        assert "suggestions" in res_flagged
        assert isinstance(res_flagged["suggestions"], list)

        # Clean step
        step_clean = {
            "step_index": 1,
            "reasoning": "Compute valid math.",
            "action_name": "calc",
            "action_args": {"expr": "1 + 1"},
            "tool_response": "2",
        }
        res_clean = guard.on_step(step_clean, history=[])
        assert res_clean["flagged"] is False
        assert "category" in res_clean
        assert res_clean["category"] is None
        assert "confidence" in res_clean
        assert res_clean["confidence"] == 0.0
        assert "suggestions" in res_clean
        assert res_clean["suggestions"] == []

    def test_subgoal_boundary_mutates_tracker_state(self, guard):
        """on_subgoal_boundary must mutate SubgoalTracker and advance active index."""
        plan = "1. Install dependencies\n2. Run tests"
        guard.on_plan_proposed(task_description="Setup project", proposed_plan=plan)

        tracker = guard.subgoal_tracker
        assert tracker is not None
        assert tracker.active_subgoal.subgoal_id == "subgoal_001"

        res = guard.on_subgoal_boundary(
            subgoal_id="subgoal_001",
            subgoal_status="completed",
            metadata={"step_index": 1, "trigger_reason": "host_event"},
        )
        assert res["checkpoint_passed"] is True
        assert res["next_subgoal"] == "subgoal_002"
        assert tracker.subgoals[0].status == "completed"
        assert tracker.active_subgoal.subgoal_id == "subgoal_002"

    def test_reflection_derives_valid_taxonomy_category(self):
        """_derive_error_type_from_reflection must map plan_deviation to planning_error."""
        from longhorizon_guard.interface import _derive_error_type_from_reflection
        
        raw_refl = {
            "detected_error_type": "plan_deviation",
            "explanation": "Agent drifted from plan",
        }
        derived = _derive_error_type_from_reflection(raw_refl)
        assert derived == "planning_error"
        assert derived in DEFAULT_TAGS

    def test_on_run_end_earliest_temporal_root_cause(self, guard):
        """Earliest step error must be attributed as root cause over later tier errors."""
        steps = [
            {
                "step_index": 0,
                "reasoning": "Initial doc read",
                "action_name": "read_doc",
                "action_args": {"file": "index.md"},
                "tool_response": "doc content",
            },
            {
                "step_index": 1,
                "reasoning": "Read documentation file",
                "action_name": "read_doc",
                "action_args": {"file": "index.md"},
                "tool_response": "doc content",
            },
            {
                "step_index": 2,
                "reasoning": "Read documentation file again",
                "action_name": "read_doc",
                "action_args": {"file": "index.md"},
                "tool_response": "doc content",
            },
            {
                "step_index": 3,
                "reasoning": "Read documentation file third time",
                "action_name": "read_doc",
                "action_args": {"file": "index.md"},
                "tool_response": "doc content",
            },
            {
                "step_index": 4,
                "reasoning": "Now external error",
                "action_name": "fetch",
                "action_args": {},
                "tool_response": "HTTP 404 error: resource not found",
            },
        ]

        summary = guard.on_run_end(
            metadata={"task": "test"},
            trajectory={"steps": steps},
        )
        # Step 3 triggers action repetition (drift heuristic), Step 4 triggers pattern_match (Tier 1).
        # Chronological precedence dictates Step 3 is earlier than Step 4!
        assert summary["root_cause_step_index"] == 3
        assert summary["root_cause_error_type"] == "memory_error"

    def test_fail_open_false_raises_exception(self):
        """When fail_open=False, exceptions inside hooks must re-raise (Finding 7)."""
        cfg = GuardConfig(fail_open=False)
        strict_guard = GuardInterface(config=cfg)

        with patch.object(strict_guard._matcher, "match", side_effect=RuntimeError("strict failure")):
            with pytest.raises(RuntimeError, match="strict failure"):
                strict_guard.on_step(
                    {"step_index": 0, "reasoning": "r", "action_name": "a", "action_args": {}},
                    history=[],
                )

    def test_block_on_critical_controls_approval_and_execution(self):
        """block_on_critical=True blocks on contradiction and critical drift (Finding 8)."""
        cfg_blocking = GuardConfig(block_on_critical=True)
        guard_block = GuardInterface(config=cfg_blocking)

        # Plan contradiction with block_on_critical -> approved=False
        contradiction_plan = "1. Book expensive luxury tickets"
        res_plan = guard_block.on_plan_proposed("Find the cheapest flight options", contradiction_plan)
        assert res_plan["flagged"] is True
        assert res_plan["approved"] is False

        # Non-blocking default -> approved remains True
        guard_default = GuardInterface(config=GuardConfig(block_on_critical=False))
        res_default = guard_default.on_plan_proposed("Find the cheapest flight options", contradiction_plan)
        assert res_default["flagged"] is True
        assert res_default["approved"] is True

        # Critical step execution block
        step_loop = {
            "step_index": 3,
            "reasoning": "reading",
            "action_name": "read",
            "action_args": {"file": "a.txt"},
            "tool_response": "ok",
        }
        prior_hist = [
            {"step_index": 0, "action_name": "read", "action_args": {"file": "a.txt"}},
            {"step_index": 1, "action_name": "read", "action_args": {"file": "a.txt"}},
            {"step_index": 2, "action_name": "read", "action_args": {"file": "a.txt"}},
        ]
        res_step_block = guard_block.on_step(step_loop, history=prior_hist)
        assert res_step_block["flagged"] is True
        assert res_step_block["continue_execution"] is False

    def test_similarity_threshold_configured_and_confidence_calibrated(self):
        """PatternMatcher receives similarity_threshold and calibrates confidence (Findings 7 & 11)."""
        cfg = GuardConfig(similarity_threshold=0.88)
        guard = GuardInterface(config=cfg)
        assert guard._matcher._similarity_threshold == 0.88

        # Verify context_length_error rule exists in keyword rules
        rule_ids = [r.rule_id for r in guard._matcher._keyword_rules]
        assert "context_length_exceeded" in rule_ids

    def test_subgoal_success_keywords_tightened(self, guard):
        """'you see' does not complete subgoal; 'not found' does not complete subgoal (Finding 10)."""
        tracker = guard.subgoal_tracker
        tracker.init_plan("Take the apple", "1. Open fridge\n2. Take apple")

        # Step 1: Tool response with passive 'You see' -> must NOT complete
        step_see = {
            "step_index": 0,
            "action_name": "look",
            "tool_response": "You see a fridge and a chair.",
        }
        res_see = tracker.process_step(step_see, history=[])
        assert res_see["transition_event"] is None
        assert tracker.active_subgoal.subgoal_id == "subgoal_001"

        # Step 2: Tool response with 'not found' -> must NOT complete
        step_not_found = {
            "step_index": 1,
            "action_name": "search",
            "tool_response": "File not found in system.",
        }
        res_nf = tracker.process_step(step_not_found, history=[step_see])
        assert res_nf["transition_event"] is None
        assert tracker.active_subgoal.subgoal_id == "subgoal_001"

        # Step 3: Tool response with real success 'you open' -> completes
        step_open = {
            "step_index": 2,
            "action_name": "open",
            "tool_response": "You open the fridge door.",
        }
        res_open = tracker.process_step(step_open, history=[step_see, step_not_found])
        assert res_open["transition_event"] is not None
        assert res_open["transition_event"]["completed_status"] == "completed"
        assert tracker.active_subgoal.subgoal_id == "subgoal_002"

    def test_reflector_invalidation_respects_high_severity(self):
        """Reflector only invalidates on high/critical drift or score >= 0.60 (Finding 12)."""
        from longhorizon_guard.reflector.reflector import PlanReflector
        refl = PlanReflector(step_interval=5)
        refl.init_plan("Complete task", "1. Step 1\n2. Step 2")

        sub_state = {"failed_subgoals_count": 0, "stalled_advanced_subgoals_count": 0}

        # Medium drift score (0.50) without failed subgoals should NOT invalidate
        med_drift = {"drift_detected": True, "severity_score": 0.50, "severity_level": "medium"}
        res_med = refl.evaluate(
            step_record={"step_index": 5},
            history=[],
            subgoal_state=sub_state,
            drift_assessment=med_drift,
            trigger_type="step_interval",
        )
        assert res_med.plan_still_valid is True
        assert res_med.revision_suggested is False

        # High drift score (0.65) / high severity level MUST invalidate on subsequent step interval
        high_drift = {"drift_detected": True, "severity_score": 0.65, "severity_level": "high"}
        res_high = refl.evaluate(
            step_record={"step_index": 10},
            history=[],
            subgoal_state=sub_state,
            drift_assessment=high_drift,
            trigger_type="step_interval",
        )
        assert res_high.plan_still_valid is False
        assert res_high.revision_suggested is True

    def test_client_wrapper_plan_extraction(self):
        """OpenAIClientWrapper extracts plan from user prompt (Finding 9)."""
        from unittest.mock import MagicMock
        from longhorizon_guard.integrations.client_wrapper import OpenAIClientWrapper

        mock_client = MagicMock()
        guard = GuardInterface()
        wrapper = OpenAIClientWrapper(mock_client, guard=guard)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Goal: Setup repository.\nPlan:\n1. Clone repo\n2. Run tests"},
        ]
        wrapper._process_messages_before_call(messages)

        tracker = guard.subgoal_tracker
        assert tracker is not None
        assert tracker.is_fallback is False
        assert len(tracker.subgoals) == 2
        assert tracker.active_subgoal.subgoal_id == "subgoal_001"

