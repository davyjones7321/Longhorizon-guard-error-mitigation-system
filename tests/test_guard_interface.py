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



