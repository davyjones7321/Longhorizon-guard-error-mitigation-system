"""
Production GuardInterface for longhorizon_guard.

Implements the 4-method hook interface with two-layer failure pattern matching:
  (a) Primary:  TF-IDF cosine-similarity against pattern_library.json centroids
  (b) Fallback: Per-category keyword/heuristic rules

Design principles:
  - FAIL OPEN on any exception: log the error, let the agent proceed.
  - Structured logging via stdlib `logging` (not print).
  - Matching timeout via threading.
  - Per-category confidence thresholds calibrated from Phase 4 propagation accuracy.
"""

import json
import logging
import math
import re
import time
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from longhorizon_guard.pattern_library.schema import PatternEntry
from longhorizon_guard.pattern_library.store import load_patterns
from longhorizon_guard.taxonomy.categories import DEFAULT_TAGS
from longhorizon_guard.subgoals.tracker import SubgoalTracker
from longhorizon_guard.subgoals.schema import SubgoalStatePayload
from longhorizon_guard.drift_monitor.monitor import DriftMonitor
from longhorizon_guard.drift_monitor.schema import DriftAssessment
from longhorizon_guard.reflector.reflector import PlanReflector
from longhorizon_guard.reflector.schema import ReflectionResult

logger = logging.getLogger("longhorizon_guard.guard")


# ---------------------------------------------------------------------------
# Per-category confidence thresholds — recalibrated strictly from trusted 
# 30-record baseline (findings/holdout_v2_judged.json, Llama 3.3 70B, 56.7% overall):
#
#   planning_error   : 83.3% accuracy (10/12) → threshold 0.20 (highest trust)
#   reflection_error : 62.5% accuracy ( 5/ 8) → threshold 0.25 (solid trust)
#   memory_error     : 28.6% accuracy ( 2/ 7) → threshold 0.40 (low trust)
#   external_error   :  0.0% accuracy ( 0/ 2) → threshold 0.40 (insufficient data / conservative default)
#   tool_use_error   :  0.0% accuracy ( 0/ 1) → threshold 0.40 (insufficient data / conservative default)
#   grader_error     :  N/A accuracy ( 0/ 0) → threshold 0.50 (no baseline samples)
#   other            :  N/A accuracy ( 0/ 0) → threshold 0.50 (no baseline samples)
#
# Lower judge accuracy → higher confidence required before we flag.
# ---------------------------------------------------------------------------
DEFAULT_CATEGORY_THRESHOLDS: Dict[str, float] = {
    "planning_error":   0.20,
    "reflection_error": 0.25,
    "memory_error":     0.40,
    "tool_use_error":   0.40,
    "external_error":   0.40,
    "grader_error":     0.50,
    "other":            0.50,
}

# Maximum time (seconds) for a single matching call before we bail.
DEFAULT_MATCH_TIMEOUT_SECONDS: float = 2.0

# TF-IDF stopwords (minimal standard English stopwords only)
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "of", "and", "in",
    "that", "with", "for", "on", "at", "by", "from", "it", "this", "these",
    "those", "not", "or", "but", "as", "if", "they", "their", "them", "agent",
    "step", "steps", "which", "than", "because", "so", "can",
    "could", "would", "may", "might", "very", "also",
}


# =========================================================================
# Keyword / Heuristic rule definitions (Layer B — fallback)
# =========================================================================

@dataclass
class KeywordRule:
    """A single keyword/heuristic rule for a specific error category."""
    rule_id: str
    category: str
    description: str
    # Any of these keyword sets must match (OR logic across sets).
    # Each set is AND logic: all keywords in the set must appear.
    keyword_sets: List[Set[str]] = field(default_factory=list)
    # Structural checks (callables) — evaluated in addition to keywords
    # Not serialised; set at construction time.
    min_keyword_hits: int = 1  # how many keyword sets must match


# Pre-defined rules per category
def _build_keyword_rules() -> List[KeywordRule]:
    """Build the static keyword/heuristic rule table."""
    return [
        # --- tool_use_error ---
        KeywordRule(
            rule_id="tool_use_malformed_action",
            category="tool_use_error",
            description="Malformed or invalid action structure (syntax errors, bad params)",
            keyword_sets=[
                {"nothing", "happens"},
                {"invalid", "action"},
                {"malformed"},
                {"syntax", "error"},
                {"unknown", "action"},
                {"unrecognized"},
                {"parse", "error"},
            ],
        ),
        KeywordRule(
            rule_id="tool_use_wrong_format",
            category="tool_use_error",
            description="Action name or arguments formatted incorrectly",
            keyword_sets=[
                {"wrong", "format"},
                {"incorrect", "format"},
                {"missing", "argument"},
                {"missing", "parameter"},
                {"quote", "error"},
            ],
        ),
        # --- external_error ---
        KeywordRule(
            rule_id="external_step_limit",
            category="external_error",
            description="Agent hit the environment step limit / timeout",
            keyword_sets=[
                {"step", "limit"},
                {"maximum", "steps"},
                {"timeout"},
                {"exceeded", "limit"},
                {"max", "turns"},
                {"truncated"},
            ],
        ),
        KeywordRule(
            rule_id="external_env_error",
            category="external_error",
            description="Environment or API returned an error outside agent control",
            keyword_sets=[
                {"connection", "error"},
                {"server", "error"},
                {"404"},
                {"500"},
                {"api", "error"},
                {"network", "error"},
                {"rate", "limit"},
            ],
        ),
        # --- memory_error ---
        KeywordRule(
            rule_id="memory_repeated_action",
            category="memory_error",
            description="Agent repeated an identical action it already tried",
            keyword_sets=[
                {"already", "tried"},
                {"already", "visited"},
                {"already", "explored"},
                {"repeated", "action"},
                {"revisit"},
                {"looping"},
                {"same", "action", "again"},
            ],
        ),
        KeywordRule(
            rule_id="memory_forgot_observation",
            category="memory_error",
            description="Agent forgot or contradicted a prior observation",
            keyword_sets=[
                {"forgot"},
                {"contradicts", "earlier"},
                {"previously", "observed"},
                {"already", "found"},
                {"oversimplif"},  # matches oversimplified, oversimplifying
                {"hallucinated"},
            ],
        ),
        # --- planning_error ---
        KeywordRule(
            rule_id="planning_constraint_violation",
            category="planning_error",
            description="Agent violated an explicit task constraint",
            keyword_sets=[
                {"constraint", "ignored"},
                {"constraint", "violated"},
                {"ignored", "requirement"},
                {"wrong", "constraint"},
                {"explicit", "requirement"},
            ],
        ),
        KeywordRule(
            rule_id="planning_exhaustive_search",
            category="planning_error",
            description="Agent used blind sequential enumeration instead of informed search",
            keyword_sets=[
                {"one", "by", "one"},
                {"sequential", "search"},
                {"exhaustive"},
                {"cabinet", "by", "cabinet"},
                {"drawer", "by", "drawer"},
                {"mechanical", "strategy"},
            ],
        ),
        # --- reflection_error ---
        KeywordRule(
            rule_id="reflection_wrong_object",
            category="reflection_error",
            description="Agent acted on the wrong object or misjudged object state",
            keyword_sets=[
                {"wrong", "object"},
                {"instead", "of"},
                {"not", "clean"},
                {"misjudged"},
                {"misidentified"},
                {"incomplete", "state"},
            ],
        ),
        KeywordRule(
            rule_id="reflection_ignored_feedback",
            category="reflection_error",
            description="Agent ignored environment feedback indicating failure",
            keyword_sets=[
                {"ignored", "feedback"},
                {"nothing", "happens"},
                {"failed", "recognize"},
                {"misinterpreted"},
                {"did", "not", "notice"},
            ],
        ),
    ]


# =========================================================================
# TF-IDF helper functions (lightweight, self-contained)
# =========================================================================

def _tokenize(text: str) -> List[str]:
    """Tokenize, lowercase, strip punctuation, filter stopwords."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute a single normalized TF-IDF vector given pre-computed IDF."""
    if not tokens:
        return {}
    tf_counts = Counter(tokens)
    doc_len = len(tokens)
    vec: Dict[str, float] = {}
    norm_sq = 0.0
    for w, count in tf_counts.items():
        if w in idf:
            val = (count / doc_len) * idf[w]
            vec[w] = val
            norm_sq += val * val
    norm = math.sqrt(norm_sq)
    if norm > 0:
        for w in vec:
            vec[w] /= norm
    return vec


def _cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two normalized sparse vectors."""
    if not v1 or not v2:
        return 0.0
    common = set(v1.keys()) & set(v2.keys())
    if not common:
        return 0.0
    return sum(v1[k] * v2[k] for k in common)


# =========================================================================
# MatchResult dataclass
# =========================================================================

@dataclass
class MatchResult:
    """Result of a pattern/rule match check."""
    matched: bool = False
    category: str = ""
    pattern_id: str = ""
    confidence: float = 0.0
    layer: str = ""          # "cluster" or "keyword"
    rule_id: str = ""        # populated for keyword matches
    description: str = ""
    safe_alternative: str = ""


# =========================================================================
# PatternMatcher — the core matching engine
# =========================================================================

class PatternMatcher:
    """Two-layer matching engine: cluster similarity + keyword rules.

    Thread-safe for concurrent on_step calls.
    """

    def __init__(
        self,
        pattern_library_path: str = "findings/pattern_library.json",
        category_thresholds: Optional[Dict[str, float]] = None,
        match_timeout: float = DEFAULT_MATCH_TIMEOUT_SECONDS,
    ) -> None:
        self._match_timeout = match_timeout
        self._category_thresholds = category_thresholds or dict(DEFAULT_CATEGORY_THRESHOLDS)
        self._keyword_rules = _build_keyword_rules()

        # Load and index pattern library
        self._patterns: List[PatternEntry] = []
        self._pattern_centroids: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._loaded = False

        try:
            self._load_patterns(pattern_library_path)
            self._loaded = True
            logger.info(
                "PatternMatcher initialised: %d patterns loaded from %s",
                len(self._patterns), pattern_library_path,
            )
        except Exception:
            logger.exception(
                "Failed to load pattern library from %s — cluster layer disabled, keyword-only mode",
                pattern_library_path,
            )

    def _load_patterns(self, path: str) -> None:
        """Load patterns and pre-computed broad-corpus IDF + centroid vectors."""
        self._patterns = load_patterns(path)
        if not self._patterns:
            return

        # Resolve path to absolute location
        path_obj = Path(path)
        candidates = [
            Path(__file__).resolve().parents[2] / "findings" / path_obj.name,
            Path(__file__).resolve().parents[2] / path_obj,
            Path.cwd().parent / path_obj,
            Path.cwd() / path_obj,
        ]
        for cand in candidates:
            if cand.exists():
                path_obj = cand
                break

        # Check if pre-computed broad-corpus IDF table exists in pattern library or sibling idf_table.json
        precomputed_idf: Dict[str, float] = {}
        if path_obj.exists():
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict) and "idf" in raw_data:
                        precomputed_idf = raw_data["idf"]
            except Exception as e:
                logger.warning("Failed reading idf from %s: %s", path_obj, e)

        if not precomputed_idf and path_obj.parent.exists():
            sibling_idf_path = path_obj.parent / "idf_table.json"
            if sibling_idf_path.exists():
                try:
                    with open(sibling_idf_path, "r", encoding="utf-8") as f:
                        precomputed_idf = json.load(f)
                except Exception as e:
                    logger.warning("Failed reading sibling idf_table.json: %s", e)

        # Collect all trigger descriptions tokens
        all_docs_tokens: List[List[str]] = [
            _tokenize(p.trigger_description) for p in self._patterns
        ]

        if precomputed_idf:
            self._idf = precomputed_idf
        else:
            # Fallback: compute IDF over pattern trigger descriptions if precomputed IDF is unavailable
            num_docs = len(all_docs_tokens)
            df: Dict[str, int] = defaultdict(int)
            for doc in all_docs_tokens:
                for w in set(doc):
                    df[w] += 1
            self._idf = {
                w: math.log((num_docs + 1.0) / (count + 1.0)) + 1.0
                for w, count in df.items()
            }

        # Compute centroid vectors for each pattern
        self._pattern_centroids = []
        for tokens in all_docs_tokens:
            vec = _tfidf_vector(tokens, self._idf)
            self._pattern_centroids.append(vec)

    def match(self, text: str, run_id: str = "", step_index: int = -1) -> MatchResult:
        """Run two-layer matching with timeout. Fails open on any error.

        Args:
            text: The combined reasoning + action text to check.
            run_id: For logging context.
            step_index: For logging context.

        Returns:
            MatchResult (matched=False if nothing flagged or on error).
        """
        result = MatchResult()

        def _do_match():
            nonlocal result
            result = self._match_impl(text)

        thread = threading.Thread(target=_do_match, daemon=True)
        thread.start()
        thread.join(timeout=self._match_timeout)

        if thread.is_alive():
            logger.warning(
                "match_timeout run_id=%s step=%d timeout=%.1fs — failing open",
                run_id, step_index, self._match_timeout,
            )
            return MatchResult()  # fail open

        if result.matched:
            logger.info(
                "match_found run_id=%s step=%d category=%s confidence=%.3f "
                "layer=%s pattern_id=%s rule_id=%s description=%s",
                run_id, step_index, result.category, result.confidence,
                result.layer, result.pattern_id, result.rule_id,
                result.description[:120],
            )

        return result

    def _match_impl(self, text: str) -> MatchResult:
        """Internal two-layer match logic (no timeout wrapper)."""
        # Layer A: Cluster similarity
        cluster_result = self._cluster_match(text)
        if cluster_result.matched:
            return cluster_result

        # Layer B: Keyword rules
        keyword_result = self._keyword_match(text)
        if keyword_result.matched:
            return keyword_result

        return MatchResult()

    def _cluster_match(self, text: str) -> MatchResult:
        """Layer A: TF-IDF cosine similarity against pattern centroids."""
        if not self._loaded or not self._patterns:
            return MatchResult()

        tokens = _tokenize(text)
        if not tokens:
            return MatchResult()

        query_vec = _tfidf_vector(tokens, self._idf)
        if not query_vec:
            return MatchResult()

        best_sim = -1.0
        best_idx = -1
        for idx, centroid in enumerate(self._pattern_centroids):
            sim = _cosine_sim(query_vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx < 0:
            return MatchResult()

        pattern = self._patterns[best_idx]
        threshold = self._category_thresholds.get(pattern.category, 0.40)

        if best_sim >= threshold:
            return MatchResult(
                matched=True,
                category=pattern.category,
                pattern_id=pattern.pattern_id,
                confidence=round(best_sim, 4),
                layer="cluster",
                description=pattern.trigger_description[:200],
                safe_alternative=pattern.safe_alternative[:200],
            )

        return MatchResult()

    def _keyword_match(self, text: str) -> MatchResult:
        """Layer B: Keyword/heuristic rule matching."""
        if not text:
            return MatchResult()

        text_lower = text.lower()
        text_tokens = set(_tokenize(text))

        best_rule: Optional[KeywordRule] = None
        best_hits = 0

        for rule in self._keyword_rules:
            hits = 0
            for kw_set in rule.keyword_sets:
                # Check if ALL keywords in this set appear in text
                if all(kw in text_lower for kw in kw_set):
                    hits += 1

            if hits >= rule.min_keyword_hits and hits > best_hits:
                best_hits = hits
                best_rule = rule

        if best_rule is None:
            return MatchResult()

        # Confidence from keyword density (more keyword sets matched → higher)
        max_possible = len(best_rule.keyword_sets)
        confidence = min(0.85, 0.40 + (best_hits / max(max_possible, 1)) * 0.45)

        return MatchResult(
            matched=True,
            category=best_rule.category,
            pattern_id="",
            confidence=round(confidence, 4),
            layer="keyword",
            rule_id=best_rule.rule_id,
            description=best_rule.description,
        )


# =========================================================================
# Structural heuristic checks (applied in on_step with history context)
# =========================================================================

def _detect_action_repetition(
    current_step: Dict[str, Any],
    history: List[Dict[str, Any]],
    lookback: int = 5,
) -> Optional[MatchResult]:
    """Detect if the current action is identical to one of the last N steps."""
    cur_action = current_step.get("action_name", "")
    cur_args = str(current_step.get("action_args", ""))
    if not cur_action:
        return None

    cur_key = f"{cur_action}|{cur_args}"
    recent = history[-lookback:] if len(history) > lookback else history

    repeat_count = 0
    for prev in recent:
        prev_key = f"{prev.get('action_name', '')}|{str(prev.get('action_args', ''))}"
        if prev_key == cur_key:
            repeat_count += 1

    if repeat_count >= 2:
        return MatchResult(
            matched=True,
            category="memory_error",
            confidence=min(0.90, 0.50 + repeat_count * 0.10),
            layer="keyword",
            rule_id="structural_action_repetition",
            description=f"Action '{cur_action}' repeated {repeat_count} times in last {lookback} steps",
        )
    return None


def _detect_nothing_happens_loop(
    current_step: Dict[str, Any],
    history: List[Dict[str, Any]],
    lookback: int = 3,
) -> Optional[MatchResult]:
    """Detect repeated 'Nothing happens' tool responses."""
    cur_response = str(current_step.get("tool_response", "")).lower()
    if "nothing happens" not in cur_response:
        return None

    recent = history[-lookback:] if len(history) > lookback else history
    nothing_count = sum(
        1 for s in recent
        if "nothing happens" in str(s.get("tool_response", "")).lower()
    )

    if nothing_count >= 2:
        return MatchResult(
            matched=True,
            category="tool_use_error",
            confidence=min(0.85, 0.50 + nothing_count * 0.10),
            layer="keyword",
            rule_id="structural_nothing_happens_loop",
            description=f"'Nothing happens' response {nothing_count + 1} times in last {lookback + 1} steps",
        )
    return None


# =========================================================================
# Production GuardInterface
# =========================================================================

class GuardInterface:
    """Production guard interface with two-layer failure pattern matching.

    Wires into the existing 4-method hook API. Fails open on all errors.
    """

    def __init__(
        self,
        pattern_library_path: str = "findings/pattern_library.json",
        category_thresholds: Optional[Dict[str, float]] = None,
        match_timeout: float = DEFAULT_MATCH_TIMEOUT_SECONDS,
    ) -> None:
        try:
            self._matcher = PatternMatcher(
                pattern_library_path=pattern_library_path,
                category_thresholds=category_thresholds,
                match_timeout=match_timeout,
            )
        except Exception:
            logger.exception("GuardInterface init failed — pattern matcher disabled")
            self._matcher = None  # type: ignore[assignment]

        try:
            self._subgoal_tracker = SubgoalTracker()
        except Exception:
            logger.exception("SubgoalTracker init failed — subgoal tracking disabled")
            self._subgoal_tracker = None  # type: ignore[assignment]

        try:
            self._drift_monitor = DriftMonitor()
        except Exception:
            logger.exception("DriftMonitor init failed — drift monitoring disabled")
            self._drift_monitor = None  # type: ignore[assignment]

        try:
            self._reflector = PlanReflector(step_interval=5)
        except Exception:
            logger.exception("PlanReflector init failed — periodic plan reflection disabled")
            self._reflector = None  # type: ignore[assignment]

    # ---- Hook 1: on_plan_proposed ----

    def on_plan_proposed(
        self,
        task_description: str,
        proposed_plan: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check proposed plan against planning_error patterns before execution
        and parse declared subgoals into tracking records.

        Returns:
            {
                "approved": bool,
                "flags": list[str],
                "suggestions": list[str],
                "subgoals": list[dict]
            }
        """
        result: Dict[str, Any] = {"approved": True, "flags": [], "suggestions": [], "subgoals": []}
        try:
            if self._subgoal_tracker is not None:
                subgoal_init = self._subgoal_tracker.init_plan(task_description, proposed_plan, metadata)
                result["subgoals"] = self._subgoal_tracker.get_state_payload().to_dict()

            if self._reflector is not None:
                self._reflector.init_plan(task_description, proposed_plan)

            if self._drift_monitor is not None:
                self._drift_monitor.reset()

            if self._matcher is None:
                return result

            run_id = (metadata or {}).get("run_id", "unknown")
            combined_text = f"{task_description} {proposed_plan}"

            match = self._matcher.match(combined_text, run_id=run_id, step_index=-1)

            if match.matched and match.category == "planning_error":
                result["flags"].append(
                    f"[{match.layer}] planning_error detected (confidence={match.confidence:.2f}): "
                    f"{match.description}"
                )
                if match.safe_alternative:
                    result["suggestions"].append(match.safe_alternative)

                logger.info(
                    "on_plan_proposed flag run_id=%s category=%s confidence=%.3f layer=%s",
                    run_id, match.category, match.confidence, match.layer,
                )

        except Exception:
            logger.exception("on_plan_proposed error — failing open, run_id=%s",
                             (metadata or {}).get("run_id", "?"))

        return result

    # ---- Hook 2: on_step ----

    def on_step(
        self,
        step_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Primary hook: check each step for pattern/rule matches.

        Returns:
            {
                "continue_execution": bool,
                "drift_detected": bool,
                "warning": str | None,
                "match_details": dict | None  # structured match info
            }
        """
        result: Dict[str, Any] = {
            "continue_execution": True,
            "drift_detected": False,
            "warning": None,
            "match_details": None,
        }
        try:
            # Process step in SubgoalTracker first
            if self._subgoal_tracker is not None:
                sub_res = self._subgoal_tracker.process_step(step_record, history, metadata)
                result["subgoal_state"] = sub_res.get("state_payload")
                if sub_res.get("transition_event"):
                    result["subgoal_transition"] = sub_res["transition_event"]

            if self._matcher is None:
                return result

            run_id = (metadata or {}).get("run_id", "unknown")
            step_idx = step_record.get("step_index", -1)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")

            # Build text for pattern matching (Layer A & B): reasoning + action ONLY
            # Excludes tool_response to prevent environment prompt boilerplate false matches
            reasoning = step_record.get("reasoning", "") or ""
            action_name = step_record.get("action_name", "") or ""
            action_args = str(step_record.get("action_args", "")) or ""

            combined_text = f"{reasoning} {action_name} {action_args}"

            # --- Layer A + B via PatternMatcher ---
            match = self._matcher.match(combined_text, run_id=run_id, step_index=step_idx)

            # --- Structural heuristics (always run) ---
            if not match.matched:
                structural = _detect_action_repetition(step_record, history)
                if structural is None:
                    structural = _detect_nothing_happens_loop(step_record, history)
                if structural and structural.matched:
                    match = structural
                    logger.info(
                        "structural_match run_id=%s step=%d category=%s confidence=%.3f "
                        "rule_id=%s description=%s timestamp=%s",
                        run_id, step_idx, match.category, match.confidence,
                        match.rule_id, match.description, ts,
                    )

            if match.matched:
                result["drift_detected"] = True
                result["warning"] = (
                    f"[{match.layer}:{match.rule_id or match.pattern_id}] "
                    f"{match.category} (confidence={match.confidence:.2f}): {match.description}"
                )
                result["match_details"] = {
                    "timestamp": ts,
                    "run_id": run_id,
                    "step_index": step_idx,
                    "category": match.category,
                    "confidence": match.confidence,
                    "layer": match.layer,
                    "pattern_id": match.pattern_id,
                    "rule_id": match.rule_id,
                    "description": match.description,
                    "safe_alternative": match.safe_alternative,
                }

            # Evaluate incremental drift in DriftMonitor
            if self._drift_monitor is not None:
                drift_assessment = self._drift_monitor.evaluate_step(
                    subgoal_state=result.get("subgoal_state"),
                    step_record=step_record,
                    history=history,
                    match_details=result.get("match_details"),
                )
                result["drift_assessment"] = drift_assessment.to_dict()
                if drift_assessment.drift_detected:
                    result["drift_detected"] = True
                    drift_msg = (
                        f"[drift_monitor:{','.join(drift_assessment.triggered_signals)}] "
                        f"severity={drift_assessment.severity_level}: {'; '.join(drift_assessment.reasons)}"
                    )
                    result["warning"] = f"{result['warning']} | {drift_msg}" if result.get("warning") else drift_msg

            # Evaluate periodic plan reflection in PlanReflector
            if self._reflector is not None:
                is_subgoal_boundary = result.get("subgoal_transition") is not None
                refl_res = self._reflector.evaluate(
                    step_record=step_record,
                    history=history,
                    subgoal_state=result.get("subgoal_state"),
                    drift_assessment=result.get("drift_assessment"),
                    match_details=result.get("match_details"),
                    trigger_type="subgoal_boundary" if is_subgoal_boundary else "step_interval",
                    force=is_subgoal_boundary,
                )
                result["reflection_result"] = refl_res.to_dict()
                if refl_res.revision_suggested:
                    refl_msg = f"[reflector:{refl_res.trigger_type}] Plan revision suggested: {refl_res.revision_reasoning}"
                    result["warning"] = f"{result['warning']} | {refl_msg}" if result.get("warning") else refl_msg

        except Exception:
            logger.exception(
                "on_step error — failing open, run_id=%s step=%d",
                (metadata or {}).get("run_id", "?"),
                step_record.get("step_index", -1),
            )

        return result

    # ---- Hook 3: on_subgoal_boundary ----

    def on_subgoal_boundary(
        self,
        subgoal_id: str,
        subgoal_status: str,
        step_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Hook called when a subgoal checkpoint boundary is reached (Layer 4 - Subgoal Tracker).

        Returns:
            {
                "checkpoint_passed": bool,
                "next_subgoal": str | None,
                "subgoal_state": dict | None,
                "drift_assessment": dict | None,
                "reflection_result": dict | None
            }
        """
        result: Dict[str, Any] = {
            "checkpoint_passed": (subgoal_status in ("completed", "stalled_advanced")),
            "next_subgoal": None,
            "subgoal_state": None,
            "drift_assessment": None,
            "reflection_result": None,
        }
        try:
            if self._subgoal_tracker is not None:
                state = self._subgoal_tracker.get_state_payload()
                result["subgoal_state"] = state.to_dict()
                result["next_subgoal"] = state.current_subgoal_id

            if self._drift_monitor is not None:
                boundary_assessment = self._drift_monitor.evaluate_boundary(
                    subgoal_id=subgoal_id,
                    subgoal_status=subgoal_status,
                    step_history=step_history,
                    subgoal_state=result.get("subgoal_state"),
                )
                result["drift_assessment"] = boundary_assessment.to_dict()

            if self._reflector is not None:
                dummy_step = {"step_index": len(step_history)}
                refl_res = self._reflector.evaluate(
                    step_record=dummy_step,
                    history=step_history,
                    subgoal_state=result.get("subgoal_state"),
                    drift_assessment=result.get("drift_assessment"),
                    trigger_type="subgoal_boundary",
                    force=False,
                )
                result["reflection_result"] = refl_res.to_dict()

        except Exception:
            logger.exception("on_subgoal_boundary error — failing open")
        return result

    # ---- Hook 4: on_run_end ----

    def on_run_end(
        self,
        run_metadata: Dict[str, Any],
        trajectory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Post-run summary: scan full trajectory for flags and finalize subgoal report.

        Returns:
            {
                "processed": bool,
                "root_cause_step_index": int | None,
                "root_cause_error_type": str | None,
                "flags_summary": list[dict],
                "subgoals_summary": dict | None,
                "drift_summary": dict | None,
                "reflection_summary": dict | None
            }
        """
        result: Dict[str, Any] = {
            "processed": True,
            "root_cause_step_index": None,
            "root_cause_error_type": None,
            "flags_summary": [],
            "subgoals_summary": None,
            "drift_summary": None,
            "reflection_summary": None,
        }
        try:
            if self._subgoal_tracker is not None:
                result["subgoals_summary"] = self._subgoal_tracker.finalize_run(run_metadata, trajectory)

            if self._drift_monitor is not None:
                result["drift_summary"] = self._drift_monitor.finalize_run(
                    run_metadata, trajectory, result.get("subgoals_summary")
                )

            steps = trajectory.get("steps", [])
            if self._reflector is not None and steps:
                last_refl = self._reflector.evaluate(
                    step_record=steps[-1],
                    history=steps[:-1],
                    subgoal_state=result.get("subgoals_summary"),
                    drift_assessment=result.get("drift_summary"),
                    trigger_type="run_end",
                    force=True,
                )
                result["reflection_summary"] = last_refl.to_dict()

            if self._matcher is None:
                return result

            run_id = run_metadata.get("run_id", "unknown")
            steps = trajectory.get("steps", [])

            # Scan all steps in order and collect flags
            # Root-cause rule (per MASTER doc): earliest tagged step, not highest-confidence
            best_match: Optional[MatchResult] = None
            best_step_idx: Optional[int] = None
            all_flags: List[Dict[str, Any]] = []
            history: List[Dict[str, Any]] = []

            for step in steps:
                step_result = self.on_step(step, history, metadata=run_metadata)
                if step_result.get("match_details"):
                    details = step_result["match_details"]
                    all_flags.append(details)
                    # Earliest-step selection: take the FIRST step that produced a match
                    if best_match is None:
                        best_step_idx = details["step_index"]
                        best_match = MatchResult(
                            matched=True,
                            category=details["category"],
                            confidence=details["confidence"],
                            layer=details["layer"],
                            pattern_id=details.get("pattern_id", ""),
                            rule_id=details.get("rule_id", ""),
                            description=details.get("description", ""),
                        )
                history.append(step)

            if best_match and best_match.matched:
                result["root_cause_step_index"] = best_step_idx
                result["root_cause_error_type"] = best_match.category

            result["flags_summary"] = all_flags

            logger.info(
                "on_run_end run_id=%s total_steps=%d flags=%d "
                "root_cause=%s root_step=%s",
                run_id, len(steps), len(all_flags),
                result["root_cause_error_type"],
                result["root_cause_step_index"],
            )

        except Exception:
            logger.exception(
                "on_run_end error — failing open, run_id=%s",
                run_metadata.get("run_id", "?"),
            )

        return result
