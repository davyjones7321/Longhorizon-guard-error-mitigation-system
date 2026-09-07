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

from longhorizon_guard.config import GuardConfig

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


def _derive_error_type_from_drift(drift_info: Dict[str, Any]) -> str:
    """Map drift monitor diagnostic signals to a taxonomy error category.

    Signal Mappings:
      - REPEATED_STALLED_SUBGOALS: Agent forced to advance repeatedly without completing
        subgoals, indicating an unachievable plan or flawed task decomposition -> planning_error.
      - ACCUMULATED_SUBGOAL_FAILURES: Multiple subgoals explicitly failed -> planning_error.
      - SLOW_PROGRESS_RATIO: Inefficient wandering or stuck in single subgoal without progress -> planning_error.
      - PATTERN_REPETITION_DRIFT: Repeated action repetition or failure patterns -> reflection_error.
      - Default fallback: planning_error (drift monitor tracks plan divergence).
    """
    signals = set(drift_info.get("triggered_signals") or [])
    if "PATTERN_REPETITION_DRIFT" in signals:
        return "reflection_error"
    if "REPEATED_STALLED_SUBGOALS" in signals or "ACCUMULATED_SUBGOAL_FAILURES" in signals or "SLOW_PROGRESS_RATIO" in signals:
        return "planning_error"
    reasons = " ".join(drift_info.get("reasons") or []).lower()
    if "tool" in reasons:
        return "tool_use_error"
    if "memory" in reasons:
        return "memory_error"
    if "reflection" in reasons:
        return "reflection_error"
    return "planning_error"


def _derive_error_type_from_reflection(refl_info: Dict[str, Any]) -> str:
    """Map reflector revision diagnostics to a taxonomy error category.

    Inspects evidence sources and revision reasoning text:
      - If reasoning cites planning or subgoal failures -> planning_error.
      - If reasoning cites reflection or repeated mistakes -> reflection_error.
      - If reasoning cites memory or forgot -> memory_error.
      - If reasoning cites tool failure -> tool_use_error.
      - Default: "plan_deviation" (fallback when no specific category applies).
    """
    evidence = [str(s).lower() for s in (refl_info.get("evidence_sources") or [])]
    reasoning = (refl_info.get("revision_reasoning") or "").lower()
    combined = f"{' '.join(evidence)} {reasoning}"

    if "planning" in combined or "subgoal" in combined or "goal" in combined:
        return "planning_error"
    if "reflection" in combined or "feedback" in combined:
        return "reflection_error"
    if "memory" in combined or "forgot" in combined:
        return "memory_error"
    if "tool" in combined:
        return "tool_use_error"
    return "plan_deviation"

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
                {"malformed", "action"},
                {"malformed", "call"},
                {"syntax", "error"},
                {"unknown", "action"},
                {"unrecognized", "action"},
                {"unrecognized", "command"},
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
                {"step", "timeout"},
                {"timed", "out"},
                {"timeout", "limit"},
                {"execution", "timeout"},
                {"exceeded", "limit"},
                {"max", "turns"},
                {"trajectory", "truncated"},
                {"steps", "truncated"},
                {"truncated", "limit"},
            ],
        ),
        KeywordRule(
            rule_id="external_env_error",
            category="external_error",
            description="Environment or API returned an error outside agent control",
            keyword_sets=[
                {"connection", "error"},
                {"server", "error"},
                {"404", "error"},
                {"http", "404"},
                {"api", "404"},
                {"500", "error"},
                {"http", "500"},
                {"api", "500"},
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
                {"revisit", "action"},
                {"revisit", "same"},
                {"infinite", "loop"},
                {"looping", "action"},
                {"stuck", "loop"},
                {"same", "action", "again"},
            ],
        ),
        KeywordRule(
            rule_id="memory_forgot_observation",
            category="memory_error",
            description="Agent forgot or contradicted a prior observation",
            keyword_sets=[
                {"forgot", "observation"},
                {"forgot", "previously"},
                {"forgot", "earlier"},
                {"contradicts", "earlier"},
                {"previously", "observed"},
                {"already", "found"},
                {"oversimplif", "memory"},
                {"oversimplif", "experience"},
                {"oversimplif", "recall"},
                {"hallucinated", "observation"},
                {"hallucinated", "action"},
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
                {"one by one"},
                {"sequential", "search"},
                {"exhaustive", "search"},
                {"exhaustive", "enumeration"},
                {"cabinet by cabinet"},
                {"drawer by drawer"},
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
                {"wrong", "instead", "of"},
                {"picked", "instead", "of"},
                {"not", "clean"},
                {"misjudged", "object"},
                {"misjudged", "state"},
                {"misidentified", "object"},
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
                {"misinterpreted", "feedback"},
                {"misinterpreted", "response"},
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
        pattern_library_path: Optional[str] = "findings/pattern_library.json",
        category_thresholds: Optional[Dict[str, float]] = None,
        match_timeout: float = DEFAULT_MATCH_TIMEOUT_SECONDS,
    ) -> None:
        pattern_library_path = pattern_library_path or "findings/pattern_library.json"
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
        bundled_data_dir = Path(__file__).resolve().parents[0] / "data"
        candidates = [
            path_obj,
            bundled_data_dir / path_obj.name,
            bundled_data_dir / "pattern_library.json",
            Path(__file__).resolve().parents[1] / "findings" / path_obj.name,
            Path(__file__).resolve().parents[1] / path_obj,
            Path.cwd() / "findings" / path_obj.name,
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

        if not precomputed_idf:
            idf_candidates = [
                path_obj.parent / "idf_table.json",
                bundled_data_dir / "idf_table.json",
                Path(__file__).resolve().parents[1] / "findings" / "idf_table.json",
            ]
            for sibling_idf_path in idf_candidates:
                if sibling_idf_path.exists():
                    try:
                        with open(sibling_idf_path, "r", encoding="utf-8") as f:
                            precomputed_idf = json.load(f)
                            break
                    except Exception as e:
                        logger.warning("Failed reading sibling idf_table.json from %s: %s", sibling_idf_path, e)

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
# Plan constraint and contradiction heuristics (Rule 4 / Approaches 1 & 2)
# =========================================================================

CONTRADICTORY_TERM_PAIRS = (
    ("men", (r"\bmen(?:'s)?\b", r"\bmens\b"), "women", (r"\bwomen(?:'s)?\b", r"\bwomens\b")),
    ("male", (r"\bmale\b",), "female", (r"\bfemale\b",)),
    ("boys", (r"\bboys?\b",), "girls", (r"\bgirls?\b",)),
    ("cheap", (r"\bcheapest\b", r"\bcheap\b"), "expensive", (r"\bexpensive\b", r"\bcostly\b")),
    ("non-stop", (r"\bnon-?stop\b", r"\bdirect\b"), "layover", (r"\blayover\b", r"\bconnecting\b")),
)

EXPLICIT_REQUIRED_TERMS = (
    ("men", (r"\bmen(?:'s)?\b", r"\bmens\b")),
    ("women", (r"\bwomen(?:'s)?\b", r"\bwomens\b")),
    ("male", (r"\bmale\b",)),
    ("female", (r"\bfemale\b",)),
    ("boys", (r"\bboys?\b",)),
    ("girls", (r"\bgirls?\b",)),
)


def _detect_plan_contradiction(
    task_description: str,
    proposed_plan: str,
) -> Optional[MatchResult]:
    """Detect if proposed plan directly contradicts explicit task requirements (Approach 2)."""
    if not task_description or not proposed_plan:
        return None

    task_lower = task_description.lower()
    plan_lower = proposed_plan.lower()

    for left_name, left_patterns, right_name, right_patterns in CONTRADICTORY_TERM_PAIRS:
        task_has_left = any(re.search(p, task_lower) for p in left_patterns)
        task_has_right = any(re.search(p, task_lower) for p in right_patterns)
        plan_has_left = any(re.search(p, plan_lower) for p in left_patterns)
        plan_has_right = any(re.search(p, plan_lower) for p in right_patterns)

        if task_has_left and not task_has_right and plan_has_right and not plan_has_left:
            return MatchResult(
                matched=True,
                category="planning_error",
                confidence=0.92,
                layer="keyword",
                rule_id="planning_contradiction",
                description=f"Task explicitly requires '{left_name}', but proposed plan targets '{right_name}'",
                safe_alternative=f"Revise plan to target '{left_name}' as specified in task requirements",
            )

        if task_has_right and not task_has_left and plan_has_left and not plan_has_right:
            return MatchResult(
                matched=True,
                category="planning_error",
                confidence=0.92,
                layer="keyword",
                rule_id="planning_contradiction",
                description=f"Task explicitly requires '{right_name}', but proposed plan targets '{left_name}'",
                safe_alternative=f"Revise plan to target '{right_name}' as specified in task requirements",
            )

    return None


def _detect_plan_constraint_omission(
    task_description: str,
    proposed_plan: str,
) -> Optional[MatchResult]:
    """Detect if proposed plan omits explicit task constraints in its search/action strategy (Approach 1)."""
    if not task_description or not proposed_plan:
        return None

    task_lower = task_description.lower()
    plan_lower = proposed_plan.lower()

    has_action_intent = bool(
        re.search(r"\b(?:search|query|find|buy|purchase|filter|select|order|navigate)\b", plan_lower)
    )
    if not has_action_intent:
        return None

    for term_name, term_patterns in EXPLICIT_REQUIRED_TERMS:
        task_has_term = any(re.search(p, task_lower) for p in term_patterns)
        if task_has_term:
            plan_has_term = any(re.search(p, plan_lower) for p in term_patterns)
            if not plan_has_term:
                return MatchResult(
                    matched=True,
                    category="planning_error",
                    confidence=0.85,
                    layer="keyword",
                    rule_id="planning_constraint_omission",
                    description=f"Task explicitly requires '{term_name}', but proposed plan omits it from planned search/actions",
                    safe_alternative=f"Include '{term_name}' in search parameters and filtering steps",
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
        pattern_library_path: Optional[str] = None,
        category_thresholds: Optional[Dict[str, float]] = None,
        match_timeout: float = DEFAULT_MATCH_TIMEOUT_SECONDS,
        max_subgoal_steps: Optional[int] = None,
        drift_threshold: Optional[float] = None,
        config: Optional[GuardConfig] = None,
    ) -> None:
        if config is None:
            config = GuardConfig.from_env()

        self.config = config
        self.fail_open: bool = config.fail_open

        effective_pattern_path = pattern_library_path or config.pattern_library_path or "findings/pattern_library.json"
        effective_max_subgoal_steps = max_subgoal_steps if max_subgoal_steps is not None else config.max_subgoal_steps
        effective_drift_threshold = drift_threshold if drift_threshold is not None else config.drift_threshold
        effective_refl_interval = config.reflection_step_interval

        try:
            self._matcher = PatternMatcher(
                pattern_library_path=effective_pattern_path,
                category_thresholds=category_thresholds,
                match_timeout=match_timeout,
            )
        except Exception:
            logger.exception("GuardInterface init failed — pattern matcher disabled")
            self._matcher = None  # type: ignore[assignment]

        try:
            self._subgoal_tracker = SubgoalTracker(max_subgoal_steps=effective_max_subgoal_steps)
        except Exception:
            logger.exception("SubgoalTracker init failed — subgoal tracking disabled")
            self._subgoal_tracker = None  # type: ignore[assignment]

        try:
            self._drift_monitor = DriftMonitor(drift_threshold=effective_drift_threshold)
        except Exception:
            logger.exception("DriftMonitor init failed — drift monitoring disabled")
            self._drift_monitor = None  # type: ignore[assignment]

        try:
            self._reflector = PlanReflector(step_interval=effective_refl_interval)
        except Exception:
            logger.exception("PlanReflector init failed — periodic plan reflection disabled")
            self._reflector = None  # type: ignore[assignment]

        self._live_steps_recorded: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        self._in_replay: bool = False

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
                "flagged": bool,
                "flags": list[str],
                "suggestions": list[str],
                "subgoals": list[dict]
            }
        """
        self._live_steps_recorded = []
        result: Dict[str, Any] = {
            "approved": True,
            "flagged": False,
            "flags": [],
            "suggestions": [],
            "subgoals": [],
        }
        try:
            if self._subgoal_tracker is not None:
                subgoal_init = self._subgoal_tracker.init_plan(task_description, proposed_plan, metadata)
                result["subgoals"] = self._subgoal_tracker.get_state_payload().to_dict()

            if self._reflector is not None:
                self._reflector.init_plan(task_description, proposed_plan)

            if self._drift_monitor is not None:
                self._drift_monitor.reset()

            run_id = (metadata or {}).get("run_id", "unknown")

            # 1. Structural plan contradiction (Approach 2)
            contradiction_match = _detect_plan_contradiction(task_description, proposed_plan)
            if contradiction_match and contradiction_match.matched:
                result["flags"].append(
                    f"[{contradiction_match.layer}:{contradiction_match.rule_id}] "
                    f"planning_error detected (confidence={contradiction_match.confidence:.2f}): "
                    f"{contradiction_match.description}"
                )
                if contradiction_match.safe_alternative:
                    result["suggestions"].append(contradiction_match.safe_alternative)

            # 2. Structural constraint omission (Approach 1)
            omission_match = _detect_plan_constraint_omission(task_description, proposed_plan)
            if omission_match and omission_match.matched:
                result["flags"].append(
                    f"[{omission_match.layer}:{omission_match.rule_id}] "
                    f"planning_error detected (confidence={omission_match.confidence:.2f}): "
                    f"{omission_match.description}"
                )
                if omission_match.safe_alternative:
                    result["suggestions"].append(omission_match.safe_alternative)

            # 3. Layer A + B via PatternMatcher
            if self._matcher is not None:
                combined_text = f"{task_description} {proposed_plan}"
                match = self._matcher.match(combined_text, run_id=run_id, step_index=-1)

                if match.matched and match.category == "planning_error":
                    result["flags"].append(
                        f"[{match.layer}] planning_error detected (confidence={match.confidence:.2f}): "
                        f"{match.description}"
                    )
                    if match.safe_alternative:
                        result["suggestions"].append(match.safe_alternative)

            if result["flags"]:
                logger.info(
                    "on_plan_proposed flags run_id=%s count=%d",
                    run_id, len(result["flags"]),
                )

        except Exception:
            logger.exception("on_plan_proposed error — failing open, run_id=%s",
                             (metadata or {}).get("run_id", "?"))

        result["flagged"] = bool(result.get("flags"))
        return result

    # ---- Hook 2: on_step ----

    def on_step(
        self,
        step_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Primary hook: check each step for pattern/rule matches, drift, and plan reflection.

        Returns:
            {
                "continue_execution": bool,
                "flagged": bool,              # True if any sub-system flagged this step
                "drift_detected": bool,
                "warning": str | None,
                "match_details": dict | None, # structured match info
                "subgoal_state": dict | None,
                "drift_assessment": dict | None,
                "reflection_result": dict | None
            }
        """
        result: Dict[str, Any] = {
            "continue_execution": True,
            "flagged": False,
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

            run_id = (metadata or {}).get("run_id", "unknown")
            step_idx = step_record.get("step_index", -1)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")

            if self._matcher is not None:
                # Build text for pattern matching (Layer A & B): reasoning + action ONLY
                # Excludes tool_response to prevent environment prompt boilerplate false matches
                reasoning = step_record.get("reasoning", "") or ""
                action_name = step_record.get("action_name", "") or ""
                action_args = str(step_record.get("action_args", "")) or ""

                combined_text = f"{reasoning} {action_name} {action_args}"

                match: Optional[MatchResult] = None

                # Check explicit contradiction first if metadata provides task description
                if metadata:
                    task_desc = metadata.get("task_description") or metadata.get("description") or ""
                    if task_desc and "search" in action_name.lower():
                        contra = _detect_plan_contradiction(task_desc, f"{action_name} {action_args}")
                        if contra and contra.matched:
                            match = contra

                # --- Layer A + B via PatternMatcher if not already matched ---
                if match is None or not match.matched:
                    match = self._matcher.match(combined_text, run_id=run_id, step_index=step_idx)

                # --- Structural heuristics (always run) ---
                if not match.matched:
                    structural = _detect_action_repetition(step_record, history)
                    if structural is None:
                        structural = _detect_nothing_happens_loop(step_record, history)
                    if structural is None and metadata:
                        task_desc = metadata.get("task_description") or metadata.get("description") or ""
                        if task_desc and "search" in action_name.lower():
                            structural = _detect_plan_constraint_omission(task_desc, f"{action_name} {action_args}")
                    if structural and structural.matched:
                        match = structural
                        logger.info(
                            "structural_match run_id=%s step=%d category=%s confidence=%.3f "
                            "rule_id=%s description=%s timestamp=%s",
                            run_id, step_idx, match.category, match.confidence,
                            match.rule_id, match.description, ts,
                        )

                # Check tool_response for external environment/API errors if not already matched
                if (match is None or not match.matched) and step_record.get("tool_response"):
                    resp_str = str(step_record["tool_response"])
                    resp_match = self._matcher.match(resp_str, run_id=run_id, step_index=step_idx)
                    if resp_match.matched and resp_match.category == "external_error":
                        match = resp_match

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
        finally:
            # FIX C: flagged convenience field across all 3 sub-systems
            is_matched = result.get("match_details") is not None
            is_drift = bool(result.get("drift_detected"))
            refl = result.get("reflection_result")
            is_refl_revision = bool(refl and refl.get("revision_suggested"))
            result["flagged"] = is_matched or is_drift or is_refl_revision
            if not getattr(self, "_in_replay", False):
                self._live_steps_recorded.append((dict(step_record), dict(result)))

        return result

    # ---- Hook 3: on_subgoal_boundary ----

    def on_subgoal_boundary(
        self,
        subgoal_id: str,
        subgoal_status: str = "completed",
        step_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        step_history = step_history or []
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
        metadata: Optional[Dict[str, Any]] = None,
        trajectory: Optional[Dict[str, Any]] = None,
        *,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post-run summary: scan full trajectory for flags and finalize subgoal report.

        Root-Cause Fallback Chain:
          Tier 1 ("pattern_match"): Earliest step producing pattern-matcher or structural rule match_details.
          Tier 2 ("drift_monitor"): Earliest step triggering drift_detected=True, mapped from triggered_signals.
          Tier 3 ("reflector"): Earliest step where revision_suggested=True, mapped from reasoning/evidence.
          Tier 4 ("none"): Clean run with no failure signals detected across all 3 layers.

        Returns:
            {
                "processed": bool,
                "root_cause_step_index": int | None,
                "root_cause_error_type": str | None,
                "root_cause_source": str,  # "pattern_match" | "drift_monitor" | "reflector" | "none"
                "flags_summary": list[dict],
                "subgoals_summary": dict | None,
                "drift_summary": dict | None,
                "reflection_summary": dict | None
            }
        """
        if metadata is None and run_metadata is not None:
            metadata = run_metadata
        if metadata is None:
            metadata = {}
        if trajectory is None:
            trajectory = {}

        result: Dict[str, Any] = {
            "processed": True,
            "root_cause_step_index": None,
            "root_cause_error_type": None,
            "root_cause_source": "none",
            "flags_summary": [],
            "subgoals_summary": None,
            "drift_summary": None,
            "reflection_summary": None,
        }
        try:
            steps = trajectory.get("steps", [])
            live_steps = list(self._live_steps_recorded)

            # Branch: Standalone offline mode vs. live monitoring mode
            if not live_steps and steps:
                # Standalone offline/post-hoc evaluation:
                # Replay on_step() first in chronological order to build real tracker & drift state
                self._in_replay = True
                replay_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                history: List[Dict[str, Any]] = []
                try:
                    for step in steps:
                        step_res = self.on_step(step, history, metadata=metadata)
                        replay_results.append((step, step_res))
                        history.append(step)
                finally:
                    self._in_replay = False
                processed_steps = replay_results
            else:
                # Live mode: use step results already captured during live execution
                processed_steps = live_steps

            # Finalize tracker & drift monitor using the current, real state
            if self._subgoal_tracker is not None:
                result["subgoals_summary"] = self._subgoal_tracker.finalize_run(metadata, trajectory)

            if self._drift_monitor is not None:
                result["drift_summary"] = self._drift_monitor.finalize_run(
                    metadata, trajectory, result.get("subgoals_summary")
                )

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

            run_id = metadata.get("run_id", "unknown")

            # Track earliest candidate across each layer from processed_steps:
            earliest_pattern_match: Optional[Dict[str, Any]] = None
            earliest_pattern_step_idx: Optional[int] = None

            earliest_drift_assessment: Optional[Dict[str, Any]] = None
            earliest_drift_step_idx: Optional[int] = None

            earliest_refl_result: Optional[Dict[str, Any]] = None
            earliest_refl_step_idx: Optional[int] = None

            all_flags: List[Dict[str, Any]] = []

            for idx, (step, step_result) in enumerate(processed_steps):
                s_idx = step.get("step_index", idx)

                # 1. Pattern Matcher / Structural Detection
                if step_result.get("match_details"):
                    details = step_result["match_details"]
                    all_flags.append(details)
                    if earliest_pattern_match is None:
                        earliest_pattern_step_idx = details.get("step_index", s_idx)
                        earliest_pattern_match = details

                # 2. Drift Monitor Signal
                drift_info = step_result.get("drift_assessment") or {}
                if step_result.get("drift_detected") or drift_info.get("drift_detected"):
                    if earliest_drift_assessment is None:
                        earliest_drift_step_idx = drift_info.get("step_index", s_idx)
                        earliest_drift_assessment = drift_info

                # 3. Reflector Signal
                refl_info = step_result.get("reflection_result") or {}
                if refl_info.get("revision_suggested"):
                    if earliest_refl_result is None:
                        earliest_refl_step_idx = refl_info.get("step_index", s_idx)
                        earliest_refl_result = refl_info

            result["flags_summary"] = all_flags

            # Fallback Chain Application:
            # Tier 1: Pattern Matcher / Structural Rules
            if earliest_pattern_match is not None:
                result["root_cause_step_index"] = earliest_pattern_step_idx
                result["root_cause_error_type"] = earliest_pattern_match["category"]
                result["root_cause_source"] = "pattern_match"

            # Tier 2: Drift Monitor Signals
            elif earliest_drift_assessment is not None:
                result["root_cause_step_index"] = earliest_drift_step_idx
                result["root_cause_error_type"] = _derive_error_type_from_drift(earliest_drift_assessment)
                result["root_cause_source"] = "drift_monitor"

            # Tier 3: Plan Reflector Signals
            elif earliest_refl_result is not None or (
                result.get("reflection_summary") and result["reflection_summary"].get("revision_suggested")
            ):
                if earliest_refl_result is not None:
                    target_refl = earliest_refl_result
                    target_step = earliest_refl_step_idx
                else:
                    target_refl = result["reflection_summary"]
                    target_step = steps[-1].get("step_index", len(steps) - 1) if steps else 0

                result["root_cause_step_index"] = target_step
                result["root_cause_error_type"] = _derive_error_type_from_reflection(target_refl)
                result["root_cause_source"] = "reflector"

            # Tier 4: Clean Run
            else:
                result["root_cause_step_index"] = None
                result["root_cause_error_type"] = None
                result["root_cause_source"] = "none"

            logger.info(
                "on_run_end run_id=%s total_steps=%d flags=%d "
                "root_cause=%s root_step=%s root_source=%s",
                run_id, len(steps), len(all_flags),
                result["root_cause_error_type"],
                result["root_cause_step_index"],
                result["root_cause_source"],
            )

        except Exception:
            logger.exception(
                "on_run_end error — failing open, run_id=%s",
                (metadata or {}).get("run_id", "?"),
            )
        finally:
            self._live_steps_recorded = []

        return result
