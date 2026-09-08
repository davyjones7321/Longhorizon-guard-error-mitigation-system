"""
Production SubgoalTracker for longhorizon_guard.

Parses agent-declared subgoals from proposed plans, tracks lifecycle transitions
(not_started -> in_progress -> completed/failed/abandoned) on each step, fires
on_subgoal_boundary transition events, and exposes SubgoalStatePayload for drift_monitor.

Design Principles:
  - Agent-Declared: Tracks subgoals as proposed by the agent's plan (does not generate them).
  - Explicit Rules: Documented, deterministic step-to-subgoal status mapping rules.
  - Messiness Handling: Fallback to single-subgoal tracking if plan lacks structure.
  - Fail-Open: Graceful error handling; tracking failures never block execution.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from longhorizon_guard.subgoals.schema import (
    SubgoalRecord,
    SubgoalStatus,
    SubgoalStatePayload,
)

logger = logging.getLogger("longhorizon_guard.guard")

# Regex patterns for extracting declared subgoals from plan text
_NUMBERED_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Step\s*\d+|Subgoal\s*\d+|Phase\s*\d+|\d+)[\.\:\)]\s*(.+?)(?=\n\s*(?:Step\s*\d+|Subgoal\s*\d+|Phase\s*\d+|\d+)[\.\:\)]|\n\s*[\-\*]|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_BULLET_PATTERN = re.compile(
    r"(?:^|\n)\s*[\-\*]\s*(.+?)(?=\n\s*[\-\*]|\n\s*(?:Step|Subgoal|Phase|\d+)[\.\:\)]|\Z)",
    re.DOTALL,
)

_TRANSITION_WORDS_PATTERN = re.compile(
    r"\b(First|Second|Third|Fourth|Fifth|Then|Next|Finally)\b[\,\:\s]+(.+?)(?=\b(?:Second|Third|Fourth|Fifth|Then|Next|Finally)\b|\Z)",
    re.IGNORECASE,
)


def parse_plan_subgoals(proposed_plan: str) -> Tuple[List[SubgoalRecord], bool]:
    """Parse agent-declared subgoals from proposed plan text.

    Returns:
        (records, is_fallback)
    """
    if not proposed_plan or not proposed_plan.strip():
        return [
            SubgoalRecord(
                subgoal_id="subgoal_001",
                description="Execute full task (default single-subgoal fallback)",
            )
        ], True

    plan_text = proposed_plan.strip()
    extracted_descriptions: List[str] = []

    # Strategy 1: Numbered / Step / Subgoal prefix match
    matches = _NUMBERED_PATTERN.findall(plan_text)
    if len(matches) >= 2:
        extracted_descriptions = [m.strip().replace("\n", " ") for m in matches if m.strip()]

    # Strategy 2: Bullet list match
    if len(extracted_descriptions) < 2:
        matches = _BULLET_PATTERN.findall(plan_text)
        if len(matches) >= 2:
            extracted_descriptions = [m.strip().replace("\n", " ") for m in matches if m.strip()]

    # Strategy 3: Transition keywords (First, Next, Then, Finally)
    if len(extracted_descriptions) < 2:
        matches = _TRANSITION_WORDS_PATTERN.findall(plan_text)
        if len(matches) >= 2:
            extracted_descriptions = [
                f"{w} {desc}".strip().replace("\n", " ") for w, desc in matches if desc.strip()
            ]

    # Strategy 4: Line-by-line fallback if multiple clean non-empty lines exist
    if len(extracted_descriptions) < 2:
        lines = [line.strip() for line in plan_text.splitlines() if line.strip() and len(line.strip()) > 10]
        if len(lines) >= 2:
            extracted_descriptions = lines

    # Fallback if no clean multi-step structure was found
    if len(extracted_descriptions) < 2:
        clean_desc = plan_text[:200].replace("\n", " ")
        return [
            SubgoalRecord(
                subgoal_id="subgoal_001",
                description=f"Execute task: {clean_desc}",
            )
        ], True

    records: List[SubgoalRecord] = []
    now = time.time()
    for idx, desc in enumerate(extracted_descriptions, 1):
        # Truncate very long descriptions
        short_desc = desc[:200]
        records.append(
            SubgoalRecord(
                subgoal_id=f"subgoal_{idx:03d}",
                description=short_desc,
                created_at=now,
            )
        )

    return records, False


# ---------------------------------------------------------------------------
# Step outcome mapping rules (Explicit & Inspectable)
# ---------------------------------------------------------------------------
# RULE 1: Success Keywords in Tool Response
_SUCCESS_KEYWORDS = {
    "you pick up", "you take", "you open", "you close", "you clean",
    "you heat", "you cool", "you put", "you arrive at",
    "found", "success", "task complete", "unlocked",
}

_NEGATIVE_FOUND_PATTERN = re.compile(
    r"\b(?:not|0|none|no|nothing|never|couldn't|cannot)\s+(?:found|results?)\b|\b(?:not|nothing)\s+found\b",
    re.IGNORECASE,
)
_POSITIVE_FOUND_PATTERN = re.compile(r"\b(?:found|discovered)\b", re.IGNORECASE)

# RULE 2: Failure Keywords & Patterns in Tool Response (F-07)
# Negative patterns for benign phrasings where "error" is explicitly absent or zero:
_BENIGN_ERROR_PATTERNS = [
    re.compile(r"\b0\s+errors?\b", re.IGNORECASE),
    re.compile(r"\bno\s+errors?\b", re.IGNORECASE),
    re.compile(r"\berrors?\s*:\s*(?:0|none|null)\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+errors?\b", re.IGNORECASE),
]

_MULTI_WORD_FAILURE_PATTERNS = [
    re.compile(r"\bcannot\s+find\b", re.IGNORECASE),
    re.compile(r"\bnothing\s+happens\b", re.IGNORECASE),
    re.compile(r"\binvalid\s+action\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+open\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+take\b", re.IGNORECASE),
    re.compile(r"\bsyntax\s+error\b", re.IGNORECASE),
]

_SINGLE_WORD_FAILURE_PATTERN = re.compile(r"\b(?:error|errors|failed|failure)\b", re.IGNORECASE)


def _detect_tool_failure(tool_resp: str) -> Optional[str]:
    """Detect real failure indicators in tool response with word-boundary matching,
    excluding common benign phrasings (e.g. '0 errors', 'no error', 'error: none')."""
    if not tool_resp or not tool_resp.strip():
        return None

    # 1. Check explicit multi-word failure expressions
    for pat in _MULTI_WORD_FAILURE_PATTERNS:
        m = pat.search(tool_resp)
        if m:
            return m.group(0)

    # 2. Strip benign negative patterns before testing general error/failed words
    cleaned_resp = tool_resp
    for benign_pat in _BENIGN_ERROR_PATTERNS:
        cleaned_resp = benign_pat.sub(" ", cleaned_resp)

    # 3. Check single-word failure terms on word boundaries
    match = _SINGLE_WORD_FAILURE_PATTERN.search(cleaned_resp)
    if match:
        return match.group(0)

    return None


def _check_step_outcome(
    step_record: Dict[str, Any],
    active_subgoal: SubgoalRecord,
    next_subgoal: Optional[SubgoalRecord],
    subgoal_step_count: int,
    max_subgoal_steps: int = 10,
) -> Tuple[str, Optional[str]]:
    """Apply explicit outcome mapping rules to evaluate active subgoal state.

    Returns:
        (new_status, trigger_reason)
        Where new_status is one of: "in_progress", "completed", "failed", "stalled_advanced"
    """
    tool_resp = str(step_record.get("tool_response") or "").lower()
    reasoning = str(step_record.get("reasoning") or "").lower()
    action = str(step_record.get("action_name") or "").lower()
    action_args = str(step_record.get("action_args") or "").lower()

    combined_step_text = f"{reasoning} {action} {action_args} {tool_resp}"

    # --- Rule F1: Critical Failure Detection (F-07) ---
    failure_term = _detect_tool_failure(tool_resp)
    if failure_term:
        return SubgoalStatus.FAILED.value, f"Step tool response contained failure keyword '{failure_term}'"

    # --- Rule S1: Alignment with Next Subgoal (Early Advancement) ---
    if next_subgoal:
        next_desc = next_subgoal.description.lower()
        # Extract key content words (>3 chars) from next subgoal
        next_words = [w for w in re.findall(r"[a-z0-9]+", next_desc) if len(w) > 3]
        if next_words:
            matched_next_words = [w for w in next_words if w in f"{reasoning} {action} {action_args}"]
            # If 2 or more key words match next subgoal, previous active subgoal completed!
            if len(matched_next_words) >= 2:
                return SubgoalStatus.COMPLETED.value, f"Step actions aligned with next subgoal '{next_subgoal.subgoal_id}'"

    # --- Rule S2: Explicit Observation Success ---
    for kw in _SUCCESS_KEYWORDS:
        if kw in tool_resp:
            if kw == "found":
                if _NEGATIVE_FOUND_PATTERN.search(tool_resp) or not _POSITIVE_FOUND_PATTERN.search(tool_resp):
                    continue
            # If active subgoal step count >= 1 and success kw observed
            if subgoal_step_count >= 1:
                return SubgoalStatus.COMPLETED.value, f"Observation confirmed success keyword '{kw}'"

    # --- Rule S3: Max Step Threshold Auto-Advancement (F-08) ---
    # NOTE: The default threshold (10 steps) is an empirical heuristic baseline based on
    # typical multi-step actions (which empirical agent benchmarks show often take 7-12 steps)
    # without prematurely marking active subgoals as stalled. This threshold is uncalibrated
    # against domain-specific distributions and is fully configurable via constructor / GuardConfig.
    if subgoal_step_count >= max_subgoal_steps:
        return SubgoalStatus.STALLED_ADVANCED.value, (
            f"Subgoal step threshold reached ({subgoal_step_count} steps >= max {max_subgoal_steps} "
            "without completion — force advanced)"
        )

    # Default: Continue in progress
    return SubgoalStatus.IN_PROGRESS.value, None


class SubgoalTracker:
    """Stateful subgoal tracker for a single agent trajectory run.

    Args:
        max_subgoal_steps: Maximum steps permitted for an active subgoal before Rule S3
            auto-advances it as STALLED_ADVANCED. Defaults to 10 (uncalibrated baseline).
    """

    def __init__(self, max_subgoal_steps: int = 10) -> None:
        self.max_subgoal_steps = max(1, int(max_subgoal_steps))
        self._subgoals: List[SubgoalRecord] = []
        self._active_idx: int = 0
        self._is_fallback: bool = False
        self._step_counter: int = 0
        self._completed_count: int = 0
        self._stalled_advanced_count: int = 0
        self._failed_count: int = 0
        self._abandoned_count: int = 0

    @property
    def is_initialized(self) -> bool:
        return len(self._subgoals) > 0

    @property
    def subgoals(self) -> List[SubgoalRecord]:
        return self._subgoals

    @property
    def active_subgoal(self) -> Optional[SubgoalRecord]:
        if 0 <= self._active_idx < len(self._subgoals):
            return self._subgoals[self._active_idx]
        return None

    @property
    def is_fallback(self) -> bool:
        """Return True if SubgoalTracker is in single-subgoal fallback mode."""
        return self._is_fallback

    def init_plan(
        self,
        task_description: str,
        proposed_plan: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse plan and initialize tracked subgoal records (on_plan_proposed)."""
        try:
            self._subgoals, self._is_fallback = parse_plan_subgoals(proposed_plan)
            self._active_idx = 0
            self._step_counter = 0
            self._completed_count = 0
            self._stalled_advanced_count = 0
            self._failed_count = 0
            self._abandoned_count = 0

            # Start first subgoal as IN_PROGRESS
            if self._subgoals:
                first = self._subgoals[0]
                first.status = SubgoalStatus.IN_PROGRESS.value
                first.started_at = time.time()

            run_id = (metadata or {}).get("run_id", "?")
            logger.info(
                "SubgoalTracker initialized run_id=%s subgoals_count=%d fallback=%s",
                run_id, len(self._subgoals), self._is_fallback,
            )
            return {
                "subgoals_count": len(self._subgoals),
                "is_fallback": self._is_fallback,
                "subgoal_ids": [s.subgoal_id for s in self._subgoals],
            }
        except Exception:
            logger.exception("init_plan error in SubgoalTracker — degrading gracefully")
            # Fail-open fallback
            self._subgoals = [
                SubgoalRecord(
                    subgoal_id="subgoal_001",
                    description="Execute task (fallback on init error)",
                    status=SubgoalStatus.IN_PROGRESS.value,
                    started_at=time.time(),
                )
            ]
            self._is_fallback = True
            return {"subgoals_count": 1, "is_fallback": True, "subgoal_ids": ["subgoal_001"]}

    def process_step(
        self,
        step_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a single step and update active subgoal state (on_step).

        Returns:
            {
                "checkpoint_passed": bool,
                "transition_event": dict | None,
                "state_payload": SubgoalStatePayload dict
            }
        """
        try:
            result: Dict[str, Any] = {
                "checkpoint_passed": True,
                "transition_event": None,
                "state_payload": self.get_state_payload().to_dict(),
            }

            if not self._subgoals:
                return result

            step_idx = step_record.get("step_index", self._step_counter)
            self._step_counter = step_idx + 1

            if self._active_idx >= len(self._subgoals):
                # All declared subgoals already finished
                return result

            active = self._subgoals[self._active_idx]
            active.step_indices.append(step_idx)
            subgoal_step_count = len(active.step_indices)

            next_subgoal = (
                self._subgoals[self._active_idx + 1]
                if self._active_idx + 1 < len(self._subgoals)
                else None
            )

            # Map step outcome to status transition
            new_status, trigger_reason = _check_step_outcome(
                step_record, active, next_subgoal, subgoal_step_count, max_subgoal_steps=self.max_subgoal_steps
            )

            # Handle state transitions
            if new_status in (
                SubgoalStatus.COMPLETED.value,
                SubgoalStatus.STALLED_ADVANCED.value,
                SubgoalStatus.FAILED.value,
            ):
                now = time.time()
                active.status = new_status
                active.completed_at = now
                active.completion_trigger = trigger_reason

                if new_status == SubgoalStatus.COMPLETED.value:
                    self._completed_count += 1
                elif new_status == SubgoalStatus.STALLED_ADVANCED.value:
                    self._stalled_advanced_count += 1
                else:
                    self._failed_count += 1

                # Transition event for on_subgoal_boundary
                transition_event = {
                    "completed_subgoal_id": active.subgoal_id,
                    "completed_status": new_status,
                    "trigger_reason": trigger_reason,
                    "step_index": step_idx,
                    "next_subgoal_id": next_subgoal.subgoal_id if next_subgoal else None,
                }
                result["transition_event"] = transition_event
                result["checkpoint_passed"] = (new_status in (SubgoalStatus.COMPLETED.value, SubgoalStatus.STALLED_ADVANCED.value))

                logger.info(
                    "subgoal_transition run_id=%s subgoal_id=%s status=%s step=%d trigger=%s",
                    (metadata or {}).get("run_id", "?"), active.subgoal_id, new_status,
                    step_idx, trigger_reason,
                )

                # Advance to next subgoal
                self._active_idx += 1
                if self._active_idx < len(self._subgoals):
                    next_active = self._subgoals[self._active_idx]
                    next_active.status = SubgoalStatus.IN_PROGRESS.value
                    next_active.started_at = now

            result["state_payload"] = self.get_state_payload().to_dict()
            return result

        except Exception:
            logger.exception("process_step error in SubgoalTracker — failing open")
            return {
                "checkpoint_passed": True,
                "transition_event": None,
                "state_payload": {
                    "current_subgoal_id": None,
                    "current_subgoal_description": None,
                    "status": "error",
                    "steps_in_current_subgoal": 0,
                    "time_in_current_subgoal_seconds": 0.0,
                    "completed_subgoals_count": 0,
                    "stalled_advanced_subgoals_count": 0,
                    "failed_subgoals_count": 0,
                    "abandoned_subgoals_count": 0,
                    "total_subgoals_count": 0,
                    "subgoal_progress_ratio": 0.0,
                    "active_subgoal_index": 0,
                    "is_fallback": True,
                },
            }

    def update_subgoal_status(
        self,
        subgoal_id: str,
        status: str,
        step_index: int = -1,
        trigger_reason: str = "host_boundary_event",
    ) -> Optional[Dict[str, Any]]:
        """Directly update active subgoal status from an external boundary event."""
        try:
            if not self._subgoals:
                return None

            # Find matching subgoal by id or fallback to current active subgoal
            target_idx = None
            for idx, sg in enumerate(self._subgoals):
                if sg.subgoal_id == subgoal_id:
                    target_idx = idx
                    break
            if target_idx is None:
                target_idx = self._active_idx

            if target_idx >= len(self._subgoals):
                return None

            target = self._subgoals[target_idx]
            now = time.time()
            target.status = status
            target.completed_at = now
            target.completion_trigger = trigger_reason

            if status == SubgoalStatus.COMPLETED.value:
                self._completed_count += 1
            elif status == SubgoalStatus.STALLED_ADVANCED.value:
                self._stalled_advanced_count += 1
            elif status == SubgoalStatus.FAILED.value:
                self._failed_count += 1

            # If the updated subgoal was the active one and it reached terminal state, advance
            if target_idx == self._active_idx and status in (
                SubgoalStatus.COMPLETED.value,
                SubgoalStatus.STALLED_ADVANCED.value,
                SubgoalStatus.FAILED.value,
            ):
                self._active_idx += 1
                if self._active_idx < len(self._subgoals):
                    next_active = self._subgoals[self._active_idx]
                    next_active.status = SubgoalStatus.IN_PROGRESS.value
                    next_active.started_at = now

            logger.info(
                "subgoal_status_updated subgoal_id=%s status=%s active_idx=%d",
                target.subgoal_id, status, self._active_idx,
            )
            return {"updated_subgoal_id": target.subgoal_id, "status": status, "active_subgoal_index": self._active_idx}
        except Exception:
            logger.exception("update_subgoal_status error in SubgoalTracker — failing open")
            return None

    def get_state_payload(self) -> SubgoalStatePayload:
        """Expose current state payload format for drift_monitor consumption."""
        try:
            if not self._subgoals:
                return SubgoalStatePayload(
                    current_subgoal_id=None,
                    current_subgoal_description=None,
                    status="not_started",
                    steps_in_current_subgoal=0,
                    time_in_current_subgoal_seconds=0.0,
                    completed_subgoals_count=0,
                    stalled_advanced_subgoals_count=0,
                    failed_subgoals_count=0,
                    abandoned_subgoals_count=0,
                    total_subgoals_count=0,
                    subgoal_progress_ratio=0.0,
                    active_subgoal_index=0,
                    is_fallback=True,
                )

            total = len(self._subgoals)
            if self._active_idx < total:
                active = self._subgoals[self._active_idx]
                cur_id = active.subgoal_id
                cur_desc = active.description
                cur_status = active.status
                steps_count = len(active.step_indices)
                dur = time.time() - (active.started_at or time.time())
            else:
                last = self._subgoals[-1]
                cur_id = last.subgoal_id
                cur_desc = last.description
                cur_status = last.status
                steps_count = len(last.step_indices)
                dur = (last.completed_at or time.time()) - (last.started_at or time.time())

            progress_ratio = (self._completed_count + self._stalled_advanced_count) / max(total, 1)

            return SubgoalStatePayload(
                current_subgoal_id=cur_id,
                current_subgoal_description=cur_desc,
                status=cur_status,
                steps_in_current_subgoal=steps_count,
                time_in_current_subgoal_seconds=round(dur, 2),
                completed_subgoals_count=self._completed_count,
                stalled_advanced_subgoals_count=self._stalled_advanced_count,
                failed_subgoals_count=self._failed_count,
                abandoned_subgoals_count=self._abandoned_count,
                total_subgoals_count=total,
                subgoal_progress_ratio=round(progress_ratio, 3),
                active_subgoal_index=min(self._active_idx, total - 1),
                is_fallback=self._is_fallback,
            )
        except Exception:
            logger.exception("get_state_payload error — returning empty payload")
            return SubgoalStatePayload(
                current_subgoal_id=None,
                current_subgoal_description=None,
                status="error",
                steps_in_current_subgoal=0,
                time_in_current_subgoal_seconds=0.0,
                completed_subgoals_count=0,
                stalled_advanced_subgoals_count=0,
                failed_subgoals_count=0,
                abandoned_subgoals_count=0,
                total_subgoals_count=0,
                subgoal_progress_ratio=0.0,
                active_subgoal_index=0,
                is_fallback=True,
            )

    def finalize_run(
        self,
        run_metadata: Dict[str, Any],
        trajectory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce final subgoal summary at run end (on_run_end)."""
        try:
            now = time.time()
            # Any remaining unstarted/in_progress subgoals are marked ABANDONED
            for idx, s in enumerate(self._subgoals):
                if s.status in (SubgoalStatus.NOT_STARTED.value, SubgoalStatus.IN_PROGRESS.value):
                    s.status = SubgoalStatus.ABANDONED.value
                    s.completed_at = now
                    s.completion_trigger = "Run ended before subgoal completion"
                    self._abandoned_count += 1

            summary_records = [s.to_dict() for s in self._subgoals]

            return {
                "total_subgoals": len(self._subgoals),
                "completed_count": self._completed_count,
                "stalled_advanced_count": self._stalled_advanced_count,
                "failed_count": self._failed_count,
                "abandoned_count": self._abandoned_count,
                "is_fallback": self._is_fallback,
                "subgoals": summary_records,
            }
        except Exception:
            logger.exception("finalize_run error in SubgoalTracker — failing open")
            return {
                "total_subgoals": len(self._subgoals),
                "completed_count": self._completed_count,
                "stalled_advanced_count": self._stalled_advanced_count,
                "failed_count": self._failed_count,
                "abandoned_count": self._abandoned_count,
                "is_fallback": self._is_fallback,
                "subgoals": [],
            }
