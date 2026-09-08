"""
Production PlanReflector for longhorizon_guard (Phase 5).

Performs periodic re-evaluation of the agent's plan during execution by synthesizing
evidence from GuardInterface, subgoals, and drift_monitor.

Questions answered jointly:
  (a) Is the ORIGINAL plan still achievable given what has actually happened?
  (b) Is the agent's current approach still the best path, or should the plan be revised?

Design Principles:
  - Non-Blocking: Suggests plan revisions without halting host execution.
  - Periodic Triggers: Fires on subgoal boundary transitions AND every N steps (default=5).
  - Coincidence Guard: Prevents double-firing on the same step when boundary & step interval coincide.
  - Thread-Isolated Timeout: 2.0s match timeout protection.
  - Fail-Open: Any exception in reflection logic returns plan_still_valid=True.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from longhorizon_guard.reflector.schema import ReflectionResult

logger = logging.getLogger("longhorizon_guard.reflector")

DEFAULT_STEP_INTERVAL: int = 5
DEFAULT_TIMEOUT_SECONDS: float = 2.0


class PlanReflector:
    """Stateful PreFlect plan re-evaluation engine."""

    def __init__(
        self,
        step_interval: int = DEFAULT_STEP_INTERVAL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.step_interval = max(1, step_interval)
        self.timeout_seconds = timeout_seconds
        self._last_evaluated_step: int = -1
        self._task_description: str = ""
        self._proposed_plan: str = ""

    def init_plan(self, task_description: str, proposed_plan: str) -> None:
        """Initialize original task description and proposed plan context."""
        self._task_description = task_description or ""
        self._proposed_plan = proposed_plan or ""
        self._last_evaluated_step = -1

    def evaluate(
        self,
        step_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        subgoal_state: Optional[Dict[str, Any]] = None,
        drift_assessment: Optional[Dict[str, Any]] = None,
        match_details: Optional[Dict[str, Any]] = None,
        trigger_type: str = "step_interval",
        force: bool = False,
    ) -> ReflectionResult:
        """Perform periodic plan re-evaluation with thread-isolated timeout.

        Args:
            step_record: Current step dict.
            history: Trajectory step history.
            subgoal_state: SubgoalStatePayload dict.
            drift_assessment: DriftAssessment dict.
            match_details: GuardInterface match_details dict.
            trigger_type: 'step_interval' or 'subgoal_boundary'.
            force: Force evaluation regardless of step interval logic.

        Returns:
            ReflectionResult dataclass.
        """
        step_idx = step_record.get("step_index", len(history))

        # Coincidence Guard: Do not double-fire if evaluated on the same step
        if not force and step_idx == self._last_evaluated_step:
            logger.debug("Coincidence guard: step %d already evaluated, skipping", step_idx)
            return ReflectionResult(
                plan_still_valid=True,
                revision_suggested=False,
                trigger_type=trigger_type,
                step_index=step_idx,
            )

        # Step Interval Check (for step_interval trigger)
        if trigger_type == "step_interval" and not force:
            if step_idx <= 0 or (step_idx % self.step_interval != 0):
                return ReflectionResult(
                    plan_still_valid=True,
                    revision_suggested=False,
                    trigger_type=trigger_type,
                    step_index=step_idx,
                )

        result = ReflectionResult(trigger_type=trigger_type, step_index=step_idx)

        def _do_evaluate():
            nonlocal result
            try:
                result = self._evaluate_impl(
                    step_idx=step_idx,
                    subgoal_state=subgoal_state,
                    drift_assessment=drift_assessment,
                    match_details=match_details,
                    trigger_type=trigger_type,
                )
            except Exception:
                logger.exception("evaluate error in PlanReflector thread — failing open")

        thread = threading.Thread(target=_do_evaluate, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            logger.warning(
                "PlanReflector timeout step=%d timeout=%.1fs — failing open",
                step_idx, self.timeout_seconds,
            )
            return ReflectionResult(
                plan_still_valid=True,
                revision_suggested=False,
                revision_reasoning="Evaluation timed out — failing open",
                trigger_type=trigger_type,
                step_index=step_idx,
            )

        self._last_evaluated_step = step_idx

        if result.revision_suggested:
            logger.info(
                "plan_revision_suggested step=%d trigger=%s reasoning=%s confidence=%.2f",
                step_idx, trigger_type, result.revision_reasoning, result.confidence,
            )

        return result

    def _evaluate_impl(
        self,
        step_idx: int,
        subgoal_state: Optional[Dict[str, Any]],
        drift_assessment: Optional[Dict[str, Any]],
        match_details: Optional[Dict[str, Any]],
        trigger_type: str,
    ) -> ReflectionResult:
        """Internal synthesis of evidence from subgoals, drift_monitor, and GuardInterface."""
        evidence_sources: List[str] = []
        invalidation_reasons: List[str] = []

        # 1. Synthesize Subgoal Evidence
        failed_count = 0
        stalled_count = 0
        progress_ratio = 1.0
        cur_subgoal_id = None

        if subgoal_state:
            evidence_sources.append("subgoals")
            failed_count = subgoal_state.get("failed_subgoals_count", 0)
            stalled_count = subgoal_state.get("stalled_advanced_subgoals_count", 0)
            progress_ratio = subgoal_state.get("subgoal_progress_ratio", 1.0)
            cur_subgoal_id = subgoal_state.get("current_subgoal_id")

            # 1 failed subgoal is normal recoverable agent behavior; >= 2 is repeated failure
            if failed_count >= 2:
                invalidation_reasons.append(
                    f"{failed_count} subgoals failed explicitly (active: '{cur_subgoal_id}')"
                )

            if stalled_count >= 2:
                invalidation_reasons.append(
                    f"{stalled_count} subgoal(s) force-advanced due to step limit stalls"
                )

        # 2. Synthesize Drift Monitor Evidence
        drift_severity = 0.0
        drift_detected = False

        if drift_assessment:
            evidence_sources.append("drift_monitor")
            drift_detected = drift_assessment.get("drift_detected", False)
            drift_severity = drift_assessment.get("severity_score", 0.0)
            drift_level = drift_assessment.get("severity_level", "none")

            if drift_level in ("high", "critical") or drift_severity >= 0.60 or (drift_detected and (stalled_count + failed_count) >= 1):
                signals = drift_assessment.get("triggered_signals", [])
                invalidation_reasons.append(
                    f"Drift monitor severity high ({drift_severity:.2f}, signals: {','.join(signals)})"
                )

        # 3. Synthesize GuardInterface Pattern Evidence
        if match_details and match_details.get("layer"):
            evidence_sources.append("guard_interface")
            cat = match_details.get("category", "")
            conf = match_details.get("confidence", 0.0)
            if conf >= 0.40 and cat in ("planning_error", "reflection_error"):
                invalidation_reasons.append(
                    f"GuardInterface flagged severe {cat} pattern (confidence={conf:.2f})"
                )

        # Combined Judgment
        plan_still_valid = len(invalidation_reasons) == 0
        revision_suggested = not plan_still_valid

        if revision_suggested:
            reasoning_str = (
                f"Plan re-evaluation at step {step_idx} ({trigger_type}): "
                + "; ".join(invalidation_reasons)
                + ". Original plan strategy is no longer optimal — revision recommended."
            )
            confidence = min(0.95, 0.50 + len(invalidation_reasons) * 0.20)
        else:
            reasoning_str = f"Plan re-evaluation at step {step_idx} ({trigger_type}): Original plan remains valid and achievable."
            confidence = 0.85

        return ReflectionResult(
            plan_still_valid=plan_still_valid,
            revision_suggested=revision_suggested,
            revision_reasoning=reasoning_str,
            confidence=round(confidence, 2),
            trigger_type=trigger_type,
            step_index=step_idx,
            evidence_sources=evidence_sources,
        )
