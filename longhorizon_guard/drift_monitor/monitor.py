"""
Production DriftMonitor for longhorizon_guard.

Monitors agent trajectories for goal divergence and plan drift by analyzing
SubgoalStatePayload inputs from subgoals.py across execution steps.

Drift Signals:
  1. REPEATED_STALLED_SUBGOALS: >= 2 stalled_advanced subgoals in a single run.
  2. SLOW_PROGRESS_RATIO: >= 8 total steps with progress_ratio < 0.20 or step count in single subgoal >= 8.
  3. ACCUMULATED_SUBGOAL_FAILURES: >= 1 failed subgoals or total non-success subgoals >= 3.
  4. PATTERN_REPETITION_DRIFT: Repeated GuardInterface pattern match flags in trajectory history.

Design Principles:
  - Consumes existing SubgoalStatePayload (no duplicated subgoal logic).
  - Fail-open error handling: matching or evaluation failures never block execution.
  - Structured logging via logging.getLogger("longhorizon_guard.drift_monitor").
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from longhorizon_guard.drift_monitor.schema import DriftAssessment
from longhorizon_guard.subgoals.schema import SubgoalStatePayload

logger = logging.getLogger("longhorizon_guard.drift_monitor")


# Signal definitions & weights
SIGNAL_REPEATED_STALLED = "REPEATED_STALLED_SUBGOALS"
SIGNAL_SLOW_PROGRESS = "SLOW_PROGRESS_RATIO"
SIGNAL_ACCUMULATED_FAILURES = "ACCUMULATED_SUBGOAL_FAILURES"
SIGNAL_PATTERN_REPETITION = "PATTERN_REPETITION_DRIFT"

SIGNAL_WEIGHTS: Dict[str, float] = {
    SIGNAL_REPEATED_STALLED: 0.45,
    SIGNAL_SLOW_PROGRESS: 0.35,
    SIGNAL_ACCUMULATED_FAILURES: 0.40,
    SIGNAL_PATTERN_REPETITION: 0.30,
}


def _calculate_severity_level(score: float) -> str:
    """Map numeric drift score [0.0, 1.0] to discrete severity level."""
    if score <= 0.0:
        return "none"
    elif score < 0.35:
        return "low"
    elif score < 0.60:
        return "medium"
    elif score < 0.85:
        return "high"
    else:
        return "critical"


class DriftMonitor:
    """Stateful drift monitoring engine for a single agent trajectory run."""

    def __init__(
        self,
        drift_threshold: float = 0.35,
        min_steps_for_progress_check: int = 8,
        min_stalled_threshold: int = 2,
    ) -> None:
        self.drift_threshold = float(drift_threshold)
        self._min_steps_progress = min_steps_for_progress_check
        self._min_stalled_threshold = min_stalled_threshold
        self._pattern_flag_count: int = 0

    def reset(self) -> None:
        """Reset internal trajectory counters for a new run."""
        self._pattern_flag_count = 0

    def evaluate_step(
        self,
        subgoal_state: Optional[Dict[str, Any]],
        step_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        match_details: Optional[Dict[str, Any]] = None,
    ) -> DriftAssessment:
        """Evaluate a single step for trajectory drift signals.

        Args:
            subgoal_state: SubgoalStatePayload dict from subgoals tracker.
            step_record: Current step dict.
            history: List of past step dicts.
            match_details: Optional match_details dict from GuardInterface.

        Returns:
            DriftAssessment object.
        """
        try:
            step_idx = step_record.get("step_index", len(history))
            total_steps = len(history) + 1

            if match_details and match_details.get("layer"):
                self._pattern_flag_count += 1

            if not subgoal_state:
                return DriftAssessment(step_index=step_idx)

            cur_subgoal_id = subgoal_state.get("current_subgoal_id")
            stalled_count = subgoal_state.get("stalled_advanced_subgoals_count", 0)
            failed_count = subgoal_state.get("failed_subgoals_count", 0)
            progress_ratio = subgoal_state.get("subgoal_progress_ratio", 0.0)
            steps_in_subgoal = subgoal_state.get("steps_in_current_subgoal", 0)

            triggered_signals: List[str] = []
            reasons: List[str] = []

            # --- Signal 1: Repeated Stalled Subgoals ---
            # 1 alone is a slow step; >= 2 in a run is strong drift
            if stalled_count >= self._min_stalled_threshold:
                triggered_signals.append(SIGNAL_REPEATED_STALLED)
                reasons.append(
                    f"Repeated stalled subgoals detected: {stalled_count} subgoals force-advanced after step limit"
                )

            # --- Signal 2: Slow Progress Ratio ---
            # Total steps >= 8 with progress_ratio < 0.20 OR current subgoal stuck for >= 8 steps
            if (total_steps >= self._min_steps_progress and progress_ratio < 0.20) or steps_in_subgoal >= 8:
                triggered_signals.append(SIGNAL_SLOW_PROGRESS)
                reasons.append(
                    f"Slow progress ratio ({progress_ratio:.2f} after {total_steps} total steps, "
                    f"{steps_in_subgoal} steps in active subgoal '{cur_subgoal_id}')"
                )

            # --- Signal 3: Accumulated Subgoal Failures ---
            # 1 failed subgoal is normal recoverable behavior; >= 2 is repeated failure
            if failed_count >= 2 or (failed_count + stalled_count) >= 3:
                triggered_signals.append(SIGNAL_ACCUMULATED_FAILURES)
                reasons.append(
                    f"Accumulated subgoal failures: {failed_count} failed, {stalled_count} stalled-advanced"
                )

            # --- Signal 4: Pattern Repetition Drift ---
            if self._pattern_flag_count >= 2:
                triggered_signals.append(SIGNAL_PATTERN_REPETITION)
                reasons.append(
                    f"Pattern repetition drift: {self._pattern_flag_count} GuardInterface pattern flags triggered"
                )

            # Calculate score and severity
            raw_score = sum(SIGNAL_WEIGHTS.get(sig, 0.25) for sig in triggered_signals)
            severity_score = min(1.0, round(raw_score, 3))
            # FIX F-09: drift_detected requires severity_score >= drift_threshold (default 0.35).
            # Removed redundant 'or len(triggered_signals) > 0' clause which made severity_score dead code.
            drift_detected = severity_score >= self.drift_threshold
            severity_level = _calculate_severity_level(severity_score if drift_detected else 0.0)

            assessment = DriftAssessment(
                drift_detected=drift_detected,
                severity_score=severity_score,
                severity_level=severity_level,
                triggered_signals=triggered_signals,
                reasons=reasons,
                step_index=step_idx,
                subgoal_id=cur_subgoal_id,
            )

            if drift_detected:
                logger.info(
                    "drift_detected step=%d subgoal=%s score=%.2f severity=%s signals=%s reasons=%s",
                    step_idx, cur_subgoal_id, severity_score, severity_level,
                    triggered_signals, reasons,
                )

            return assessment

        except Exception:
            logger.exception("evaluate_step error in DriftMonitor — failing open")
            return DriftAssessment(step_index=step_record.get("step_index", -1))

    def evaluate_boundary(
        self,
        subgoal_id: str,
        subgoal_status: str,
        step_history: List[Dict[str, Any]],
        subgoal_state: Optional[Dict[str, Any]] = None,
    ) -> DriftAssessment:
        """Reevaluate trajectory drift at subgoal checkpoint boundary."""
        dummy_step = {"step_index": len(step_history)}
        return self.evaluate_step(
            subgoal_state=subgoal_state,
            step_record=dummy_step,
            history=step_history,
        )

    def finalize_run(
        self,
        run_metadata: Dict[str, Any],
        trajectory: Dict[str, Any],
        subgoals_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce final drift verdict for the full run at on_run_end."""
        try:
            steps = trajectory.get("steps", [])
            last_step = steps[-1] if steps else {"step_index": len(steps)}

            # Build mock state payload from subgoals_summary if available
            state_dict = None
            if subgoals_summary:
                state_dict = {
                    "current_subgoal_id": "final",
                    "stalled_advanced_subgoals_count": subgoals_summary.get("stalled_advanced_count", 0),
                    "failed_subgoals_count": subgoals_summary.get("failed_count", 0),
                    "subgoal_progress_ratio": subgoals_summary.get("completed_count", 0) / max(subgoals_summary.get("total_subgoals", 1), 1),
                    "steps_in_current_subgoal": 0,
                }

            final_assessment = self.evaluate_step(
                subgoal_state=state_dict,
                step_record=last_step,
                history=steps[:-1] if steps else [],
            )

            return {
                "run_id": run_metadata.get("run_id", "?"),
                "total_steps": len(steps),
                "final_drift_assessment": final_assessment.to_dict(),
            }
        except Exception:
            logger.exception("finalize_run error in DriftMonitor — failing open")
            return {
                "run_id": run_metadata.get("run_id", "?"),
                "total_steps": 0,
                "final_drift_assessment": DriftAssessment().to_dict(),
            }
