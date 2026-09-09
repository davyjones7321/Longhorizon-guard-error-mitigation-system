"""Session Working Memory for LongHorizon Guard.

Maintains in-flight trajectory execution state, sliding action windows for loop
detection, running semantic drift trajectories, and session-scoped memory advisories.
"""

import datetime
import json
from typing import Any, Dict, List, Optional

from longhorizon_guard.memory.schema import MemoryAdvisory


class WorkingMemory:
    """Unified session-scoped working memory container for an active trajectory."""

    def __init__(self, session_id: str = "default", max_window: int = 10) -> None:
        self.session_id: str = session_id
        self.max_window: int = max_window
        self.task_description: str = ""
        self.proposed_plan: str = ""
        self.metadata: Dict[str, Any] = {}
        self.created_at: str = datetime.datetime.now().isoformat()

        self.history: List[Dict[str, Any]] = []
        self.recent_actions: List[Dict[str, Any]] = []
        self.drift_trajectory: List[Dict[str, float]] = []
        self.advisories: List[MemoryAdvisory] = []
        self.tool_failure_counts: Dict[str, int] = {}

    def init_session(
        self,
        task_description: str,
        proposed_plan: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize or reset working memory for a new task session."""
        self.task_description = task_description
        self.proposed_plan = proposed_plan
        self.metadata = dict(metadata or {})
        self.history.clear()
        self.recent_actions.clear()
        self.drift_trajectory.clear()
        self.advisories.clear()
        self.tool_failure_counts.clear()
        self.created_at = datetime.datetime.now().isoformat()

    def record_step(
        self,
        step_record: Dict[str, Any],
        has_error: bool = False,
    ) -> None:
        """Record an executed step and update the action sliding window."""
        self.history.append(step_record)

        tool_name = str(step_record.get("action_name") or "tool")
        tool_args = step_record.get("action_args", {})
        sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

        self.recent_actions.append({"signature": sig, "failed": has_error})
        if len(self.recent_actions) > self.max_window:
            self.recent_actions.pop(0)

        if has_error:
            self.tool_failure_counts[tool_name] = self.tool_failure_counts.get(tool_name, 0) + 1

    def record_drift(
        self,
        step_index: int,
        drift_score: float,
        velocity: float = 0.0,
        acceleration: float = 0.0,
    ) -> None:
        """Track instantaneous and derivative semantic drift values."""
        self.drift_trajectory.append({
            "step_index": float(step_index),
            "drift_score": float(drift_score),
            "velocity": float(velocity),
            "acceleration": float(acceleration),
        })

    def add_advisory(self, advisory: MemoryAdvisory) -> None:
        """Record an advisory emitted by memory or guard subsystems."""
        self.advisories.append(advisory)

    def get_action_repeat_count(self, tool_name: str, tool_args: Any, failed_only: bool = False) -> int:
        """Count occurrences of identical action-argument signatures in the sliding window."""
        sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        count = 0
        for item in self.recent_actions:
            if item.get("signature") == sig:
                if not failed_only or item.get("failed"):
                    count += 1
        return count

    def get_latest_drift(self) -> Optional[Dict[str, float]]:
        """Return the most recent drift metrics snapshot, if any."""
        return self.drift_trajectory[-1] if self.drift_trajectory else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current working memory state to a JSON-compatible dictionary."""
        return {
            "session_id": self.session_id,
            "task_description": self.task_description,
            "proposed_plan": self.proposed_plan,
            "total_steps": len(self.history),
            "recent_actions": list(self.recent_actions),
            "drift_trajectory": list(self.drift_trajectory),
            "advisories_count": len(self.advisories),
            "tool_failure_counts": dict(self.tool_failure_counts),
            "created_at": self.created_at,
        }
