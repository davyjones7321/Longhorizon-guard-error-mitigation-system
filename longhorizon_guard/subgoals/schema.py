"""
Subgoal tracking data schemas and state payload definitions.

Exposes structured subgoal state for consumption by drift_monitor and GuardInterface.
"""

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class SubgoalStatus(str, Enum):
    """Lifecycle status of a tracked subgoal."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STALLED_ADVANCED = "stalled_advanced"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class SubgoalRecord:
    """A single tracked subgoal extracted from the agent's proposed plan.

    Attributes:
        subgoal_id: Unique identifier (e.g. 'subgoal_001').
        description: Description as stated by the agent's proposed plan.
        status: Current lifecycle status.
        step_indices: List of 0-based step indices mapped to this subgoal.
        created_at: Epoch timestamp when parsed.
        started_at: Epoch timestamp when status changed to in_progress.
        completed_at: Epoch timestamp when status changed to completed/failed/abandoned/stalled_advanced.
        completion_trigger: Evidence/reason for the last status transition.
    """

    subgoal_id: str
    description: str
    status: str = SubgoalStatus.NOT_STARTED.value
    step_indices: List[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    completion_trigger: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubgoalRecord":
        """Deserialize from plain dict."""
        return cls(**d)


@dataclass
class SubgoalStatePayload:
    """State snapshot format consumed by drift_monitor and external harnesses.

    This payload provides full visibility into active subgoal progress, duration,
    and completion statistics without requiring internal subgoals.py rebuilding.
    """

    current_subgoal_id: Optional[str]
    current_subgoal_description: Optional[str]
    status: str
    steps_in_current_subgoal: int
    time_in_current_subgoal_seconds: float
    completed_subgoals_count: int
    stalled_advanced_subgoals_count: int
    failed_subgoals_count: int
    abandoned_subgoals_count: int
    total_subgoals_count: int
    subgoal_progress_ratio: float
    active_subgoal_index: int
    is_fallback: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize payload to plain dict."""
        return asdict(self)
