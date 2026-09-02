"""
Subgoal decomposition & checkpoint tracking subpackage for longhorizon_guard (Phase 5).
Tracks agent-declared task decomposition and step-by-step checkpoint completion.
"""

from longhorizon_guard.subgoals.schema import (
    SubgoalRecord,
    SubgoalStatus,
    SubgoalStatePayload,
)
from longhorizon_guard.subgoals.tracker import (
    SubgoalTracker,
    parse_plan_subgoals,
)

__all__ = [
    "SubgoalRecord",
    "SubgoalStatus",
    "SubgoalStatePayload",
    "SubgoalTracker",
    "parse_plan_subgoals",
]
