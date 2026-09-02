"""
In-flight drift monitor subpackage for longhorizon_guard (Phase 5).
Monitors step execution for trajectory divergence and goal drift.
"""

from longhorizon_guard.drift_monitor.schema import DriftAssessment
from longhorizon_guard.drift_monitor.monitor import (
    DriftMonitor,
    SIGNAL_REPEATED_STALLED,
    SIGNAL_SLOW_PROGRESS,
    SIGNAL_ACCUMULATED_FAILURES,
    SIGNAL_PATTERN_REPETITION,
)

__all__ = [
    "DriftAssessment",
    "DriftMonitor",
    "SIGNAL_REPEATED_STALLED",
    "SIGNAL_SLOW_PROGRESS",
    "SIGNAL_ACCUMULATED_FAILURES",
    "SIGNAL_PATTERN_REPETITION",
]
