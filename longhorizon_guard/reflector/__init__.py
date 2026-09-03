"""
Pre-flight plan reflector subpackage for longhorizon_guard (Phase 4/5).
Validates proposed plans and performs periodic in-flight plan re-evaluation.
"""

from longhorizon_guard.reflector.schema import ReflectionResult
from longhorizon_guard.reflector.reflector import (
    PlanReflector,
    DEFAULT_STEP_INTERVAL,
    DEFAULT_TIMEOUT_SECONDS,
)

__all__ = [
    "ReflectionResult",
    "PlanReflector",
    "DEFAULT_STEP_INTERVAL",
    "DEFAULT_TIMEOUT_SECONDS",
]
