"""
longhorizon_guard: Long-horizon agent error mitigation system.
"""

__version__ = "0.2.0"

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.config import GuardConfig
from longhorizon_guard.subgoals.tracker import SubgoalTracker
from longhorizon_guard.drift_monitor.monitor import DriftMonitor
from longhorizon_guard.reflector.reflector import PlanReflector
from longhorizon_guard.integrations.client_wrapper import wrap_guard
from longhorizon_guard.integrations.langchain_callback import LongHorizonGuardCallback
from longhorizon_guard.proxy import run_proxy

__all__ = [
    "__version__",
    "GuardInterface",
    "GuardConfig",
    "SubgoalTracker",
    "DriftMonitor",
    "PlanReflector",
    "wrap_guard",
    "LongHorizonGuardCallback",
    "run_proxy",
]
