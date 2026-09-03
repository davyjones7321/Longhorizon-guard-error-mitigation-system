"""
Data schema for in-flight drift monitoring (Phase 5).

Defines DriftAssessment output structure consumed by GuardInterface and external harnesses.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DriftAssessment:
    """Structured drift assessment produced by DriftMonitor.

    Attributes:
        drift_detected: Boolean flag indicating if trajectory is drifting off-plan.
        severity_score: Floating point score between 0.0 (no drift) and 1.0 (critical drift).
        severity_level: Human-readable severity tag ('none', 'low', 'medium', 'high', 'critical').
        triggered_signals: List of drift signal codes that fired.
        reasons: List of human-readable diagnostic messages.
        step_index: Current step index evaluated.
        subgoal_id: Current active subgoal ID when evaluated.
    """

    drift_detected: bool = False
    severity_score: float = 0.0
    severity_level: str = "none"
    triggered_signals: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    step_index: int = -1
    subgoal_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize assessment to plain JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftAssessment":
        """Deserialize from dict."""
        return cls(**d)
