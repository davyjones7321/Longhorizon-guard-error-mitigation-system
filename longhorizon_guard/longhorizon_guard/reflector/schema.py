"""
Data schema for PreFlect periodic plan re-evaluation (Phase 5 Reflector).

Defines ReflectionResult output structure produced by PlanReflector.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ReflectionResult:
    """Structured result produced by PlanReflector periodic plan check.

    Attributes:
        plan_still_valid: Whether the original plan is still achievable given execution history.
        revision_suggested: Whether PlanReflector recommends plan revision.
        revision_reasoning: Plain-language diagnostic explanation citing concrete execution evidence.
        confidence: Confidence score of the assessment [0.0, 1.0].
        trigger_type: Reason for evaluation ('subgoal_boundary' or 'step_interval').
        step_index: Trajectory step index when evaluated.
        evidence_sources: List of component signal sources that contributed to the verdict.
    """

    plan_still_valid: bool = True
    revision_suggested: bool = False
    revision_reasoning: str = ""
    confidence: float = 1.0
    trigger_type: str = ""
    step_index: int = -1
    evidence_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to plain JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReflectionResult":
        """Deserialize from dict."""
        return cls(**d)
