"""
Data schema for the Phase 4 pattern library.

Defines the structure for a single failure pattern entry, using the same
7-category taxonomy from taxonomy/categories.py.  Pattern entries will be
populated once the LLM-as-judge calibration (Phase 3) passes and judged
output is available.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from longhorizon_guard.taxonomy.categories import DEFAULT_TAGS


@dataclass
class TrajectorySnippet:
    """A minimal trajectory excerpt illustrating the failure pattern."""

    step_index: int
    reasoning: str = ""
    action_name: str = ""
    action_args: Dict[str, Any] = field(default_factory=dict)
    tool_response: Optional[str] = None


@dataclass
class PatternEntry:
    """A single failure pattern in the pattern library.

    Attributes:
        pattern_id:          Unique identifier for this pattern.
        category:            One of DEFAULT_TAGS (e.g. 'planning_error').
        trigger_description: Human-readable description of the conditions
                             that trigger this failure pattern.
        example_snippet:     A short trajectory excerpt showing the pattern
                             in action.
        safe_alternative:    Description of the corrective or safe behavior
                             the agent should have taken instead.
        source_run_ids:      Run IDs from which this pattern was extracted.
        confidence:          How confidently this pattern generalises (0–1).
    """

    pattern_id: str
    category: str
    trigger_description: str
    example_snippet: List[TrajectorySnippet] = field(default_factory=list)
    safe_alternative: str = ""
    source_run_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.category not in DEFAULT_TAGS:
            raise ValueError(
                f"Invalid category '{self.category}'. "
                f"Must be one of: {DEFAULT_TAGS}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatternEntry":
        """Deserialise from a plain dict."""
        snippets = [
            TrajectorySnippet(**s) if isinstance(s, dict) else s
            for s in d.get("example_snippet", [])
        ]
        return cls(
            pattern_id=d["pattern_id"],
            category=d["category"],
            trigger_description=d["trigger_description"],
            example_snippet=snippets,
            safe_alternative=d.get("safe_alternative", ""),
            source_run_ids=d.get("source_run_ids", []),
            confidence=d.get("confidence", 1.0),
        )
