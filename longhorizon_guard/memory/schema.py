"""Data schemas and typed contracts for LongHorizon Guard Memory.

Defines nodes, edges, and advisory contracts for the Causal Knowledge Graph
and working memory representations.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    """Canonical node categories in the Causal Memory Graph."""
    TASK_CONCEPT = "task_concept"
    SUBGOAL = "subgoal"
    ACTION_PATTERN = "action_pattern"
    ERROR_SIGNATURE = "error_signature"
    RECOVERY_ACTION = "recovery_action"


class EdgeType(str, Enum):
    """Canonical relational and causal edge types."""
    PREREQUISITE_OF = "prerequisite_of"
    TRIGGERS_ERROR = "triggers_error"
    PROPAGATES_TO = "propagates_to"
    REMEDIED_BY = "remedied_by"


@dataclass
class TaskConceptNode:
    """Represents a high-level task goal or concept."""
    concept_id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_type: str = NodeType.TASK_CONCEPT.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubgoalNode:
    """Represents a discrete milestone with explicit prerequisite requirements."""
    subgoal_id: str
    name: str
    description: str
    required_preconditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_type: str = NodeType.SUBGOAL.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionPatternNode:
    """Represents a normalized tool call or action signature."""
    action_id: str
    tool_name: str
    argument_pattern: str
    normalized_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_type: str = NodeType.ACTION_PATTERN.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorSignatureNode:
    """Represents a known failure mode or taxonomy category."""
    error_id: str
    category: str
    pattern_regex: str
    description: str
    severity: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_type: str = NodeType.ERROR_SIGNATURE.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryNode:
    """Represents a verified corrective action or safe alternative."""
    recovery_id: str
    action_description: str
    safe_alternative: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    verification_count: int = 1
    node_type: str = NodeType.RECOVERY_ACTION.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrerequisiteEdge:
    """Directional dependency edge indicating source_id MUST be fulfilled before target_id."""
    source_id: str
    target_id: str
    is_strict: bool = True
    description: str = ""
    edge_type: str = EdgeType.PREREQUISITE_OF.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TriggersErrorEdge:
    """Causal edge indicating an action pattern leads to an error signature."""
    source_id: str
    target_id: str
    frequency: int = 1
    confidence: float = 0.8
    sample_errors: List[str] = field(default_factory=list)
    edge_type: str = EdgeType.TRIGGERS_ERROR.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PropagatesToEdge:
    """Temporal cascade edge indicating an early error transitions into a downstream error."""
    source_id: str
    target_id: str
    step_lag: int = 1
    transition_prob: float = 0.5
    edge_type: str = EdgeType.PROPAGATES_TO.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RemediedByEdge:
    """Prescriptive edge mapping an error signature to an effective recovery action."""
    source_id: str
    target_id: str
    success_rate: float = 1.0
    times_applied: int = 1
    edge_type: str = EdgeType.REMEDIED_BY.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryAdvisory:
    """Actionable advisory emitted by the memory system during plan or step evaluation."""
    advisory_type: str            # 'prerequisite_violation', 'predicted_error', 'recovery_recommendation'
    severity: str                 # 'low', 'medium', 'high', 'critical'
    message: str
    confidence: float
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
