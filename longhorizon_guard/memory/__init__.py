"""LongHorizon Guard Memory Package.

Provides a 3-tier cognitive memory architecture:
1. Working Memory: In-flight session trajectory and sliding window.
2. Causal Error Graph: NetworkX-backed knowledge graph of tools, errors, subgoals, and recoveries.
3. Associative Retrieval: Fast (<15ms) Personalized PageRank multi-hop risk discovery.
"""

from longhorizon_guard.memory.schema import (
    ActionPatternNode,
    EdgeType,
    ErrorSignatureNode,
    MemoryAdvisory,
    NodeType,
    PrerequisiteEdge,
    PropagatesToEdge,
    RecoveryNode,
    RemediedByEdge,
    SubgoalNode,
    TaskConceptNode,
    TriggersErrorEdge,
)
from longhorizon_guard.memory.associative_engine import AssociativeMemoryEngine
from longhorizon_guard.memory.capability_classifier import (
    classify_action_capability,
    classify_subgoal_category,
)
from longhorizon_guard.memory.causal_graph import CausalErrorGraph
from longhorizon_guard.memory.memory_guard import MemoryGuard
from longhorizon_guard.memory.seed_loader import bootstrap_memory_graph
from longhorizon_guard.memory.vector_index import LocalConceptIndex
from longhorizon_guard.memory.working_memory import WorkingMemory

__all__ = [
    "NodeType",
    "EdgeType",
    "TaskConceptNode",
    "SubgoalNode",
    "ActionPatternNode",
    "ErrorSignatureNode",
    "RecoveryNode",
    "PrerequisiteEdge",
    "TriggersErrorEdge",
    "PropagatesToEdge",
    "RemediedByEdge",
    "MemoryAdvisory",
    "WorkingMemory",
    "CausalErrorGraph",
    "AssociativeMemoryEngine",
    "LocalConceptIndex",
    "bootstrap_memory_graph",
    "MemoryGuard",
    "classify_action_capability",
    "classify_subgoal_category",
]
