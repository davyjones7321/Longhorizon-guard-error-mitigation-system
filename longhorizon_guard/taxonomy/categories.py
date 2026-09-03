"""
Standard error categories and taxonomy definitions for longhorizon_guard.
Paper references: AgentDebug (Paper 2) & PreFlect (Paper 3).
"""

from typing import List

# Default error tags used for root-cause classification
DEFAULT_TAGS: List[str] = [
    "planning_error",    # Plan was flawed from the start (wrong strategy or missed constraint)
    "memory_error",      # Lost track of facts or context from earlier steps
    "tool_use_error",    # Action execution failed (bad tool params, syntax error in call)
    "reflection_error",  # Misjudged progress (e.g. thought task was complete when it wasn't)
    "external_error",    # External API / environment error
    "grader_error",      # Grader mistake or regex mismatch on valid answer
    "other",             # Unclassified failure
]
