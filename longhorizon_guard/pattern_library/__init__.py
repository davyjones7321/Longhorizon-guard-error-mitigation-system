"""
Pattern library subpackage for longhorizon_guard (Phase 4).
Stores past failure patterns and enables vector/semantic retrieval for pre-flight plan checking.
"""

from longhorizon_guard.pattern_library.schema import PatternEntry, TrajectorySnippet
from longhorizon_guard.pattern_library.store import load_patterns, save_patterns

__all__ = ["PatternEntry", "TrajectorySnippet", "load_patterns", "save_patterns"]
