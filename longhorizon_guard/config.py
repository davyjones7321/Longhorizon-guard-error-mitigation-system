"""
Configuration management for longhorizon_guard.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardConfig:
    """Centralized configuration for GuardInterface and monitoring sub-engines."""

    pattern_library_path: Optional[str] = None
    similarity_threshold: float = 0.82
    max_subgoal_steps: int = 10
    drift_threshold: float = 0.35
    reflection_step_interval: int = 5
    fail_open: bool = True
    block_on_critical: bool = False
    enable_memory: bool = False
    memory_storage_path: Optional[str] = None
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"

    @classmethod
    def from_env(cls) -> "GuardConfig":
        return cls(
            pattern_library_path=os.getenv("GUARD_PATTERN_LIB_PATH"),
            similarity_threshold=float(os.getenv("GUARD_SIMILARITY_THRESHOLD", "0.82")),
            max_subgoal_steps=int(os.getenv("GUARD_MAX_SUBGOAL_STEPS", "10")),
            drift_threshold=float(os.getenv("GUARD_DRIFT_THRESHOLD", "0.35")),
            reflection_step_interval=int(os.getenv("GUARD_REFLECTION_INTERVAL", "5")),
            fail_open=os.getenv("GUARD_FAIL_OPEN", "true").lower() in ("true", "1", "yes"),
            block_on_critical=os.getenv("GUARD_BLOCK_ON_CRITICAL", "false").lower() in ("true", "1", "yes"),
            enable_memory=os.getenv("GUARD_ENABLE_MEMORY", "false").lower() in ("true", "1", "yes"),
            memory_storage_path=os.getenv("GUARD_MEMORY_STORAGE_PATH"),
            provider=os.getenv("GUARD_PROVIDER", "gemini"),
            model=os.getenv("GUARD_MODEL", "gemini-2.5-flash"),
        )
