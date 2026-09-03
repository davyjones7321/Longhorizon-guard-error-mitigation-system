"""
Configuration management for longhorizon_guard.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class GuardConfig:
    """Configurable paths and settings for longhorizon_guard."""
    data_dir: Path = field(default_factory=lambda: Path("output"))
    tags_path: Path = field(default_factory=lambda: Path("output/tags.csv"))
    clean_runs_dir: Path = field(default_factory=lambda: Path("eval/clean_runs"))
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key_env_var: str = "GEMINI_API_KEY"

    @classmethod
    def from_env(cls) -> "GuardConfig":
        return cls(
            data_dir=Path(os.getenv("GUARD_DATA_DIR", "output")),
            tags_path=Path(os.getenv("GUARD_TAGS_PATH", "output/tags.csv")),
            clean_runs_dir=Path(os.getenv("GUARD_CLEAN_RUNS_DIR", "eval/clean_runs")),
            provider=os.getenv("GUARD_PROVIDER", "gemini"),
            model=os.getenv("GUARD_MODEL", "gemini-2.5-flash"),
        )
