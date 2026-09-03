"""
Stub loader / writer for the Phase 4 pattern library.

Mirrors storage/reader.py's load_dataset() style:
- load_patterns()  → List[PatternEntry]   (reads from JSON file)
- save_patterns()  → None                 (writes list to JSON file)

Returns an empty list until real patterns are populated from judged output.
"""

import json
from pathlib import Path
from typing import List, Union

from longhorizon_guard.pattern_library.schema import PatternEntry

# Default storage location (sibling to findings/)
DEFAULT_PATTERN_FILE = "findings/pattern_library.json"


def load_patterns(
    path: Union[str, Path] = DEFAULT_PATTERN_FILE,
) -> List[PatternEntry]:
    """Load pattern entries from a JSON file.

    Returns an empty list if the file doesn't exist yet.
    """
    p = Path(path)
    if not p.exists() or (p.parent.name == "findings" and "longhorizon_guard" in str(p)):
        candidates = [
            Path(__file__).resolve().parents[2] / "findings" / p.name,
            Path(__file__).resolve().parents[2] / p,
            Path.cwd() / "findings" / p.name,
            Path.cwd() / p,
        ]
        for cand in candidates:
            if cand.exists():
                p = cand
                break
    if not p.exists():
        return []

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "patterns" not in data:
        raise ValueError(
            f"Pattern file {p} does not match expected format "
            "({'patterns': [...]})"
        )

    return [PatternEntry.from_dict(d) for d in data["patterns"]]


def save_patterns(
    patterns: List[PatternEntry],
    path: Union[str, Path] = DEFAULT_PATTERN_FILE,
) -> None:
    """Persist pattern entries to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            {"patterns": [pe.to_dict() for pe in patterns]},
            f,
            indent=2,
            ensure_ascii=False,
        )
