"""
Audit module for detecting infrastructure noise and silent fallback contamination in trajectories.
Uses longhorizon_guard.storage.reader to inspect data without harness dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from .reader import find_all_runs

CONTAMINATION_MARKERS = [
    '"action_name": "unknown"',
    "ERROR: Gemini API request failed",
    "ERROR: OpenRouter API request failed",
    "OK: executed unknown",
    "rate limit exceeded",
    "429",
]

def audit_directory(search_dir: Union[str, Path]) -> Dict[str, Any]:
    """Scan directory for contaminated or rate-limited trajectory files."""
    runs = find_all_runs(search_dir)
    contaminated = []
    clean = []

    for run in runs:
        meta = run["metadata"]
        traj = run["trajectory"]
        meta_path = run["meta_path"]

        is_contaminated = False
        raw_str = (json.dumps(meta) + json.dumps(traj if traj else {})).lower()

        for marker in CONTAMINATION_MARKERS:
            if marker.lower() in raw_str:
                is_contaminated = True
                break

        if meta.get("final_status") == "error" and traj and traj.get("steps", []) == []:
            is_contaminated = True

        if is_contaminated:
            contaminated.append(meta_path)
        else:
            clean.append(meta_path)

    return {
        "total_scanned": len(runs),
        "contaminated_count": len(contaminated),
        "clean_count": len(clean),
        "contaminated_files": contaminated,
        "clean_files": clean,
    }

def print_audit_report(search_dir: Union[str, Path]) -> None:
    result = audit_directory(search_dir)
    print("=" * 60)
    print("  LONG-HORIZON GUARD AUDIT REPORT")
    print("=" * 60)
    print(f"Directory:           {search_dir}")
    print(f"Total Runs Scanned:  {result['total_scanned']}")
    print(f"Clean Runs:          {result['clean_count']}")
    print(f"Contaminated Runs:   {result['contaminated_count']}")
    print("-" * 60)
    for f in sorted(result["contaminated_files"]):
        print(f"  - [CONTAMINATED] {f}")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print_audit_report(target)
