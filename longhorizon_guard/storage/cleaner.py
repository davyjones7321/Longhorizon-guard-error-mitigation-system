"""
Data consolidation and deduplication utility (Phase 0).
Consolidates genuine agent-executed runs from multiple source directories into a clean dataset.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Union
from .reader import find_all_runs

def consolidate_clean_runs(
    search_dirs: List[Union[str, Path]],
    target_dir: Union[str, Path],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Scan source directories, filter out rate-limit/infra noise, deduplicate runs, and write clean runs to target_dir."""
    target_dir = Path(target_dir)
    seen_run_keys: Set[str] = set()

    genuine_runs = []
    infra_noise_runs = []

    for s_dir in search_dirs:
        s_dir = Path(s_dir)
        if not s_dir.exists():
            continue

        runs = find_all_runs(s_dir)
        for run in runs:
            meta = run["metadata"]
            traj = run["trajectory"]

            # Key for deduplication
            task_id = meta.get("task_id", "")
            trial_num = meta.get("trial_number", 0)
            run_key = f"{task_id}_trial_{trial_num}"

            # Check infra noise / rate limit
            raw_str = (json.dumps(meta) + json.dumps(traj if traj else {})).lower()
            is_infra_noise = False

            if "rate limit" in raw_str or "429" in raw_str or "ratelimit" in raw_str:
                is_infra_noise = True
            elif meta.get("final_status") == "error" and (not traj or traj.get("steps", []) == []):
                is_infra_noise = True
            elif meta.get("total_steps_taken", 0) == 1 and "action_name" in raw_str and '"action_name": "error"' in raw_str:
                is_infra_noise = True

            if is_infra_noise:
                infra_noise_runs.append(run)
            else:
                if run_key not in seen_run_keys:
                    seen_run_keys.add(run_key)
                    genuine_runs.append(run)

    if not dry_run and genuine_runs:
        target_dir.mkdir(parents=True, exist_ok=True)
        for run in genuine_runs:
            meta = run["metadata"]
            traj = run["trajectory"]
            task_id = meta.get("task_id", "unknown")
            trial_num = meta.get("trial_number", 1)

            dest_trial_dir = target_dir / task_id / f"trial_{trial_num}"
            dest_trial_dir.mkdir(parents=True, exist_ok=True)

            (dest_trial_dir / "run_metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if traj:
                (dest_trial_dir / "trajectory.json").write_text(
                    json.dumps(traj, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    return {
        "genuine_count": len(genuine_runs),
        "infra_noise_count": len(infra_noise_runs),
        "target_dir": str(target_dir),
        "genuine_runs": genuine_runs,
    }
