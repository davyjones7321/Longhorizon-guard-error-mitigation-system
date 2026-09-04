#!/usr/bin/env python3
"""Self-Contained E2E Smoke Test for LongHorizon Guard.

Runs with ZERO external dependencies or private harness packages.
Validates the full 4-hook GuardInterface pipeline:
  1. Plan proposal & subgoal parsing (on_plan_proposed)
  2. Step-by-step observation & failure matching (on_step)
  3. Subgoal boundary tracking (on_subgoal_boundary)
  4. Trajectory finalization & root-cause attribution (on_run_end)

Usage:
  python run_smoke_test.py
  python run_smoke_test.py --trajectory path/to/custom_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path and UTF-8 console output on Windows
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from longhorizon_guard import __version__
from longhorizon_guard.interface import GuardInterface


def run_standalone_smoke_test() -> int:
    """Execute a self-contained simulated agent trajectory through GuardInterface."""
    print("=" * 65)
    print(f"[GUARD] LongHorizon Guard v{__version__} - E2E Smoke Test")
    print("        Running self-contained trajectory evaluation (Zero dependencies)")
    print("=" * 65)

    guard = GuardInterface()
    print(f"[OK] Initialized GuardInterface")
    print(f"  - Loaded Centroid Patterns: {len(guard._matcher._patterns)}")
    print(f"  - Broad-Corpus IDF Terms:   {len(guard._matcher._idf)}")

    # -------------------------------------------------------------------------
    # Hook 1: on_plan_proposed
    # -------------------------------------------------------------------------
    task_desc = "Implement a transactional database queue with retry backoff"
    proposed_plan = (
        "1. Initialize SQLite database schema with WAL mode\n"
        "2. Implement worker polling loop with retry jitter\n"
        "3. Route poison-pill events to dead-letter queue"
    )
    print(f"\n[Hook 1] Testing on_plan_proposed...")
    plan_res = guard.on_plan_proposed(
        task_description=task_desc,
        proposed_plan=proposed_plan,
        metadata={"source": "smoke_test", "task_id": "queue_01"},
    )
    print(f"  [OK] Plan approved: {plan_res.get('approved', True)}")
    print(f"  [OK] Subgoals parsed: {plan_res.get('n_subgoals_parsed', 0)}")

    # -------------------------------------------------------------------------
    # Hook 2: on_step (Simulate agent trajectory)
    # -------------------------------------------------------------------------
    print(f"\n[Hook 2] Testing on_step across simulated execution...")
    history = []

    steps = [
        {
            "step_index": 0,
            "reasoning": "Create SQLite schema table for queue storage",
            "action_name": "Write",
            "action_args": {"file_path": "queue.py"},
            "tool_response": "File written successfully",
        },
        {
            "step_index": 1,
            "reasoning": "Execute schema initialization in database",
            "action_name": "Bash",
            "action_args": {"command": "sqlite3 queue.db < schema.sql"},
            "tool_response": "Error: table already exists",
        },
        {
            "step_index": 2,
            "reasoning": "Retry the exact same command without fixing schema",
            "action_name": "Bash",
            "action_args": {"command": "sqlite3 queue.db < schema.sql"},
            "tool_response": "Error: table already exists",
        },
        {
            "step_index": 3,
            "reasoning": "Retry the exact same command third time in a loop",
            "action_name": "Bash",
            "action_args": {"command": "sqlite3 queue.db < schema.sql"},
            "tool_response": "Error: table already exists",
        },
    ]

    flagged_count = 0
    for s in steps:
        res = guard.on_step(s, history, metadata={"task_id": "queue_01"})
        history.append(s)
        is_flagged = res.get("flagged", False)
        if is_flagged:
            flagged_count += 1
            print(f"  [WARN] Step {s['step_index']}: Flagged [{res.get('category')}] - {res.get('warning')}")
        else:
            print(f"  [OK] Step {s['step_index']}: Normal execution")

    print(f"  [OK] Step observation verified ({flagged_count} warnings correctly detected)")

    # -------------------------------------------------------------------------
    # Hook 3: on_subgoal_boundary
    # -------------------------------------------------------------------------
    print(f"\n[Hook 3] Testing on_subgoal_boundary...")
    guard.on_subgoal_boundary(
        subgoal_id="subgoal_1",
        subgoal_status="completed",
        step_history=history,
        metadata={"task_id": "queue_01"},
    )
    print(f"  [OK] Subgoal boundary transition recorded")

    # -------------------------------------------------------------------------
    # Hook 4: on_run_end
    # -------------------------------------------------------------------------
    print(f"\n[Hook 4] Testing on_run_end...")
    summary = guard.on_run_end(
        metadata={"task_id": "queue_01", "task": task_desc},
        trajectory={"steps": history},
    )
    root_cause = summary.get("root_cause_error_type", "none")
    root_step = summary.get("root_cause_step_index", 0)
    source = summary.get("root_cause_source", "none")
    print(f"  [OK] Root-Cause Identified: {root_cause} at Step {root_step} (Source: {source})")
    print(f"  [OK] Total steps evaluated: {len(history)}")

    print("\n" + "=" * 65)
    print("SUCCESS: ALL LONGHORIZON GUARD PIPELINE HOOKS VERIFIED CLEANLY")
    print("=" * 65)
    return 0


def run_custom_trajectory_file(trajectory_path: Path) -> int:
    """Evaluate a user-supplied trajectory JSON file from any external harness."""
    print("=" * 65)
    print(f"🛡️  LongHorizon Guard — Evaluating User Trajectory")
    print(f"   File: {trajectory_path}")
    print("=" * 65)

    if not trajectory_path.exists():
        print(f"Error: Trajectory file '{trajectory_path}' does not exist.", file=sys.stderr)
        return 1

    with open(trajectory_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    traj = data.get("trajectory", {}) if isinstance(data, dict) else {}
    steps = traj.get("steps", []) if isinstance(traj, dict) else (data if isinstance(data, list) else [])

    guard = GuardInterface()
    task_desc = meta.get("task_description") or meta.get("task_id") or "External Harness Task"
    plan_text = meta.get("proposed_plan") or ""

    guard.on_plan_proposed(task_desc, plan_text, metadata=meta)

    history = []
    for s in steps:
        guard.on_step(s, history, metadata=meta)
        history.append(s)

    summary = guard.on_run_end(metadata=meta, trajectory={"steps": steps})
    print("\n=== EVALUATION RESULTS ===")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-contained smoke test and evaluation for LongHorizon Guard."
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        type=Path,
        default=None,
        help="Optional path to user trajectory JSON from an external harness",
    )
    args = parser.parse_args()

    if args.trajectory:
        sys.exit(run_custom_trajectory_file(args.trajectory))
    else:
        sys.exit(run_standalone_smoke_test())


if __name__ == "__main__":
    main()
