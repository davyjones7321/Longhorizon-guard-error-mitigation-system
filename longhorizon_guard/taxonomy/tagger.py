"""
Manual failure-tagging CLI for longhorizon_guard.

Steps through failed trajectories one by one, displays the full log, and
prompts for a tag + optional note. Tags are stored in tags.csv.

Usage:
    python -m longhorizon_guard.taxonomy.tagger --data-dir output
    python -m longhorizon_guard.taxonomy.tagger --data-dir findings/all_clean_outputs.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union

from .categories import DEFAULT_TAGS
from ..storage.reader import load_dataset, load_run


def _load_existing_tags(tags_path: Path) -> Dict[tuple, dict]:
    """Return {(task_id, trial): {tag, note}} for already-tagged runs."""
    existing: Dict[tuple, dict] = {}
    if not tags_path.exists():
        return existing
    with open(tags_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["task_id"], int(row["trial_number"]))
            existing[key] = {"tag": row["tag"], "note": row.get("note", "")}
    return existing


def _append_tag(tags_path: Path, task_id: str, trial: int, tag: str, note: str) -> None:
    write_header = not tags_path.exists()
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tags_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "trial_number", "tag", "note"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "task_id": task_id,
            "trial_number": trial,
            "tag": tag,
            "note": note,
        })


def _print_trajectory(traj: Optional[dict]) -> None:
    if not traj or "steps" not in traj:
        print("  [No trajectory steps recorded]")
        return
    for step in traj["steps"]:
        print(f"\n{'─'*60}")
        print(f"  Step {step.get('step_index', '?')}  [{step.get('action_name', 'unknown')}]")
        print(f"  Timestamp: {step.get('timestamp')}")
        print(f"  Reasoning : {step.get('reasoning')}")
        print(f"  Args      : {step.get('action_args')}")
        if step.get("tool_response"):
            print(f"  Response  : {step.get('tool_response')}")
        if step.get("state_snapshot"):
            print(f"  State     : {step.get('state_snapshot')}")
    print(f"\n{'='*60}")


def tag_failures(
    data_dir: Union[str, Path],
    tags_path: Optional[Union[str, Path]] = None,
    tag_list: Optional[List[str]] = None,
    *,
    verbose: bool = True,
) -> int:
    """Interactive CLI loop for tagging failed runs."""
    data_path = Path(data_dir)

    if tags_path is None:
        if data_path.is_file():
            tags_path = data_path.parent / "tags.csv"
        else:
            tags_path = data_path / "tags.csv"
    else:
        tags_path = Path(tags_path)

    if tag_list is None:
        tag_list = list(DEFAULT_TAGS)

    existing = _load_existing_tags(tags_path)
    all_runs = load_dataset(data_path)

    failed_runs = [
        run for run in all_runs
        if str(run["metadata"].get("final_status", "")).lower() in ("fail", "timeout", "error")
    ]

    if not failed_runs:
        print("No failed runs found. Nothing to tag.")
        return 0

    print(f"\nFound {len(failed_runs)} failed run(s). Existing tags: {len(existing)}")
    print("For each run you can:")
    print("  [s] skip  •  [v] view trajectory  •  [q] quit early")
    print("  Otherwise type a tag number to assign it.\n")

    tagged_count = 0

    for run in failed_runs:
        meta = run["metadata"]
        task_id = meta.get("task_id", "unknown")
        trial = meta.get("trial_number", 1)
        status_str = meta.get("final_status", "unknown")
        steps_count = meta.get("total_steps_taken", 0)
        dur = meta.get("duration_seconds", 0.0)

        key = (task_id, trial)
        if key in existing:
            print(f"  [skip] {task_id} trial {trial} -- already tagged as '{existing[key]['tag']}'")
            continue

        print(f"\n{'-'*60}")
        print(f"  {task_id}  trial {trial}  status={status_str}  steps={steps_count}  dur={dur:.1f}s")
        if meta.get("grader_notes"):
            print(f"  notes: {meta.get('grader_notes')}")

        print()
        for idx, t in enumerate(tag_list, 1):
            print(f"    {idx}. {t}")
        print("    v. view trajectory")
        print("    s. skip")
        print("    q. quit")

        choice = input("\n> ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "v":
            _print_trajectory(run["trajectory"])
            choice = input("\nTag (number, s=skip, q=quit): > ").strip().lower()
            if choice in ("q", "s", ""):
                continue

        try:
            idx = int(choice) - 1
            tag = tag_list[idx]
        except (ValueError, IndexError):
            print("  Invalid choice — skipping.")
            continue

        note = input("  Optional note (Enter to skip): ").strip()
        _append_tag(tags_path, task_id, trial, tag, note)
        existing[key] = {"tag": tag, "note": note}
        tagged_count += 1
        if verbose:
            print(f"  [OK] Tagged as '{tag}'")

    print(f"\nDone. Tagged {tagged_count} run(s). Tags saved to {tags_path}")
    return tagged_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual failure-tagging CLI for longhorizon_guard.")
    parser.add_argument("--data-dir", default="output", help="Path to output directory or consolidated dataset JSON file.")
    parser.add_argument("--tags-path", default=None, help="Path to tags.csv destination.")
    args = parser.parse_args()
    tag_failures(data_dir=args.data_dir, tags_path=args.tags_path)
