"""
Standalone metadata and trajectory reader for longhorizon_guard.
Operates on plain dicts and JSON files according to SCHEMA.md contract.
No dependencies on harness code (evalharness).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REQUIRED_METADATA_FIELDS = ["run_id", "task_id", "trial_number", "final_status"]


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {path}: {exc}") from exc


def validate_metadata(meta: Dict[str, Any]) -> bool:
    """Validate that metadata dict conforms to SCHEMA.md contract."""
    if not isinstance(meta, dict):
        return False
    for field in REQUIRED_METADATA_FIELDS:
        if field not in meta:
            return False
    return True


def load_run(meta_path: Union[str, Path], traj_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load metadata and optional trajectory file into a unified dict.

    Returns standard shape:
        {
            "meta_path": str | None,
            "traj_path": str | None,
            "metadata": dict,
            "trajectory": dict | None
        }
    """
    meta_path = Path(meta_path)
    meta_data = load_json(meta_path)

    if not validate_metadata(meta_data):
        raise ValueError(f"Metadata file {meta_path} does not satisfy SCHEMA.md required fields {REQUIRED_METADATA_FIELDS}")

    if traj_path is None:
        possible_traj = meta_path.parent / "trajectory.json"
        if possible_traj.exists():
            traj_path = possible_traj

    traj_data = load_json(traj_path) if traj_path and Path(traj_path).exists() else None

    return {
        "meta_path": str(meta_path),
        "traj_path": str(traj_path) if traj_path else None,
        "metadata": meta_data,
        "trajectory": traj_data,
    }


def find_all_runs(search_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Recursively search directory for run_metadata.json / metadata.json and load each run into standard shape."""
    search_dir = Path(search_dir)
    runs = []
    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")

    meta_files = list(search_dir.rglob("run_metadata.json")) + list(search_dir.rglob("metadata.json"))
    meta_files = sorted(list(set(meta_files)))

    for meta_file in meta_files:
        try:
            run_data = load_run(meta_file)
            runs.append(run_data)
        except Exception:
            pass

    return runs


def load_consolidated(file_path: Union[str, Path], verbose: bool = True) -> List[Dict[str, Any]]:
    """Load a consolidated single JSON dataset (e.g. all_clean_outputs.json) and return runs in standard shape.

    Logs explicit warnings with index, source path, and reason if any item is skipped.
    """
    file_path = Path(file_path)
    data = load_json(file_path)

    if not isinstance(data, dict) or "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError(f"File {file_path} is not a valid consolidated dataset (missing top-level 'runs' list)")

    runs = []
    skipped_items = []

    for idx, item in enumerate(data["runs"]):
        if not isinstance(item, dict):
            reason = f"Item at index {idx} is not a dict ({type(item)})"
            skipped_items.append({"index": idx, "source": None, "reason": reason})
            if verbose:
                print(f"Warning: Skipping item [{idx}]: {reason}")
            continue

        source_path = item.get("source_path") or item.get("source_dir") or f"index_{idx}"

        if "metadata" not in item:
            reason = f"Item missing 'metadata' key"
            skipped_items.append({"index": idx, "source": source_path, "reason": reason})
            if verbose:
                print(f"Warning: Skipping item [{idx}] ({source_path}): {reason}")
            continue

        meta = item["metadata"]
        if not validate_metadata(meta):
            missing_keys = [k for k in REQUIRED_METADATA_FIELDS if k not in meta] if isinstance(meta, dict) else "not a dict"
            reason = f"Metadata failed validation (missing required keys: {missing_keys})"
            skipped_items.append({"index": idx, "source": source_path, "reason": reason})
            if verbose:
                print(f"Warning: Skipping item [{idx}] ({source_path}): {reason}")
            continue

        traj = item.get("trajectory")

        runs.append({
            "meta_path": str(source_path) if source_path else None,
            "traj_path": str(source_path) if source_path else None,
            "metadata": meta,
            "trajectory": traj,
        })

    if skipped_items and verbose:
        reasons_summary = "; ".join([f"[{s['index']} ({s['source']})]: {s['reason']}" for s in skipped_items])
        print(f"Warning: Skipped {len(skipped_items)} of {len(data['runs'])} runs in consolidated dataset {file_path}: {reasons_summary}")

    if not runs:
        raise ValueError(f"Consolidated dataset {file_path} contained zero valid runs matching SCHEMA.md")

    return runs


def load_dataset(path: Union[str, Path], verbose: bool = True) -> List[Dict[str, Any]]:
    """Unified entrypoint to load a dataset from either a directory of run pairs or a consolidated JSON file.

    Auto-detects format, validates loudly on bad paths or invalid formats, and returns list of runs in standard shape.
    """
    target = Path(path)

    if not target.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {target}")

    if target.is_dir():
        runs = find_all_runs(target)
        if not runs:
            raise ValueError(f"No valid run_metadata.json files found in directory tree: {target}")
        return runs

    if target.is_file():
        if target.suffix.lower() != ".json":
            raise ValueError(f"Unsupported dataset file format '{target.suffix}'. Expected a .json file or directory.")

        try:
            data = load_json(target)
        except ValueError as exc:
            raise ValueError(f"Failed to load dataset file {target}: {exc}") from exc

        if isinstance(data, dict) and "runs" in data:
            return load_consolidated(target, verbose=verbose)

        if isinstance(data, dict) and "metadata" in data and validate_metadata(data["metadata"]):
            return [{
                "meta_path": str(target),
                "traj_path": str(target),
                "metadata": data["metadata"],
                "trajectory": data.get("trajectory"),
            }]

        raise ValueError(
            f"File {target} is a JSON file but does not match consolidated dataset format ({{'runs': [...]}}) or single run schema."
        )

    raise ValueError(f"Invalid path type for dataset: {target}")
