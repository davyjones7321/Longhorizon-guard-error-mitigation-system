"""
Unit tests for longhorizon_guard.storage.reader validating dataset loading and SCHEMA.md contract.
"""

import json
from pathlib import Path
import pytest
from longhorizon_guard.storage.reader import (
    find_all_runs,
    load_consolidated,
    load_dataset,
    load_run,
    validate_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_OUTPUTS_PATH = REPO_ROOT / "findings" / "all_clean_outputs.json"


def test_reader_schema_contract_on_real_clean_outputs():
    """Call load_dataset() directly on findings/all_clean_outputs.json to exercise auto-detection and assert SCHEMA.md fields."""
    assert CLEAN_OUTPUTS_PATH.exists(), f"Clean outputs dataset not found at {CLEAN_OUTPUTS_PATH}"

    runs = load_dataset(CLEAN_OUTPUTS_PATH)
    assert isinstance(runs, list)
    assert len(runs) > 0

    sample_run = runs[0]

    # Assert standard shape keys
    assert set(sample_run.keys()) == {"meta_path", "traj_path", "metadata", "trajectory"}

    meta = sample_run["metadata"]
    traj = sample_run["trajectory"]

    # Assert SCHEMA.md Required Metadata Fields
    required_meta_fields = ["run_id", "task_id", "trial_number", "horizon_level", "total_steps_taken", "final_status"]
    for field_name in required_meta_fields:
        assert field_name in meta, f"Metadata missing required field '{field_name}'"
        assert meta[field_name] is not None, f"Metadata required field '{field_name}' is None"

    # Assert type checks on metadata
    assert isinstance(meta["run_id"], str)
    assert isinstance(meta["task_id"], str)
    assert isinstance(meta["trial_number"], int)
    assert isinstance(meta["horizon_level"], int)
    assert isinstance(meta["total_steps_taken"], int)
    assert meta["final_status"] in ["success", "fail", "timeout", "error"]

    # Assert SCHEMA.md Required Trajectory Fields
    assert traj is not None
    assert "steps" in traj
    assert isinstance(traj["steps"], list)

    if len(traj["steps"]) > 0:
        step = traj["steps"][0]
        required_step_fields = ["step_index", "reasoning", "action_name", "action_args"]
        for field_name in required_step_fields:
            assert field_name in step, f"Step record missing required field '{field_name}'"

        assert isinstance(step["step_index"], int)
        assert isinstance(step["reasoning"], str)
        assert isinstance(step["action_name"], str)
        assert isinstance(step["action_args"], dict)


def test_load_dataset_directory_vs_consolidated_shape_parity(tmp_path):
    """Assert load_dataset() on a directory tree vs consolidated file returns exact same shape."""
    sample_meta = {
        "run_id": "test-uuid-1234",
        "task_id": "test_task",
        "trial_number": 1,
        "horizon_level": 2,
        "total_steps_taken": 1,
        "final_status": "success",
    }
    sample_traj = {
        "steps": [
            {
                "step_index": 0,
                "reasoning": "Test reasoning",
                "action_name": "done",
                "action_args": {"answer": "42"},
            }
        ]
    }

    # Setup directory-pair format in tmp_path
    trial_dir = tmp_path / "test_task" / "trial_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "run_metadata.json").write_text(json.dumps(sample_meta), encoding="utf-8")
    (trial_dir / "trajectory.json").write_text(json.dumps(sample_traj), encoding="utf-8")

    dir_runs = load_dataset(tmp_path)
    file_runs = load_dataset(CLEAN_OUTPUTS_PATH)

    assert isinstance(dir_runs, list) and len(dir_runs) == 1
    assert isinstance(file_runs, list) and len(file_runs) > 0

    dir_item = dir_runs[0]
    file_item = file_runs[0]

    # Shape Parity Assertion 1: Top-level keys must match exactly
    assert set(dir_item.keys()) == set(file_item.keys()) == {"meta_path", "traj_path", "metadata", "trajectory"}

    # Shape Parity Assertion 2: Metadata keys structure
    assert set(sample_meta.keys()).issubset(set(dir_item["metadata"].keys()))
    assert set(sample_meta.keys()).issubset(set(file_item["metadata"].keys()))

    # Shape Parity Assertion 3: Trajectory step keys structure
    assert "steps" in dir_item["trajectory"] and "steps" in file_item["trajectory"]
    assert set(sample_traj["steps"][0].keys()) == set(dir_item["trajectory"]["steps"][0].keys())
    assert set(sample_traj["steps"][0].keys()).issubset(set(file_item["trajectory"]["steps"][0].keys()))


def test_load_dataset_invalid_paths_and_formats(tmp_path):
    """Assert load_dataset raises clear errors for non-existent path, bad JSON, or invalid format."""
    # 1. Non-existent path
    with pytest.raises(FileNotFoundError, match="Dataset path does not exist"):
        load_dataset(tmp_path / "non_existent_dir")

    # 2. Unsupported file format
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported dataset file format"):
        load_dataset(txt_file)

    # 3. Invalid JSON file
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad_json: ", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load dataset file"):
        load_dataset(bad_json)

    # 4. JSON file with unknown structure
    unknown_json = tmp_path / "unknown.json"
    unknown_json.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match consolidated dataset format"):
        load_dataset(unknown_json)

    # 5. Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No valid run_metadata.json files found"):
        load_dataset(empty_dir)
