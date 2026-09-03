"""Tests for package data bundling and environment-independent pattern loading."""

from pathlib import Path
import pytest
from longhorizon_guard import GuardInterface
from longhorizon_guard.pattern_library.store import load_patterns, BUNDLED_DATA_DIR


def test_bundled_data_files_exist():
    """Verify bundled data directory contains pattern_library.json and idf_table.json."""
    assert BUNDLED_DATA_DIR.exists()
    pattern_file = BUNDLED_DATA_DIR / "pattern_library.json"
    idf_file = BUNDLED_DATA_DIR / "idf_table.json"
    assert pattern_file.exists(), f"Missing {pattern_file}"
    assert idf_file.exists(), f"Missing {idf_file}"


def test_load_patterns_bundled_default():
    """load_patterns should load all 11 mined patterns from the bundled data directory."""
    patterns = load_patterns()
    assert len(patterns) == 11
    categories = {p.category for p in patterns}
    assert "planning_error" in categories
    assert "reflection_error" in categories
    assert "memory_error" in categories


def test_guard_interface_loads_in_isolated_directory(tmp_path, monkeypatch):
    """GuardInterface must load all 11 patterns and broad-corpus IDF when CWD has no findings folder."""
    monkeypatch.chdir(tmp_path)
    assert not (Path.cwd() / "findings").exists()

    guard = GuardInterface()
    assert guard._matcher is not None
    assert guard._matcher._loaded is True
    assert len(guard._matcher._patterns) == 11
    assert len(guard._matcher._idf) > 0

    res = guard.on_plan_proposed(
        task_description="Blue cotton socks",
        proposed_plan="1. Search for socks",
    )
    assert "approved" in res
