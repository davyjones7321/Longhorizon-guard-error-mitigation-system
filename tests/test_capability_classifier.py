"""Unit tests for longhorizon_guard.memory.capability_classifier."""

import pytest

from longhorizon_guard.memory.capability_classifier import (
    classify_action_capability,
    classify_subgoal_category,
)


class TestActionCapabilityClassifier:
    """Test classification of real tool names into canonical capabilities."""

    @pytest.mark.parametrize(
        "tool_name",
        [
            "bash",
            "shell",
            "exec",
            "execute",
            "exec_command",
            "run_command",
            "runcommand",
            "terminal",
            "run_shell_command",
            "shellexecute",
            "cmd",
            "run_terminal_cmd",
            "execute_command",
            "sh",
            "zsh",
            "powershell",
            # Variations with case, spaces, hyphens
            "SHELL",
            "Run-Command",
            "  bash  ",
            "run_command",
            "EXEC-COMMAND",
        ],
    )
    def test_shell_execution_aliases(self, tool_name: str):
        assert classify_action_capability(tool_name) == "shell_execution"

    @pytest.mark.parametrize(
        "non_matching_tool",
        [
            "edit_file",
            "view_file",
            "git_commit",
            "read_url",
            "custom_linter",
            "web_search",
            "",
            None,
        ],
    )
    def test_non_matching_tools_return_none(self, non_matching_tool):
        assert classify_action_capability(non_matching_tool) is None


class TestSubgoalCategoryClassifier:
    """Test classification of natural language subgoal phrasing into canonical IDs."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Build the service",
            "Build source artifacts",
            "Compile TypeScript backend",
            "Package application release bundle",
            "Generate build artifact",
        ],
    )
    def test_build_project_category(self, phrase: str):
        assert classify_subgoal_category(phrase) == "build_project"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Run the test suite",
            "Execute automated test suite",
            "Run pytest unit tests",
            "Verify system endpoints",
            "Validate integration tests",
            "Run QA test suite",
        ],
    )
    def test_run_tests_category(self, phrase: str):
        assert classify_subgoal_category(phrase) == "run_tests"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Deploy service to production",
            "Deploy to production environment",
            "Release version 2.0.0",
            "Publish docker container to registry",
            "Execute production rollout",
        ],
    )
    def test_deploy_service_category(self, phrase: str):
        assert classify_subgoal_category(phrase) == "deploy_service"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Backup primary database",
            "Create pre-migration database snapshot",
            "Perform database backup",
        ],
    )
    def test_backup_database_category(self, phrase: str):
        assert classify_subgoal_category(phrase) == "backup_database"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Execute database schema migration",
            "Migrate database tables",
            "Run alembic migrations",
        ],
    )
    def test_migrate_database_category(self, phrase: str):
        assert classify_subgoal_category(phrase) == "migrate_database"

    @pytest.mark.parametrize(
        "unmatched_phrase",
        [
            "Write user documentation in markdown",
            "Review pull request comments",
            "Format code indentation and linting",
            "Refactor helper functions",
            "",
            None,
        ],
    )
    def test_unmatched_subgoals_return_none(self, unmatched_phrase):
        assert classify_subgoal_category(unmatched_phrase) is None
