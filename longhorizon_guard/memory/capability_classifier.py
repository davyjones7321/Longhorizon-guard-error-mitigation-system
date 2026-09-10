"""Classify real tool names and subgoal descriptions into canonical categories,
instead of requiring literal string matches against seed/library data."""

import re
from typing import Optional

_ACTION_CAPABILITY_ALIASES = {
    "shell_execution": [
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
    ],
}


def classify_action_capability(tool_name: str) -> Optional[str]:
    """Classify tool name to canonical capability category, or None if unknown."""
    t = (tool_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    for capability, aliases in _ACTION_CAPABILITY_ALIASES.items():
        if t in aliases:
            return capability
    return None


_SUBGOAL_CATEGORY_KEYWORDS = {
    "build_project": [r"\bbuild\b", r"\bcompile\b", r"\bpackage\b", r"\bartifact"],
    "run_tests": [r"\btest", r"\bpytest\b", r"\bverify\b", r"\bvalidate\b", r"\bqa\b"],
    "deploy_service": [r"\bdeploy", r"\brelease\b", r"\bpublish\b", r"\bproduction\b", r"\brollout\b"],
    "backup_database": [r"\bbackup\b", r"\bsnapshot\b"],
    "migrate_database": [r"\bmigrat"],
}


def classify_subgoal_category(description: str) -> Optional[str]:
    """Classify subgoal description to canonical subgoal category ID, or None if unknown."""
    d = (description or "").lower()
    for canonical_id, patterns in _SUBGOAL_CATEGORY_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, d):
                return canonical_id
    return None
