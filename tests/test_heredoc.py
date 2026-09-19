"""Unit tests for heredoc and here-string span detection and masking.

Tests cover:
- bash_heredoc_spans: quoted vs unquoted delimiters, tab-indented <<-, unterminated heredoc
- powershell_herestring_spans: literal @'...'@ and expandable @"..."@, unterminated here-string
- mask_spans: whitespace replacement preserving newlines, None handling
- error message after heredoc/here-string in the same command (must still be preserved/caught)
- GuardInterface.on_step integration: heredoc false positive suppression while preserving real errors
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from longhorizon_guard.heredoc import (
    bash_heredoc_spans,
    powershell_herestring_spans,
    mask_spans,
    mask_command_heredocs,
)
from longhorizon_guard.interface import GuardInterface


# =========================================================================
# Bash Heredoc Unit Tests
# =========================================================================

def test_bash_heredoc_unquoted_delimiter():
    """Unquoted delimiter <<EOF must be detected with exact terminator matching."""
    cmd = "cat << EOF > test.py\nline 1\nline 2\nEOF\n"
    spans = bash_heredoc_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    start, end = spans[0]
    # Body is "line 1\nline 2\n"
    assert cmd[start:end] == "line 1\nline 2\n"
    masked = mask_spans(cmd, spans)
    assert "line 1" not in masked
    assert "line 2" not in masked
    assert masked.startswith("cat << EOF > test.py\n")
    assert masked.endswith("EOF\n")


def test_bash_heredoc_single_quoted_delimiter():
    """Single-quoted delimiter <<'EOF' must be detected."""
    cmd = "cat << 'EOF' > test.py\nprint('syntax error')\nEOF\n"
    spans = bash_heredoc_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    start, end = spans[0]
    assert cmd[start:end] == "print('syntax error')\n"
    masked = mask_spans(cmd, spans)
    assert "syntax error" not in masked
    assert "\n" in masked


def test_bash_heredoc_double_quoted_delimiter():
    """Double-quoted delimiter <<"EOF" must be detected."""
    cmd = 'cat << "EOF" > test.py\nraise ConnectionError("server error")\nEOF\n'
    spans = bash_heredoc_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    start, end = spans[0]
    assert cmd[start:end] == 'raise ConnectionError("server error")\n'
    masked = mask_spans(cmd, spans)
    assert "server error" not in masked
    assert "ConnectionError" not in masked


def test_bash_heredoc_tab_indented_dash():
    """Tab-stripped <<- variant allows leading tabs on terminator line."""
    cmd = "cat <<-EOF > file.py\n\tline one\n\tline two\n\tEOF\n"
    spans = bash_heredoc_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    start, end = spans[0]
    assert cmd[start:end] == "\tline one\n\tline two\n"
    masked = mask_spans(cmd, spans)
    assert "line one" not in masked
    assert "line two" not in masked
    assert "\tEOF\n" in masked


def test_bash_heredoc_unterminated_masks_nothing():
    """An unterminated or unparsable heredoc must return None and mask nothing."""
    cmd = "cat << 'EOF' > test.py\nline 1\nline 2\n"
    spans = bash_heredoc_spans(cmd)
    assert spans is None
    masked = mask_spans(cmd, spans)
    assert masked == cmd


def test_bash_heredoc_error_after_heredoc_is_preserved():
    """A real error message appearing after a heredoc in the same command must be preserved."""
    cmd = (
        "cat << 'EOF' > test.py\n"
        "code_line = 1\n"
        "EOF\n"
        "echo 'api error: failed to submit'\n"
    )
    spans = bash_heredoc_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    masked = mask_spans(cmd, spans)
    # Body masked
    assert "code_line = 1" not in masked
    # Trailing command completely intact
    assert "echo 'api error: failed to submit'\n" in masked


# =========================================================================
# PowerShell Here-String Unit Tests
# =========================================================================

def test_powershell_herestring_literal():
    """PowerShell literal here-string @'...'@ must be detected and masked."""
    cmd = "Set-Content -Path script.py -Value @'\ndef run():\n    print('cannot find file')\n'@\n"
    spans = powershell_herestring_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    start, end = spans[0]
    assert cmd[start:end] == "def run():\n    print('cannot find file')\n"
    masked = mask_spans(cmd, spans)
    assert "cannot find file" not in masked
    assert masked.startswith("Set-Content -Path script.py -Value @'\n")
    assert "'@\n" in masked


def test_powershell_herestring_expandable():
    """PowerShell expandable here-string @"..."@ must be detected and masked."""
    cmd = '$script = @"\n$error_msg = "server error"\nWrite-Host $error_msg\n"@\n'
    spans = powershell_herestring_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    start, end = spans[0]
    assert cmd[start:end] == '$error_msg = "server error"\nWrite-Host $error_msg\n'
    masked = mask_spans(cmd, spans)
    assert "server error" not in masked


def test_powershell_herestring_unterminated_masks_nothing():
    """An unterminated PowerShell here-string must return None and mask nothing."""
    cmd = "Set-Content -Path script.py -Value @'\ndef run():\n    pass\n"
    spans = powershell_herestring_spans(cmd)
    assert spans is None
    masked = mask_spans(cmd, spans)
    assert masked == cmd


def test_powershell_herestring_error_after_herestring_is_preserved():
    """A real error message after a here-string in the same command must remain untouched."""
    cmd = (
        "Set-Content -Path script.py -Value @'\n"
        "class Worker: pass\n"
        "'@\n"
        "if ($LASTEXITCODE -ne 0) { Write-Error 'connection error' }\n"
    )
    spans = powershell_herestring_spans(cmd)
    assert spans is not None
    assert len(spans) == 1
    masked = mask_spans(cmd, spans)
    assert "class Worker: pass" not in masked
    assert "Write-Error 'connection error'" in masked


# =========================================================================
# mask_spans Utility Tests
# =========================================================================

def test_mask_spans_none_or_empty():
    """mask_spans returns original text when spans is None or empty."""
    text = "hello world"
    assert mask_spans(text, None) == text
    assert mask_spans(text, []) == text


def test_mask_spans_preserves_newlines_and_lengths():
    """mask_spans must preserve total string length and all newline positions."""
    text = "abc\ndef\nghi\n"
    # Mask "def\n" -> "   \n"
    spans = [(4, 8)]
    masked = mask_spans(text, spans)
    assert len(masked) == len(text)
    assert masked == "abc\n   \nghi\n"


# =========================================================================
# GuardInterface.on_step Integration Tests
# =========================================================================

def test_on_step_bash_heredoc_suppresses_false_positive():
    """Generated python code inside a bash heredoc must not trigger keyword rules."""
    guard = GuardInterface()
    step_record = {
        "step_index": 4,
        "reasoning": "Writing solution script to disk",
        "action_name": "Bash",
        "action_args": {
            "command": (
                "cat << 'EOF' > solution.py\n"
                "try:\n"
                "    parse_expression()\n"
                "except SyntaxError as e:\n"
                "    print('syntax error caught')\n"
                "EOF\n"
            )
        },
        "tool_response": "File written successfully",
    }
    result = guard.on_step(step_record, history=[])
    # Must NOT be flagged as tool_use_error (syntax error)
    assert result.get("flagged") is False
    assert result.get("category") is None


def test_on_step_powershell_herestring_suppresses_false_positive():
    """Generated python code inside a powershell here-string must not trigger keyword rules."""
    guard = GuardInterface()
    step_record = {
        "step_index": 5,
        "reasoning": "Writing solution script with PowerShell",
        "action_name": "powershell",
        "action_args": {
            "command": (
                "Set-Content -Path solution.py -Value @'\n"
                "def fetch():\n"
                "    # Handle api error and connection error\n"
                "    raise ConnectionError('server error')\n"
                "'@\n"
            )
        },
        "tool_response": "Success",
    }
    result = guard.on_step(step_record, history=[])
    # Must NOT be flagged as external_error (server error / connection error)
    assert result.get("flagged") is False
    assert result.get("category") is None


def test_on_step_genuine_error_in_action_is_still_flagged():
    """An actual error keyword in the action outside the heredoc must still be flagged."""
    guard = GuardInterface()
    step_record = {
        "step_index": 7,
        "reasoning": "Submitting task after writing code",
        "action_name": "Bash",
        "action_args": {
            "command": (
                "cat << 'EOF' > solution.py\n"
                "print('clean code')\n"
                "EOF\n"
                "echo 'server error: connection refused'\n"
            )
        },
        "tool_response": "server error: connection refused",
    }
    result = guard.on_step(step_record, history=[])
    # Genuinely failing step with external error must still be caught
    assert result.get("flagged") is True
    assert result.get("category") == "external_error"
