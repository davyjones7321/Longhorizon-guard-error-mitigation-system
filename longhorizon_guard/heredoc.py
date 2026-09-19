"""Heredoc and here-string span detection and masking for shell commands.

Adapts the conservative masking design from NousResearch/hermes-agent's tools/shell_heredoc.py.
Detects:
- Bash heredocs: <<EOF, <<'EOF', <<"EOF", <<-EOF (tab-stripped variant)
- PowerShell here-strings: @'...'@ (literal) and @"..."@ (expandable)

Contracts:
- bash_heredoc_spans(command: str) -> list[tuple[int, int]] | None
- powershell_herestring_spans(command: str) -> list[tuple[int, int]] | None
- mask_spans(text: str, spans: list[tuple[int, int]] | None) -> str
"""

import re
from typing import Any, List, Optional, Tuple

# Pattern matching any newline sequence:
# Real newlines (\r\n, \n) and escaped newlines (\\r\\n, \\n) for stringified commands/dicts
_LINE_BREAK_PATTERN = re.compile(r"(\r\n|\n|\\r\\n|\\n)")

# Bash heredoc opener pattern:
# Looks for << or <<- followed by optional whitespace and a delimiter word.
# Negative lookbehind (?<!<) and lookahead (?!<) ensure <<< (here-string) is excluded.
_BASH_HEREDOC_OPENER_PATTERN = re.compile(
    r"(?<!<)<<(-?)[ \t]*(?:'([^']+)'|\"([^\"]+)\"|\\([A-Za-z0-9_]+)|([A-Za-z_][A-Za-z0-9_.-]*))"
)

# PowerShell here-string opener pattern:
# Matches @' or @" at the end of a line (optional whitespace before newline)
_PS_HERESTRING_OPENER_PATTERN = re.compile(r"@(['\"])[ \t]*$")


def _split_lines_with_offsets(
    text: str,
) -> List[Tuple[str, int, int, int, int]]:
    """Split text into lines while tracking character offsets.

    Returns a list of tuples:
        (line_content, line_start, line_end, break_start, break_end)
    Where:
        line_content = text[line_start:line_end]
        line_break = text[break_start:break_end] (empty if last line has no trailing break)
        next line starts at break_end
    """
    lines: List[Tuple[str, int, int, int, int]] = []
    if not text:
        return lines

    cur_start = 0
    for match in _LINE_BREAK_PATTERN.finditer(text):
        break_start = match.start()
        break_end = match.end()
        line_content = text[cur_start:break_start]
        lines.append((line_content, cur_start, break_start, break_start, break_end))
        cur_start = break_end

    if cur_start <= len(text):
        line_content = text[cur_start:]
        lines.append((line_content, cur_start, len(text), len(text), len(text)))

    return lines


def _is_inside_quotes(line_prefix: str) -> bool:
    """Check if the position in line_prefix is enclosed within unclosed quotes."""
    in_single = False
    in_double = False
    i = 0
    n = len(line_prefix)
    while i < n:
        c = line_prefix[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        i += 1
    return in_single or in_double


def bash_heredoc_spans(command: str) -> Optional[List[Tuple[int, int]]]:
    """Detect Bash heredoc spans: <<EOF, <<'EOF', <<"EOF", <<-EOF.

    Returns:
        A list of (body_start, body_end) character offsets in the original string,
        or None if the terminator cannot be confidently located or syntax is ambiguous.
        Conservative: only fires on confident, well-terminated cases. Anything ambiguous
        returns None.
    """
    if not command or not isinstance(command, str) or not command.strip():
        return None

    if "<<" not in command:
        return None

    lines = _split_lines_with_offsets(command)
    if not lines:
        return None

    spans: List[Tuple[int, int]] = []
    i = 0
    num_lines = len(lines)

    while i < num_lines:
        line_content, line_start, line_end, break_start, break_end = lines[i]

        all_openers = list(_BASH_HEREDOC_OPENER_PATTERN.finditer(line_content))
        if not all_openers:
            i += 1
            continue

        # Filter openers that fall inside quotes
        valid_openers = []
        for match in all_openers:
            prefix = line_content[: match.start()]
            if not _is_inside_quotes(prefix):
                valid_openers.append(match)

        if len(valid_openers) != 1:
            # Ambiguous: multiple heredocs on a single line or all inside quotes
            if len(valid_openers) > 1:
                return None
            i += 1
            continue

        match = valid_openers[0]
        dash_group = match.group(1)
        single_quoted = match.group(2)
        double_quoted = match.group(3)
        escaped_delim = match.group(4)
        bare_delim = match.group(5)

        delimiter = single_quoted or double_quoted or escaped_delim or bare_delim
        if not delimiter:
            return None

        is_tab_stripped = bool(dash_group == "-")

        # The body begins immediately after the line break of the header line
        if break_start == break_end:
            # Header line was not terminated by a newline
            return None

        body_start = break_end

        # Locate the exact terminator line
        terminator_found = False
        j = i + 1
        while j < num_lines:
            cand_content, cand_start, cand_end, _, _ = lines[j]
            clean_cand = cand_content.rstrip("\r")

            if is_tab_stripped:
                # Terminator line allows leading tabs: ^\t*DELIMITER$
                stripped = clean_cand.lstrip("\t")
                if stripped == delimiter:
                    terminator_found = True
                    body_end = cand_start
                    spans.append((body_start, body_end))
                    i = j
                    break
            else:
                # Standard heredoc: exact match with no leading or trailing whitespace
                if clean_cand == delimiter:
                    terminator_found = True
                    body_end = cand_start
                    spans.append((body_start, body_end))
                    i = j
                    break
            j += 1

        if not terminator_found:
            # Terminator could not be located -> conservative None
            return None

        i += 1

    return spans if spans else None


def powershell_herestring_spans(command: str) -> Optional[List[Tuple[int, int]]]:
    """Detect PowerShell here-string spans: @'...'@ (literal) and @"..."@ (expandable).

    Returns:
        A list of (body_start, body_end) character offsets in the original string,
        or None if the terminator cannot be confidently located or syntax is ambiguous.
        Conservative: only fires on confident, well-terminated cases. Anything ambiguous
        returns None.
    """
    if not command or not isinstance(command, str) or not command.strip():
        return None

    if "@'" not in command and '@"' not in command:
        return None

    lines = _split_lines_with_offsets(command)
    if not lines:
        return None

    spans: List[Tuple[int, int]] = []
    i = 0
    num_lines = len(lines)

    while i < num_lines:
        line_content, line_start, line_end, break_start, break_end = lines[i]
        clean_content = line_content.rstrip("\r")

        match = _PS_HERESTRING_OPENER_PATTERN.search(clean_content)
        if not match:
            i += 1
            continue

        opener_pos = match.start()
        prefix = clean_content[:opener_pos]
        if _is_inside_quotes(prefix):
            i += 1
            continue

        quote_char = match.group(1)  # ' or "
        expected_closing = f"{quote_char}@"

        if break_start == break_end:
            # Header line not followed by newline -> unterminated
            return None

        body_start = break_end

        terminator_found = False
        j = i + 1
        while j < num_lines:
            cand_content, cand_start, cand_end, _, _ = lines[j]
            clean_cand = cand_content.rstrip("\r")

            stripped_cand = clean_cand.lstrip(" \t")
            if stripped_cand.startswith(expected_closing):
                terminator_found = True
                body_end = cand_start
                spans.append((body_start, body_end))
                i = j
                break
            j += 1

        if not terminator_found:
            return None

        i += 1

    return spans if spans else None


def mask_spans(text: str, spans: Optional[List[Tuple[int, int]]]) -> str:
    """Replace each span's content with whitespace, preserving newline structure.

    Args:
        text: The original string.
        spans: A list of (start, end) index tuples, or None.

    Returns:
        The text with characters inside spans replaced by spaces, preserving
        newlines. If spans is None or empty, returns text unchanged.
    """
    if not text or not spans:
        return text

    valid_spans: List[Tuple[int, int]] = []
    text_len = len(text)
    for s, e in spans:
        if s is None or e is None:
            continue
        start = max(0, min(int(s), text_len))
        end = max(0, min(int(e), text_len))
        if start < end:
            valid_spans.append((start, end))

    if not valid_spans:
        return text

    valid_spans.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = [valid_spans[0]]
    for cur_start, cur_end in valid_spans[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    chars = list(text)
    for start, end in merged:
        i = start
        while i < end:
            # Preserve escaped newlines in stringified text: \\r\\n or \\n
            if i + 1 < end and chars[i] == "\\" and chars[i + 1] == "n":
                i += 2
                continue
            if i + 1 < end and chars[i] == "\\" and chars[i + 1] == "r":
                i += 2
                continue
            # Preserve real newlines
            if chars[i] in ("\n", "\r"):
                i += 1
                continue
            chars[i] = " "
            i += 1

    return "".join(chars)


def mask_command_heredocs(action_args: Any) -> str:
    """Mask heredoc and here-string bodies in action_args before pattern matching.

    Supports action_args as a str or as a dict (e.g. {'command': '...'}).
    Returns the masked string representation for combined_text matching.
    """
    if not action_args:
        return ""

    if isinstance(action_args, dict):
        masked_dict = dict(action_args)
        masked_any = False
        for k in ("command", "cmd", "script", "input"):
            v = masked_dict.get(k)
            if isinstance(v, str):
                b_spans = bash_heredoc_spans(v)
                p_spans = powershell_herestring_spans(v)
                all_spans = []
                if b_spans:
                    all_spans.extend(b_spans)
                if p_spans:
                    all_spans.extend(p_spans)
                if all_spans:
                    masked_dict[k] = mask_spans(v, all_spans)
                    masked_any = True

        action_str = str(masked_dict)
        if masked_any:
            return action_str

        b_spans = bash_heredoc_spans(action_str)
        p_spans = powershell_herestring_spans(action_str)
        all_spans = []
        if b_spans:
            all_spans.extend(b_spans)
        if p_spans:
            all_spans.extend(p_spans)
        return mask_spans(action_str, all_spans) if all_spans else action_str

    action_str = str(action_args)
    b_spans = bash_heredoc_spans(action_str)
    p_spans = powershell_herestring_spans(action_str)
    all_spans = []
    if b_spans:
        all_spans.extend(b_spans)
    if p_spans:
        all_spans.extend(p_spans)
    return mask_spans(action_str, all_spans) if all_spans else action_str
