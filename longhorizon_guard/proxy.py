"""HTTP API Proxy Server for LongHorizon Guard.

Transparently proxies chat completion requests between coding assistants
(OpenCode, Claude Code, Cursor, Aider, Codex) and upstream LLM providers,
intercepting tool calls and plain-text code edits, evaluating trajectories in
real time, with automatic session logging to disk.

Supported Observation Modes:
1. Native Function Calling (`tool_calls` mode):
   Used by agents like OpenCode, Cursor, and Claude Code that emit structured
   OpenAI `tool_calls` and receive `role: "tool"` execution responses.
2. Plain-Text Edits (`text_edits` mode):
   Used by agents like Aider that default to git-style SEARCH/REPLACE diff
   blocks, unified diffs, or whole-file completions directly in `content`.
"""

import argparse
import atexit
import datetime
import gzip
import http.server
import json
import logging
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import zlib
from typing import Any, Callable, Dict, List, Optional, Set

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.storage.adapters.antigravity_session import sanitize_data

logger = logging.getLogger("longhorizon_guard.proxy")
DEFAULT_LOG_DIR = os.path.join("findings", "proxy_sessions")


def _parse_args(args_val: Any) -> Dict[str, Any]:
    """Safely parse tool arguments into a dictionary."""
    if isinstance(args_val, dict):
        return args_val
    if isinstance(args_val, str):
        try:
            parsed = json.loads(args_val)
            if isinstance(parsed, dict):
                return parsed
            return {"args": parsed}
        except Exception:
            return {"raw": args_val}
    return {"raw": str(args_val)} if args_val is not None else {}


def _decompress_response_body(body: bytes, content_encoding: Optional[str] = None) -> bytes:
    """Decompress HTTP response body based on Content-Encoding or magic bytes.

    Supports gzip, deflate (zlib format or raw deflate), and brotli (if installed).
    Returns the original body if uncompressed or if decompression fails.
    """
    if not body:
        return body

    encoding = (content_encoding or "").lower().strip()

    # 1. Gzip (Content-Encoding: gzip/x-gzip or gzip magic bytes 1f 8b)
    if encoding in ("gzip", "x-gzip") or body.startswith(b"\x1f\x8b"):
        try:
            return gzip.decompress(body)
        except Exception as exc:
            logger.debug("Failed gzip decompression: %s", exc)

    # 2. Deflate (zlib format or raw deflate)
    if encoding == "deflate":
        try:
            return zlib.decompress(body)
        except zlib.error:
            try:
                return zlib.decompress(body, -zlib.MAX_WBITS)
            except Exception as exc:
                logger.debug("Failed deflate decompression: %s", exc)

    # 3. Brotli (Content-Encoding: br)
    if encoding == "br":
        try:
            import brotli
            return brotli.decompress(body)
        except ImportError:
            logger.debug("brotli package not installed; skipping 'br' decompression")
        except Exception as exc:
            logger.debug("Failed brotli decompression: %s", exc)

    return body


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text string from message content (str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content or "")


def _clean_path(path_str: str) -> str:
    """Strip formatting wrappers (backticks, markdown headers, labels) from a file path."""
    path_str = path_str.strip()
    path_str = re.sub(r"^[#*`\-:\s]+|[#*`\-:\s]+$", "", path_str)
    path_str = re.sub(r"^(?:file|filename|filepath):\s*", "", path_str, flags=re.IGNORECASE)
    return path_str.strip("`'\"* ")


COMMON_IGNORE_EXTENSIONS: Set[str] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "g", "e", "i"}
COMMON_IGNORE_WORDS: Set[str] = {"e.g.", "i.e.", "etc.", "vs.", "al."}

FILENAME_RE = re.compile(
    r"""(?P<prefix>[`'"]?)"""
    r"""(?P<path>(?:[a-zA-Z0-9_.\-]+[/\\])*[a-zA-Z0-9_.\-]+\.[a-zA-Z0-9_]+)"""
    r"""(?P<quote>[`'"]?)"""
    r"""(?P<suffix>[:`'\s,\)]|$)"""
)


def _extract_candidates_from_line(line: str) -> List[tuple[str, int]]:
    """Extract candidate filenames with confidence scores from a single line of text."""
    candidates: List[tuple[str, int]] = []
    line_clean = line.strip()
    if not line_clean:
        return candidates

    # Skip lines that are pure code statement declarations
    if re.match(r"^(?:import |from |class |def |return |if |for |while )", line_clean):
        return candidates

    # If the entire line is solely a file path (e.g. "app.py" or "### `app.py`")
    direct_cleaned = _clean_path(line_clean)
    if re.match(r"^(?:[a-zA-Z0-9_.\-]+[/\\])*[a-zA-Z0-9_.\-]+\.[a-zA-Z0-9_]+$", direct_cleaned):
        ext = direct_cleaned.rsplit(".", 1)[-1].lower()
        if not ext.isdigit() and ext not in COMMON_IGNORE_EXTENSIONS:
            candidates.append((direct_cleaned, 100))
            return candidates

    # Search the entire line for all filename-shaped substrings
    for match in FILENAME_RE.finditer(line):
        raw_path = match.group("path")
        cleaned_path = _clean_path(raw_path)
        if not cleaned_path or "." not in cleaned_path:
            continue
        if cleaned_path.lower() in COMMON_IGNORE_WORDS:
            continue
        ext = cleaned_path.rsplit(".", 1)[-1].lower()
        if ext.isdigit() or ext in COMMON_IGNORE_EXTENSIONS:
            continue

        score = 10
        quote = match.group("quote") or match.group("prefix")
        suffix = match.group("suffix")
        end_pos = match.end()

        # Prefer candidates immediately followed by a colon or backtick
        if ":" in suffix or quote == "`":
            score += 25
        # Prefer candidates at the end of the line
        if end_pos >= len(line.rstrip()):
            score += 20
        # Preceded by action indicators like "file", "filename", "modify", "update"
        pre_text = line[:match.start()].lower()
        if re.search(r"\b(?:file|filename|filepath|in|modifying|modify|updating|update|editing|edit|patching|patch)\s*$", pre_text):
            score += 15

        candidates.append((cleaned_path, score))

    return candidates


def _extract_filename_before(text: str, start_pos: int, prev_end_pos: int = 0) -> str:
    """Find the most likely file path in lines preceding start_pos (after prev_end_pos).

    Scans entire lines for filename-shaped tokens (not just the last word),
    scoring candidate tokens higher if followed by a colon, wrapped in backticks,
    located at end-of-line, or explicitly preceded by words like 'file:'.
    """
    search_window = text[prev_end_pos:start_pos].strip()
    lines = [l.strip() for l in search_window.split("\n") if l.strip()]
    if not lines:
        return ""

    best_candidate = ""
    best_score = -1

    for idx, line in enumerate(reversed(lines[-10:])):
        recency_weight = max(0, 20 - idx * 2)
        candidates = _extract_candidates_from_line(line)
        for path, score in candidates:
            total_score = score + recency_weight
            if total_score > best_score:
                best_score = total_score
                best_candidate = path

    return best_candidate


def _parse_text_edits(content: str) -> List[Dict[str, Any]]:
    """Parse plain-text file edits produced by coding agents like Aider.

    Supports:
    1. Aider diff format (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE)
    2. Unified diff format (--- a/... +++ b/...)
    3. Whole-file replacement format (filename followed by fenced code block)
    """
    edits: List[Dict[str, Any]] = []
    if not content or not isinstance(content, str):
        return edits

    # 1. Search / Replace blocks (Aider diff format)
    diff_pattern = re.compile(
        r"<{5,9}\s*SEARCH\s*\n(.*?)\n?={5,9}\s*\n(.*?)\n?>{5,9}\s*REPLACE",
        re.DOTALL,
    )

    last_file = ""
    last_end = 0
    for match in diff_pattern.finditer(content):
        search_chunk = match.group(1)
        replace_chunk = match.group(2)
        start_pos = match.start()

        file_candidate = _extract_filename_before(content, start_pos, last_end)
        if file_candidate:
            last_file = file_candidate

        target_file = last_file or "unknown_file"
        last_end = match.end()
        edits.append({
            "path": target_file,
            "format": "diff",
            "details": {
                "search": search_chunk,
                "replace": replace_chunk,
            },
        })

    if edits:
        return edits

    # 2. Unified diff format (udiff)
    udiff_pattern = re.compile(
        r"(?:^|\n)(?:---|\+\+\+)\s+(?:[ab]/)?([^\s\n]+)\s*\n(?:\+\+\+|---)\s+(?:[ab]/)?([^\s\n]+)\s*\n(@@\s*-\d+.*?\n[\s\S]*?)(?=(?:\n---|\Z))",
        re.MULTILINE,
    )
    for match in udiff_pattern.finditer(content):
        file_path = match.group(2) or match.group(1)
        patch = match.group(3)
        edits.append({
            "path": _clean_path(file_path),
            "format": "udiff",
            "details": {"patch": patch},
        })

    if edits:
        return edits

    # 3. Whole file fenced code blocks preceded by filename
    whole_pattern = re.compile(
        r"(?:^|\n)(?:(?:#+|\*+)?\s*([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9_]+)\s*(?:#+|\*+)?\s*\n)?"
        r"```[a-zA-Z0-9_]*\s*\n"
        r"(?:(?:#|//|<!--)\s*(?:file(?:name)?:?\s*)?([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9_]+)\s*\n)?"
        r"([\s\S]*?)\n```",
        re.MULTILINE,
    )
    for match in whole_pattern.finditer(content):
        file_path = match.group(1) or match.group(2)
        code_content = match.group(3)
        if file_path:
            cleaned_file = _clean_path(file_path)
            if "." in cleaned_file and not any(kw in cleaned_file.lower() for kw in ["example", "snippet"]):
                edits.append({
                    "path": cleaned_file,
                    "format": "whole",
                    "details": {"content": code_content},
                })

    return edits


def _is_placeholder_task(text: str) -> bool:
    """Detect whether a string is harness boilerplate or an initialization handshake rather than a task."""
    if not text:
        return True
    cleaned = text.strip().lower()
    placeholder_phrases = (
        "not sharing any files",
        "no files shared",
        "no files have been",
        "can edit yet",
        "ready for your request",
        "ready for instructions",
        "system prompt test",
        "ping",
        "handshake",
    )
    if any(phrase in cleaned for phrase in placeholder_phrases):
        return True
    if cleaned in ("hi", "hello", "hey", "ping", "test", "ok", "ready"):
        return True
    return False


def _is_substantive_task(text: str) -> bool:
    """Return True if text represents a genuine task rather than boilerplate or greeting."""
    if _is_placeholder_task(text):
        return False
    return len(text.strip()) >= 8


def _extract_task_description(messages: List[Dict[str, Any]]) -> str:
    """Structurally extract the true user task description from conversation messages.

    Coding assistants like Aider frequently include few-shot demonstration examples
    in the initial prompt (e.g. system instructions followed by simulated turns like
    'Change the greeting to be more casual') or send an early placeholder handshake
    ('I am not sharing any files that you can edit yet.').

    This function:
    1. Looks through user messages in reverse to find the latest genuine, substantive task prompt.
    2. Skips test execution outputs or command responses in user messages.
    3. Skips placeholder handshake messages if a substantive user task is available.
    4. Falls back to earlier user messages if no substantive task is found.
    5. Only as a last resort falls back to a system message that explicitly defines a task.
    """
    if not messages or not isinstance(messages, (list, tuple)):
        return ""

    user_candidates: List[str] = []
    system_candidates: List[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        raw_content = msg.get("content")
        text = _extract_text_from_content(raw_content).strip()
        if not text:
            continue

        if role == "user":
            user_candidates.append(text)
        elif role == "system":
            system_candidates.append(text)

    # If user messages exist, find the real task prompt
    if user_candidates:
        # First pass: search in reverse for a substantive prompt (non-placeholder, non-test output)
        for text in reversed(user_candidates):
            first_line = text.split("\n", 1)[0].lower()
            if any(first_line.startswith(prefix) for prefix in ("pytest", "unittest", "traceback", "exit code", "error:")):
                continue
            if _is_placeholder_task(text):
                continue
            return text

        # Second pass fallback: if all were placeholders or test outputs, return the latest user message
        return user_candidates[-1]

    # Fallback to system prompt only if no user messages exist
    for text in system_candidates:
        for line in text.splitlines():
            line_l = line.lower()
            if any(line_l.startswith(p) for p in ("# task:", "task:", "goal:", "objective:")):
                return line.split(":", 1)[1].strip()
        return text

    return ""


class SessionState:
    """Tracks trajectory state and logs records to disk for an active agent session."""

    def __init__(
        self,
        session_id: str,
        guard: GuardInterface,
        on_flag: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_dir: Optional[str] = None,
        idle_timeout: float = 30.0,
    ) -> None:
        self.session_id = session_id
        self.guard = guard
        self.on_flag = on_flag
        self.log_dir = log_dir
        self.idle_timeout = idle_timeout
        self.mode: str = "auto"  # "auto", "tool_calls", or "text_edits"
        self.history: List[Dict[str, Any]] = []
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._processed_tool_ids: Set[str] = set()
        self._step_counter: int = 0
        self._plan_proposed_called: bool = False
        self._task_description: str = ""
        self._last_summarized_step_count: int = -1
        self._finalized: bool = False
        self._last_summary: Optional[Dict[str, Any]] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._lock: threading.Lock = threading.Lock()

        # Set up log file
        self.log_file_path: Optional[str] = None
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_{timestamp}_{self.session_id}.jsonl"
                self.log_file_path = os.path.join(self.log_dir, filename)
                print(f"📝 [GUARD LOG] Recording session to: {self.log_file_path}", file=sys.stderr, flush=True)
            except Exception as exc:
                logger.warning("Could not initialize session log file: %s", exc)

    def _cancel_idle_timer(self) -> None:
        """Cancel any running idle finalization timer."""
        with self._lock:
            if self._idle_timer is not None:
                try:
                    self._idle_timer.cancel()
                except Exception:
                    pass
                self._idle_timer = None

    def _schedule_idle_timer(self) -> None:
        """Schedule or reset the idle timer to finalize session if no subsequent turns arrive."""
        with self._lock:
            if self._idle_timer is not None:
                try:
                    self._idle_timer.cancel()
                except Exception:
                    pass
                self._idle_timer = None
            if self.idle_timeout > 0 and not self._finalized and len(self.history) > 0:
                timer = threading.Timer(self.idle_timeout, self._on_idle_timeout)
                timer.daemon = True
                self._idle_timer = timer
                timer.start()

    def _on_idle_timeout(self) -> None:
        """Callback invoked when session reaches idle timeout without new requests."""
        try:
            self.finalize_session()
        except Exception as exc:
            logger.warning("Error finalizing session %s on idle timeout: %s", self.session_id, exc)

    def _write_log_entry(self, entry: Dict[str, Any]) -> None:
        """Sanitize and append an entry to the session JSONL file."""
        if not self.log_file_path:
            return
        try:
            cleaned_entry = sanitize_data(entry)
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cleaned_entry) + "\n")
        except Exception as exc:
            logger.warning("Failed to write to session log file %s: %s", self.log_file_path, exc)

    def process_messages_before_call(self, messages: List[Dict[str, Any]]) -> None:
        """Inspect request messages for initial plan and prior tool execution outputs."""
        self._cancel_idle_timer()
        if not messages or not isinstance(messages, (list, tuple)):
            return

        # 1. Plan detection with placeholder handling and late-arriving task support
        task_desc = _extract_task_description(messages)
        if task_desc:
            current_is_placeholder = _is_placeholder_task(self._task_description)
            new_is_substantive = _is_substantive_task(task_desc)

            # Trigger plan evaluation if first time or if upgrading from an initial placeholder handshake
            if not self._plan_proposed_called or (current_is_placeholder and new_is_substantive):
                self._task_description = task_desc
                if new_is_substantive:
                    self._plan_proposed_called = True

                plan_res = self.guard.on_plan_proposed(
                    task_description=task_desc,
                    proposed_plan="",
                    metadata={"source": "guard_proxy", "session_id": self.session_id},
                )

                self._write_log_entry({
                    "type": "plan",
                    "session_id": self.session_id,
                    "task_description": task_desc,
                    "proposed_plan": "",
                    "approved": plan_res.get("approved", True),
                    "flags": plan_res.get("flags", []),
                    "suggestions": plan_res.get("suggestions", []),
                    "n_subgoals": plan_res.get("n_subgoals_parsed", 0),
                    "timestamp": datetime.datetime.now().isoformat(),
                })

        # 2. Correlate tool responses from previous agent action
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role in ("tool", "function"):
                self.mode = "tool_calls"
                tool_call_id = msg.get("tool_call_id") or msg.get("name")
                if not tool_call_id or tool_call_id in self._processed_tool_ids:
                    continue

                tool_content = msg.get("content", "")
                pending = self._pending_tool_calls.pop(str(tool_call_id), None)
                if not pending and self._pending_tool_calls:
                    _, pending = self._pending_tool_calls.popitem()

                reasoning = pending.get("reasoning", "") if pending else ""
                action_name = pending.get("name", "tool") if pending else (msg.get("name") or "tool")
                action_args = pending.get("args", {}) if pending else {}

                step_record = {
                    "step_index": self._step_counter,
                    "reasoning": reasoning,
                    "action_name": action_name,
                    "action_args": action_args,
                    "tool_response": str(tool_content),
                }

                step_res = self.guard.on_step(
                    step_record=step_record,
                    history=self.history,
                    metadata={"source": "guard_proxy", "session_id": self.session_id, "mode": "tool_calls"},
                )

                self.history.append(step_record)
                self._processed_tool_ids.add(str(tool_call_id))
                self._step_counter += 1

                # Record step to disk
                self._write_log_entry({
                    "type": "step",
                    "session_id": self.session_id,
                    "step_index": step_record["step_index"],
                    "reasoning": step_record["reasoning"],
                    "action_name": step_record["action_name"],
                    "action_args": step_record["action_args"],
                    "tool_response": step_record["tool_response"],
                    "flagged": step_res.get("flagged", False),
                    "warning": step_res.get("warning"),
                    "category": step_res.get("category"),
                    "confidence": step_res.get("confidence"),
                    "subgoal_status": step_res.get("subgoal_status"),
                    "drift": step_res.get("drift_assessment"),
                    "reflector": step_res.get("reflection_result"),
                    "timestamp": datetime.datetime.now().isoformat(),
                })

                if step_res.get("flagged"):
                    if self.on_flag:
                        self.on_flag(step_res)
                    else:
                        warning = step_res.get("warning", "Potential error detected")
                        cat = step_res.get("category", "unknown")
                        conf = step_res.get("confidence", 0.0)
                        print(
                            f"\n\033[93m⚠️  [LONGHORIZON GUARD ALERT]\033[0m Step {step_record['step_index']} "
                            f"[{cat} conf={conf:.2f}]:\n    {warning}\n",
                            file=sys.stderr,
                            flush=True,
                        )

    def process_response_after_call(self, response_body: Dict[str, Any]) -> None:
        """Capture assistant thought and proposed tool calls or plain-text edits from response."""
        choices = response_body.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "") or ""
        content = _extract_text_from_content(raw_content)
        reasoning = message.get("reasoning_content") or content

        tool_calls = message.get("tool_calls", [])
        if tool_calls and isinstance(tool_calls, list):
            self.mode = "tool_calls"
            for tc in tool_calls:
                tc_id = tc.get("id") or str(id(tc))
                func = tc.get("function", {})
                fname = func.get("name", "tool")
                raw_args = func.get("arguments", {})
                parsed_args = _parse_args(raw_args)

                self._pending_tool_calls[str(tc_id)] = {
                    "reasoning": reasoning,
                    "name": fname,
                    "args": parsed_args,
                }
        else:
            # Plain-text edit observation path (for coding assistants like Aider using diff/whole formats)
            edits = _parse_text_edits(content)
            if edits:
                self.mode = "text_edits"
                for edit in edits:
                    step_record = {
                        "step_index": self._step_counter,
                        "reasoning": reasoning,
                        "action_name": "edit_file",
                        "action_args": {
                            "path": edit["path"],
                            "format": edit["format"],
                            **edit.get("details", {}),
                        },
                        "tool_response": f"Applied {edit['format']} edit to {edit['path']}",
                    }

                    step_res = self.guard.on_step(
                        step_record=step_record,
                        history=self.history,
                        metadata={
                            "source": "guard_proxy",
                            "session_id": self.session_id,
                            "mode": "text_edits",
                        },
                    )

                    self.history.append(step_record)
                    self._step_counter += 1

                    # Record step to disk
                    self._write_log_entry({
                        "type": "step",
                        "session_id": self.session_id,
                        "step_index": step_record["step_index"],
                        "reasoning": step_record["reasoning"],
                        "action_name": step_record["action_name"],
                        "action_args": step_record["action_args"],
                        "tool_response": step_record["tool_response"],
                        "flagged": step_res.get("flagged", False),
                        "warning": step_res.get("warning"),
                        "category": step_res.get("category"),
                        "confidence": step_res.get("confidence"),
                        "subgoal_status": step_res.get("subgoal_status"),
                        "drift": step_res.get("drift_assessment"),
                        "reflector": step_res.get("reflection_result"),
                        "timestamp": datetime.datetime.now().isoformat(),
                    })

                    if step_res.get("flagged"):
                        if self.on_flag:
                            self.on_flag(step_res)
                        else:
                            warning = step_res.get("warning", "Potential error detected")
                            cat = step_res.get("category", "unknown")
                            conf = step_res.get("confidence", 0.0)
                            print(
                                f"\n\033[93m⚠️  [LONGHORIZON GUARD ALERT]\033[0m Step {step_record['step_index']} "
                                f"[{cat} conf={conf:.2f}]:\n    {warning}\n",
                                file=sys.stderr,
                                flush=True,
                            )

        # Start or reset idle timer for background finalization if no further turns arrive
        self._schedule_idle_timer()

    def finalize_session(self) -> Dict[str, Any]:
        """Finalize the session, execute on_run_end, and log summary to disk."""
        self._cancel_idle_timer()
        with self._lock:
            if self._finalized:
                return self._last_summary or {}

            summary = self.guard.on_run_end(
                metadata={"session_id": self.session_id, "task": self._task_description},
                trajectory={"steps": self.history},
            )
            self._write_log_entry({
                "type": "summary",
                "session_id": self.session_id,
                "total_steps": len(self.history),
                "root_cause_source": summary.get("root_cause_source"),
                "root_cause_error_type": summary.get("root_cause_error_type"),
                "root_cause_step_index": summary.get("root_cause_step_index"),
                "subgoals_summary": summary.get("subgoals_summary"),
                "drift_summary": summary.get("drift_summary"),
                "timestamp": datetime.datetime.now().isoformat(),
            })
            self._finalized = True
            self._last_summary = summary
            self._last_summarized_step_count = len(self.history)
            return summary


class GuardProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler that routes requests to upstream and inspects agent trajectories."""

    upstream_base_url: str = "https://api.openai.com/v1"
    guard: Optional[GuardInterface] = None
    on_flag: Optional[Callable[[Dict[str, Any]], None]] = None
    fail_open: bool = True
    log_dir: Optional[str] = DEFAULT_LOG_DIR
    idle_timeout: float = 30.0
    sessions: Dict[str, SessionState] = {}
    _lock: threading.Lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet default access logging unless in debug."""
        logger.debug(format, *args)

    def _get_session(self) -> SessionState:
        """Retrieve or initialize the active agent session."""
        session_id = self.headers.get("X-Session-ID") or self.headers.get("X-Run-ID") or "default"
        with self._lock:
            if session_id not in self.sessions:
                g = self.guard if self.guard is not None else GuardInterface()
                self.sessions[session_id] = SessionState(
                    session_id=session_id,
                    guard=g,
                    on_flag=self.on_flag,
                    log_dir=self.log_dir,
                    idle_timeout=self.idle_timeout,
                )
            return self.sessions[session_id]

    def do_GET(self) -> None:
        """Handle health checks and model listings."""
        path = self.path.split("?")[0]
        if path in ("/health", "/", ""):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "longhorizon-guard-proxy"}).encode("utf-8"))
            return

        # Forward GET requests (e.g. /v1/models) to upstream
        self._forward_request(method="GET", body=None)

    def do_POST(self) -> None:
        """Intercept and proxy POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        req_body = self.rfile.read(content_length) if content_length > 0 else b""

        path = self.path.split("?")[0]
        # Support explicit finalization endpoint or header signal
        if path in ("/v1/session/finalize", "/session/finalize") or self.headers.get("X-Session-Finalize", "").lower() in ("true", "1"):
            session = self._get_session()
            summary = session.finalize_session()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "finalized", "session_id": session.session_id, "summary": summary}).encode("utf-8"))
            return

        is_chat = path.endswith("/chat/completions")

        session = self._get_session()

        if is_chat and req_body:
            try:
                parsed_json = json.loads(req_body.decode("utf-8"))
                messages = parsed_json.get("messages", [])
                session.process_messages_before_call(messages)
            except Exception as exc:
                if not self.fail_open:
                    raise
                logger.warning("Guard pre-call observation failed open: %s", exc)

        # Forward request upstream
        self._forward_request(method="POST", body=req_body, is_chat=is_chat, session=session)

    def _forward_request(
        self,
        method: str,
        body: Optional[bytes],
        is_chat: bool = False,
        session: Optional[SessionState] = None,
    ) -> None:
        """Forward HTTP request to the target upstream LLM provider."""
        upstream_base = self.upstream_base_url.rstrip("/")
        path = self.path
        subpath = path[3:] if path.startswith("/v1/") else path
        if upstream_base.endswith("/v1"):
            target_url = upstream_base + subpath
        elif "/openai" in upstream_base:
            target_url = upstream_base.rstrip("/") + subpath
        else:
            target_url = upstream_base + path

        # Prepare request headers
        headers: Dict[str, str] = {}
        for key, val in self.headers.items():
            if key.lower() not in ("host", "content-length"):
                headers[key] = val

        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req) as resp:
                resp_status = resp.status
                resp_headers = resp.headers
                resp_body = resp.read()

                # Post-call observation for non-streaming completions
                if is_chat and session and resp_body:
                    try:
                        content_type = resp_headers.get("Content-Type", "")
                        content_encoding = resp_headers.get("Content-Encoding", "")
                        if "application/json" in content_type:
                            decompressed_body = _decompress_response_body(resp_body, content_encoding)
                            resp_json = json.loads(decompressed_body.decode("utf-8"))
                            session.process_response_after_call(resp_json)
                    except Exception as exc:
                        if not self.fail_open:
                            raise
                        logger.warning("Guard post-call observation failed open: %s", exc)

                # Return response to client
                self.send_response(resp_status)
                for h_key, h_val in resp_headers.items():
                    if h_key.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(h_key, h_val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        except urllib.error.HTTPError as err:
            err_body = err.read()
            self.send_response(err.code)
            for h_key, h_val in err.headers.items():
                if h_key.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(h_key, h_val)
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)

        except Exception as exc:
            logger.exception("Upstream forwarding error: %s", exc)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_msg = json.dumps({"error": {"message": f"Proxy error connecting to upstream: {exc}", "type": "bad_gateway"}}).encode("utf-8")
            self.wfile.write(err_msg)


class GuardProxyServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer that ensures all active sessions are summarized to disk on server close."""

    def server_close(self) -> None:
        try:
            _finalize_all_sessions()
        except Exception:
            pass
        super().server_close()


def run_proxy(
    host: str = "127.0.0.1",
    port: int = 8000,
    upstream: str = "https://api.openai.com/v1",
    guard: Optional[GuardInterface] = None,
    on_flag: Optional[Callable[[Dict[str, Any]], None]] = None,
    fail_open: bool = True,
    log_dir: Optional[str] = DEFAULT_LOG_DIR,
    idle_timeout: float = 30.0,
) -> GuardProxyServer:
    """Launch the LongHorizon Guard HTTP API proxy server.

    Args:
        host: Host interface to bind to (default '127.0.0.1').
        port: Port to listen on (default 8000).
        upstream: Upstream LLM provider base URL (default 'https://api.openai.com/v1').
        guard: Optional GuardInterface instance.
        on_flag: Optional callback function triggered on detected error flags.
        fail_open: Whether to allow requests through if internal guard analysis fails.
        log_dir: Directory where session JSONL files are stored (default 'findings/proxy_sessions').
        idle_timeout: Inactivity window in seconds before auto-finalizing session summary (default 30.0).

    Returns:
        The running GuardProxyServer instance.
    """
    GuardProxyHandler.upstream_base_url = upstream
    GuardProxyHandler.guard = guard if guard is not None else GuardInterface()
    GuardProxyHandler.on_flag = on_flag
    GuardProxyHandler.fail_open = fail_open
    GuardProxyHandler.log_dir = log_dir
    GuardProxyHandler.idle_timeout = idle_timeout

    server = GuardProxyServer((host, port), GuardProxyHandler)
    return server


def _finalize_all_sessions() -> None:
    """Ensure all active proxy sessions are summarized to disk on server exit."""
    with GuardProxyHandler._lock:
        for session in list(GuardProxyHandler.sessions.values()):
            try:
                if len(session.history) != session._last_summarized_step_count:
                    session.finalize_session()
            except Exception as exc:
                logger.debug("Error finalizing session %s on exit: %s", session.session_id, exc)


atexit.register(_finalize_all_sessions)


def main() -> None:
    """CLI entry point for python -m longhorizon_guard.proxy."""
    parser = argparse.ArgumentParser(
        prog="longhorizon-guard proxy",
        description="LongHorizon Guard Real-Time API Proxy for Coding Assistants",
    )
    parser.add_argument("--host", "-H", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--upstream",
        "-u",
        default="https://api.openai.com/v1",
        help="Target upstream LLM endpoint (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--log-dir",
        "-l",
        default=DEFAULT_LOG_DIR,
        help=f"Directory to save session JSONL logs (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=30.0,
        help="Inactivity timeout in seconds before auto-finalizing a session (default: 30.0)",
    )
    parser.add_argument(
        "--no-fail-open",
        action="store_true",
        help="Raise internal guard exceptions instead of failing open",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(f"🛡️  LongHorizon Guard Real-Time API Proxy Running")
    print(f"   Listening on: http://{args.host}:{args.port}")
    print(f"   Upstream LLM: {args.upstream}")
    print(f"   Log Directory: {args.log_dir}")
    print(f"   Idle Timeout: {args.idle_timeout}s")
    print(f"   Fail-Open:    {not args.no_fail_open}")
    print("=" * 65)
    print(f"\nSupported Observation Modes:")
    print(f"   • Native Tool Calls : OpenCode, Cursor, Claude Code (OpenAI function calling)")
    print(f"   • Plain-Text Edits  : Aider (SEARCH/REPLACE diff blocks, whole files, udiff)")
    print(f"\nTo monitor coding assistants, configure:")
    print(f"   export OPENAI_BASE_URL=\"http://{args.host}:{args.port}/v1\"")
    print(f"\nSession transcripts will be saved automatically to:\n   {os.path.abspath(args.log_dir)}")
    print("\nWaiting for agent requests... (Press Ctrl+C to stop)\n")

    server = run_proxy(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        fail_open=not args.no_fail_open,
        log_dir=args.log_dir,
        idle_timeout=args.idle_timeout,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LongHorizon Guard proxy...")
        _finalize_all_sessions()
        server.shutdown()


if __name__ == "__main__":
    main()
