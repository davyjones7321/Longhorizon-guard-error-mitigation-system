#!/usr/bin/env python3
"""
Adapter: Antigravity Session Hook → longhorizon_guard SCHEMA.md v1.0 format.

Capture-and-transform hook that reads session lifecycle payload on stdin
(from Antigravity Stop event), reads transcript.jsonl, transforms steps and
metadata into SCHEMA.md format, and upserts into findings/antigravity_sessions.json.

Does NOT invoke taggers, LLMs, or network services. Pure fast JSON transformation.

Usage (via hooks.json):
    python -m longhorizon_guard.storage.adapters.antigravity_session
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_PATH = os.path.join("findings", "antigravity_sessions.json")

CREDENTIAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # OpenAI, OpenRouter, Anthropic secret keys
    (re.compile(r"\b(sk-(?:proj-|ant-|or-v1-)?[A-Za-z0-9_\-]{20,})\b"), "sk-***REDACTED***"),
    # Google Gemini / Cloud API keys
    (re.compile(r"\b(AIzaSy[A-Za-z0-9_\-]{33})\b"), "AIzaSy***REDACTED***"),
    # Groq API keys
    (re.compile(r"\b(gsk_[A-Za-z0-9]{20,})\b"), "gsk_***REDACTED***"),
    # NVIDIA NIM API keys
    (re.compile(r"\b(nvapi-[A-Za-z0-9_\-]{20,})\b"), "nvapi-***REDACTED***"),
    # TokenRouter API keys
    (re.compile(r"\b(tr-[A-Za-z0-9_\-]{20,})\b"), "tr-***REDACTED***"),
    # Authorization: Bearer tokens
    (re.compile(r"\b(Bearer\s+)([A-Za-z0-9\._\-]{20,})\b", re.IGNORECASE), r"\1***REDACTED***"),
    # Key assignment patterns: api_key="...", secret='...'
    (
        re.compile(
            r'((?:api[_-]?key|access[_-]?token|auth[_-]?token|secret)\s*[:=]\s*["\'])([A-Za-z0-9_\-\.]{20,})(["\'])',
            re.IGNORECASE,
        ),
        r"\1***REDACTED***\3",
    ),
]


def sanitize_text(text: str) -> str:
    """Mask known credentials and sensitive tokens in text with ***REDACTED***."""
    if not text or not isinstance(text, str):
        return text
    sanitized = text
    for pattern, replacement in CREDENTIAL_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_data(data: Any) -> Any:
    """Recursively sanitize all string values in nested dictionaries and lists."""
    if isinstance(data, str):
        return sanitize_text(data)
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data


def _parse_iso_timestamp(ts_str: Optional[str]) -> Optional[float]:
    """Parse ISO timestamp string to Unix epoch seconds."""
    if not ts_str:
        return None
    try:
        # Handle 'Z' suffix
        cleaned = ts_str.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(cleaned)
        return dt.timestamp()
    except Exception:
        return None


def _build_task_info(user_input_content: str) -> Tuple[str, str]:
    """Extract deterministic task_id hash and human-readable task_description."""
    content = (user_input_content or "").strip()
    req_text = content
    if "<USER_REQUEST>" in content:
        try:
            req_text = content.split("<USER_REQUEST>")[1].split("</USER_REQUEST>")[0].strip()
        except IndexError:
            pass

    # Normalize prompt text for hashing
    normalized = re.sub(r"\s+", " ", req_text.lower()).strip()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    task_id = f"antigravity:{digest}" if digest else "antigravity:unknown"

    # Human-readable task description (first 500 chars of request)
    task_description = req_text[:500] if req_text else "antigravity_session"

    return task_id, task_description


def parse_transcript_lines(lines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, str, Optional[str], Optional[str]]:
    """Parse raw transcript JSONL objects into SCHEMA.md steps and metadata fields.

    Returns:
        (steps, task_id, task_description, start_iso_timestamp, end_iso_timestamp)
    """
    steps: List[Dict[str, Any]] = []
    task_id = "antigravity:unknown"
    task_description = "antigravity_session"
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    step_index = 0

    first_user_found = False

    i = 0
    num_lines = len(lines)

    while i < num_lines:
        item = lines[i]
        created_at = item.get("created_at")
        if created_at:
            if start_ts is None:
                start_ts = created_at
            end_ts = created_at

        item_type = item.get("type")
        source = item.get("source")

        # Capture first user request for task_id and task_description
        if item_type == "USER_INPUT" and not first_user_found:
            content = item.get("content", "")
            if content:
                task_id, task_description = _build_task_info(content)
                first_user_found = True

        # Process model responses
        if item_type == "PLANNER_RESPONSE" and source == "MODEL":
            reasoning_text = (item.get("content") or "").strip()
            thinking_text = (item.get("thinking") or "").strip()
            if thinking_text:
                full_reasoning = f"{thinking_text}\n\n{reasoning_text}".strip() if reasoning_text else thinking_text
            else:
                full_reasoning = reasoning_text

            tool_calls = item.get("tool_calls") or []

            # Look ahead for tool execution outputs (GENERIC steps following this model step)
            tool_responses: List[str] = []
            j = i + 1
            while j < num_lines and lines[j].get("type") == "GENERIC" and lines[j].get("source") == "MODEL":
                resp_content = lines[j].get("content", "")
                tool_responses.append(resp_content)
                if lines[j].get("created_at"):
                    end_ts = lines[j].get("created_at")
                j += 1

            step_ts = _parse_iso_timestamp(created_at)

            # Case 1: Reasoning turn with NO tool calls
            if not tool_calls:
                steps.append({
                    "step_index": step_index,
                    "timestamp": step_ts,
                    "reasoning": full_reasoning,
                    "action_name": "done" if "goal" in task_id.lower() or step_index > 0 else "none",
                    "action_args": {},
                    "tool_response": None,
                    "state_snapshot": None,
                    "error_tag": None,
                })
                step_index += 1

            # Case 2: One or more tool calls in this turn
            else:
                for tc_idx, tc in enumerate(tool_calls):
                    tc_name = tc.get("name", "unknown")
                    tc_args = tc.get("args") or {}
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except Exception:
                            tc_args = {"raw": tc_args}
                    if isinstance(tc_args, dict):
                        tc_args = {
                            k: json.loads(v) if (isinstance(v, str) and len(v) >= 2 and v.startswith('"') and v.endswith('"')) else v
                            for k, v in tc_args.items()
                        }

                    resp_str = tool_responses[tc_idx] if tc_idx < len(tool_responses) else None

                    steps.append({
                        "step_index": step_index,
                        "timestamp": step_ts,
                        "reasoning": full_reasoning if tc_idx == 0 else "",
                        "action_name": tc_name,
                        "action_args": tc_args,
                        "tool_response": resp_str,
                        "state_snapshot": None,
                        "error_tag": None,
                    })
                    step_index += 1

            i = j - 1  # Skip consumed tool response steps

        i += 1

    return steps, task_id, task_description, start_ts, end_ts


def process_session(
    payload: Dict[str, Any],
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Transform session payload + transcript into SCHEMA.md format and upsert."""
    conversation_id = payload.get("conversationId", "unknown_session")
    transcript_path = payload.get("transcriptPath")
    model_name = payload.get("modelName", "unknown")
    term_reason = payload.get("terminationReason", "model_stop")

    # Map terminationReason to final_status
    if term_reason == "model_stop":
        final_status = "completed_unverified"
    elif term_reason == "max_steps_exceeded":
        final_status = "timeout"
    elif term_reason == "error":
        final_status = "error"
    else:
        final_status = "completed_unverified"

    lines: List[Dict[str, Any]] = []
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        lines.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        continue
    else:
        logger.warning("Transcript path missing or not found: %s", transcript_path)

    steps, task_id, task_description, start_iso, end_iso = parse_transcript_lines(lines)

    # Compute wall-clock duration and active duration (excluding idle gaps > 300s)
    start_sec = _parse_iso_timestamp(start_iso)
    end_sec = _parse_iso_timestamp(end_iso)
    duration = (end_sec - start_sec) if (start_sec and end_sec and end_sec >= start_sec) else None

    step_timestamps = [s["timestamp"] for s in steps if s.get("timestamp") is not None]
    active_duration: Optional[float] = None
    if len(step_timestamps) >= 2:
        active_accum = 0.0
        for k in range(len(step_timestamps) - 1):
            delta = step_timestamps[k + 1] - step_timestamps[k]
            if 0.0 <= delta <= 300.0:  # Ignore idle gaps > 5 minutes
                active_accum += delta
        active_duration = round(active_accum, 2)

    # Construct metadata conforming to SCHEMA.md
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": conversation_id,
        "task_id": task_id,
        "task_description": task_description,
        "trial_number": 1,
        "final_status": final_status,
        "total_steps_taken": len(steps),
        "horizon_level": None,
        "duration_seconds": round(duration, 2) if duration else None,
        "active_duration_seconds": active_duration,
        "grader_notes": f"terminationReason={term_reason}",
        "created_at": start_iso,
        "root_cause_step_index": None,
        "root_cause_error_type": None,
        "tag_confidence": None,
        "tag_source": None,
        "source_dataset": "antigravity_session",
        "source_llm_model": model_name,
    }

    raw_record = {
        "metadata": metadata,
        "trajectory": {"schema_version": SCHEMA_VERSION, "steps": steps} if steps else None,
    }
    run_record = sanitize_data(raw_record)

    # Upsert into consolidated JSON
    runs: List[Dict[str, Any]] = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                runs = data.get("runs", [])
        except Exception as exc:
            logger.warning("Failed to load existing %s: %s", output_path, exc)
            runs = []

    # Find and update existing conversation record, or append
    existing_idx = None
    for idx, r in enumerate(runs):
        if r.get("metadata", {}).get("run_id") == conversation_id:
            existing_idx = idx
            break

    if existing_idx is not None:
        runs[existing_idx] = run_record
    else:
        runs.append(run_record)

    # Apply sanitization to ensure no unsanitized credentials touch disk
    sanitized_runs = sanitize_data(runs)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"runs": sanitized_runs}, f, ensure_ascii=False, indent=2)

    return run_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Antigravity session lifecycle capture adapter")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    parser.add_argument("--payload-file", help="Optional payload JSON file for testing without stdin")
    args = parser.parse_args()

    # Read stdin payload or file payload
    payload: Dict[str, Any] = {}
    if args.payload_file and os.path.exists(args.payload_file):
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif not sys.stdin.isatty():
        try:
            raw_in = sys.stdin.read()
            if raw_in.strip():
                payload = json.loads(raw_in)
        except Exception as exc:
            logger.error("Failed to parse stdin payload: %s", exc)

    if not payload:
        # Fallback dummy payload for dry-run CLI test
        payload = {
            "conversationId": "test_dry_run_session",
            "modelName": "gemini-2.5-flash",
            "terminationReason": "model_stop",
            "transcriptPath": "",
        }

    run_rec = process_session(payload, args.output)
    meta = run_rec["metadata"]
    print(json.dumps({
        "status": "success",
        "run_id": meta["run_id"],
        "steps_captured": meta["total_steps_taken"],
        "output_path": args.output,
    }))


if __name__ == "__main__":
    main()
