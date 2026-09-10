"""OpenAI Codex and Claude Code Native Lifecycle Hook Integration for LongHorizon Guard.

Reads hook event payloads from stdin, tracks multi-step agent trajectories,
flags drift/loops in real time, intercepts harmful actions (PreToolUse deny),
and injects course-correction context (PostToolUse additionalContext).
"""

import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path regardless of execution directory
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.storage.adapters.antigravity_session import sanitize_data
from longhorizon_guard.subgoals.tracker import (
    parse_plan_subgoals,
    _NUMBERED_PATTERN,
    _BULLET_PATTERN,
    _TRANSITION_WORDS_PATTERN,
)

logger = logging.getLogger("longhorizon_guard.hook")
DEFAULT_HOOK_LOG_DIR = os.path.join(repo_root, "findings", "hook_sessions")


def _get_state_path(session_id: str, log_dir: str) -> str:
    """Return the path to the temporary JSON state file for the session."""
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(log_dir, f"_state_{safe_id}.json")


def _load_session_state(session_id: str, log_dir: str) -> Dict[str, Any]:
    """Load existing session state from disk or return a new empty state."""
    state_file = _get_state_path(session_id, log_dir)
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                state.setdefault("phase_check_pending", False)
                state.setdefault("transcript_offset", 0)
                return state
        except Exception as exc:
            logger.debug("Failed to load state file %s: %s", state_file, exc)

    return {
        "session_id": session_id,
        "step_counter": 0,
        "task_description": "",
        "proposed_plan": "",
        "history": [],
        "recent_actions": [],
        "halt_next_tool": False,
        "halt_reason": "",
        "phase_check_pending": False,
        "transcript_offset": 0,
        "created_at": datetime.datetime.now().isoformat(),
    }


def _save_session_state(state: Dict[str, Any], log_dir: str) -> None:
    """Persist session state to disk."""
    session_id = state.get("session_id", "default")
    state_file = _get_state_path(session_id, log_dir)
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        logger.debug("Failed to save state file %s: %s", state_file, exc)


def _append_session_log(session_id: str, entry: Dict[str, Any], log_dir: str) -> None:
    """Sanitize and write an audit record to the session JSONL file."""
    try:
        os.makedirs(log_dir, exist_ok=True)
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        log_file = os.path.join(log_dir, f"session_{safe_id}.jsonl")
        cleaned_entry = sanitize_data(entry)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(cleaned_entry) + "\n")
    except Exception as exc:
        logger.debug("Failed to write to session log: %s", exc)


def _apply_plan_rejection(state: Dict[str, Any], plan_res: Dict[str, Any]) -> None:
    """Set halt_next_tool and halt_reason on state according to plan rejection result."""
    state["halt_next_tool"] = True
    flags = plan_res.get("flags") or []
    flags_str = "; ".join(flags) if flags else "Plan failed guard validation policy"
    suggs = plan_res.get("suggestions") or []
    sugg_str = f" Suggestion: {suggs[0]}" if suggs else ""
    state["halt_reason"] = f"Plan rejected: {flags_str}.{sugg_str}".strip()


def _is_substantive_text(s: str) -> bool:
    """Determine if a text snippet has substantive content beyond short greetings/acknowledgments."""
    cleaned = s.strip()
    if len(cleaned) < 15:
        return False
    words = [w for w in cleaned.split() if w]
    if len(words) < 3:
        return False
    lower = cleaned.lower().rstrip(".!;,")
    if lower in ("done", "ok", "okay", "sure", "acknowledged", "understood", "will do", "working on it", "got it", "i will start now"):
        return False
    return True


def _extract_recent_plan_text(chunk: str) -> Optional[str]:
    """Find the most recent real assistant text block or thinking block from a transcript chunk.

    Walks backward from the end, skips low-content messages, falls back to thinking-block content
    if present, and returns None if nothing substantive is found.
    Wrapped in broad try/except to fail open on any schema surprises.
    """
    if not chunk or not chunk.strip():
        return None

    try:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            return None

        for line_str in reversed(lines):
            try:
                item = json.loads(line_str)
            except Exception:
                continue

            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type", "")).lower()
            role = str(item.get("role", "")).lower()

            is_assistant = False
            if role == "assistant" or item_type == "assistant":
                is_assistant = True
            elif item_type == "message" and role == "assistant":
                is_assistant = True
            elif isinstance(item.get("message"), dict) and str(item["message"].get("role", "")).lower() == "assistant":
                is_assistant = True

            is_reasoning = (item_type in ("reasoning", "thinking"))

            if not is_assistant and not is_reasoning:
                continue

            text_candidates: List[str] = []
            thinking_candidates: List[str] = []

            # 1. Content field extraction
            content = item.get("content")
            if content is None and isinstance(item.get("message"), dict):
                content = item["message"].get("content")

            if isinstance(content, str) and content.strip():
                if is_reasoning:
                    thinking_candidates.append(content.strip())
                else:
                    text_candidates.append(content.strip())
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str) and block.strip():
                        if is_reasoning:
                            thinking_candidates.append(block.strip())
                        else:
                            text_candidates.append(block.strip())
                    elif isinstance(block, dict):
                        btype = str(block.get("type", "")).lower()
                        btext = block.get("text") or block.get("content") or ""
                        if isinstance(btext, str) and btext.strip():
                            if btype in ("thinking", "reasoning", "reasoning_text"):
                                thinking_candidates.append(btext.strip())
                            elif btype in ("text", "output_text") or not btype:
                                text_candidates.append(btext.strip())
                        bthinking = block.get("thinking")
                        if isinstance(bthinking, str) and bthinking.strip():
                            thinking_candidates.append(bthinking.strip())

            # 2. Raw content field (WorkBuddy/Claude Code reasoning block)
            raw_content = item.get("rawContent") or item.get("raw_content")
            if isinstance(raw_content, str) and raw_content.strip():
                thinking_candidates.append(raw_content.strip())
            elif isinstance(raw_content, list):
                for block in raw_content:
                    if isinstance(block, str) and block.strip():
                        thinking_candidates.append(block.strip())
                    elif isinstance(block, dict):
                        t = block.get("text") or block.get("content") or block.get("thinking")
                        if isinstance(t, str) and t.strip():
                            thinking_candidates.append(t.strip())

            # 3. Direct text or thinking attributes
            direct_text = item.get("text")
            if isinstance(direct_text, str) and direct_text.strip():
                text_candidates.append(direct_text.strip())
            direct_thinking = item.get("thinking")
            if isinstance(direct_thinking, str) and direct_thinking.strip():
                thinking_candidates.append(direct_thinking.strip())

            combined_text = "\n".join(text_candidates).strip()
            combined_thinking = "\n".join(thinking_candidates).strip()

            if combined_text and _is_substantive_text(combined_text):
                return combined_text
            elif combined_thinking and _is_substantive_text(combined_thinking):
                return combined_thinking

    except Exception as exc:
        logger.debug("Failed extracting assistant plan text from transcript chunk: %s", exc)
        return None

    return None


def _is_plan_shaped(text: str) -> bool:
    """Check if extracted text is structurally plan-shaped (numbered list, bullets, or transition words)."""
    try:
        subgoals, is_fallback = parse_plan_subgoals(text)
        if is_fallback:
            return False
        return bool(
            len(_NUMBERED_PATTERN.findall(text)) >= 2
            or len(_BULLET_PATTERN.findall(text)) >= 2
            or len(_TRANSITION_WORDS_PATTERN.findall(text)) >= 2
        )
    except Exception as exc:
        logger.debug("Failed checking if text is plan-shaped: %s", exc)
        return False


def _detect_agent_source(payload: Dict[str, Any]) -> str:
    """Detect whether the invoking agent is Claude Code, Codex, or generic harness."""
    transcript_path = str(payload.get("transcript_path") or "").lower()
    if ".claude" in transcript_path or "CLAUDE_PROJECT_DIR" in os.environ:
        return "claude_code_hook"
    if ".codex" in transcript_path or any(k.startswith("CODEX_") for k in os.environ):
        return "codex_hook"
    return "codex_hook"


def handle_hook(payload: Dict[str, Any], guard: Optional[GuardInterface] = None, log_dir: str = DEFAULT_HOOK_LOG_DIR) -> Dict[str, Any]:
    """Process a single lifecycle hook event payload."""
    event_name = payload.get("hook_event_name", "")
    session_id = str(payload.get("session_id") or "default")
    agent_source = _detect_agent_source(payload)
    state = _load_session_state(session_id, log_dir)

    if guard is None:
        from longhorizon_guard.config import GuardConfig
        cfg = GuardConfig.from_env()
        if os.getenv("GUARD_ENABLE_MEMORY", "true").lower() in ("true", "1", "yes"):
            cfg.enable_memory = True
        guard = GuardInterface(config=cfg)
        # Restore session state if resuming an active session across process invocations
        task_desc = state.get("task_description", "")
        plan_desc = state.get("proposed_plan", "")
        if task_desc or plan_desc:
            guard.on_plan_proposed(
                task_description=task_desc,
                proposed_plan=plan_desc,
                metadata={"source": agent_source, "session_id": session_id},
            )
            # Replay prior history steps to restore tracker, drift, and reflector state
            replay_hist = []
            for past_step in state.get("history", []):
                guard.on_step(
                    step_record=past_step,
                    history=replay_hist,
                    metadata={"source": agent_source, "session_id": session_id, "in_replay": True},
                )
                replay_hist.append(past_step)

    # 1. USER PROMPT SUBMISSION (Initial Task and Plan Proposal)
    if event_name == "UserPromptSubmit":
        prompt = payload.get("prompt", "") or ""
        plan = payload.get("plan", "") or payload.get("proposed_plan", "") or ""
        if not plan and ("1." in prompt or "Plan:" in prompt or "\n-" in prompt):
            plan = prompt
        state["task_description"] = prompt
        state["proposed_plan"] = plan
        _save_session_state(state, log_dir)

        plan_res = guard.on_plan_proposed(
            task_description=prompt,
            proposed_plan=plan,
            metadata={"source": agent_source, "session_id": session_id},
        )

        # Check if guard rejected the plan
        if plan_res.get("approved") is False:
            _apply_plan_rejection(state, plan_res)

        _save_session_state(state, log_dir)

        _append_session_log(session_id, {
            "type": "plan",
            "session_id": session_id,
            "task_description": prompt,
            "approved": plan_res.get("approved", True),
            "flags": plan_res.get("flags", []),
            "suggestions": plan_res.get("suggestions", []),
            "timestamp": datetime.datetime.now().isoformat(),
        }, log_dir)

        sys.stderr.write(
            f"\n\033[96m🛡️  [LONGHORIZON GUARD]\033[0m Observing session: {session_id[:8]}...\n"
            f"   Task: {prompt[:80]}...\n\n"
        )
        sys.stderr.flush()

        return {"continue": True}

    # 2. PRE TOOL USE (Interception & Loop/Error Prevention)
    elif event_name == "PreToolUse":
        tool_name = payload.get("tool_name", "tool")
        tool_input = payload.get("tool_input", {})
        recent_actions = state.get("recent_actions", [])

        # Check if a phase boundary occurred and a phase plan check is pending
        if state.get("phase_check_pending"):
            state["phase_check_pending"] = False
            transcript_path = payload.get("transcript_path")
            if transcript_path and isinstance(transcript_path, str) and os.path.isfile(transcript_path):
                try:
                    offset = state.get("transcript_offset", 0)
                    file_size = os.path.getsize(transcript_path)
                    if offset > file_size:
                        offset = 0

                    with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        chunk = f.read()
                        state["transcript_offset"] = f.tell()

                    extracted_text = _extract_recent_plan_text(chunk)
                    if extracted_text and _is_plan_shaped(extracted_text):
                        plan_res = guard.on_plan_proposed(
                            task_description=state.get("task_description", ""),
                            proposed_plan=extracted_text,
                            metadata={"source": agent_source, "session_id": session_id, "phase": True},
                        )
                        if plan_res.get("approved") is False:
                            _apply_plan_rejection(state, plan_res)

                        _append_session_log(session_id, {
                            "type": "plan",
                            "session_id": session_id,
                            "task_description": state.get("task_description", ""),
                            "proposed_plan": extracted_text,
                            "approved": plan_res.get("approved", True),
                            "flags": plan_res.get("flags", []),
                            "suggestions": plan_res.get("suggestions", []),
                            "phase": True,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }, log_dir)
                except Exception as exc:
                    logger.debug("Failed processing transcript for phase critique: %s", exc)

            _save_session_state(state, log_dir)

        # Check for one-shot halt triggered by critical guard flag or plan rejection
        if state.get("halt_next_tool"):
            halt_reason = state.get("halt_reason") or "Critical guard intervention halted execution"
            state["halt_next_tool"] = False
            state["halt_reason"] = ""
            _save_session_state(state, log_dir)

            sys.stderr.write(f"\n\033[91m🛑  [LONGHORIZON GUARD BLOCKED]\033[0m {halt_reason}\n\n")
            sys.stderr.flush()

            _append_session_log(session_id, {
                "type": "block",
                "session_id": session_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "reason": halt_reason,
                "timestamp": datetime.datetime.now().isoformat(),
            }, log_dir)

            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": halt_reason,
                }
            }

        # Check for repeated identical actions in a loop (e.g. failing command 3+ times)
        action_signature = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        recent_matches = sum(1 for a in recent_actions[-4:] if a.get("signature") == action_signature and a.get("failed"))

        if recent_matches >= 3:
            block_reason = (
                f"⚠️ [LONGHORIZON GUARD BLOCKED] Loop Detected: Tool '{tool_name}' has repeated "
                f"the exact same failing action {recent_matches} times. Stopping execution loop. "
                "Please analyze the previous error and adopt an alternative strategy."
            )
            sys.stderr.write(f"\n\033[91m🛑  [LONGHORIZON GUARD BLOCKED]\033[0m {block_reason}\n\n")
            sys.stderr.flush()

            _append_session_log(session_id, {
                "type": "block",
                "session_id": session_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "reason": block_reason,
                "timestamp": datetime.datetime.now().isoformat(),
            }, log_dir)

            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": block_reason,
                }
            }

        state["pending_tool"] = {
            "name": tool_name,
            "input": tool_input,
            "signature": action_signature,
        }
        _save_session_state(state, log_dir)
        return {"continue": True}

    # 3. POST TOOL USE (Observation, Drift Detection, and Context Steering)
    elif event_name == "PostToolUse":
        tool_name = payload.get("tool_name", "tool")
        tool_input = payload.get("tool_input", {})
        tool_response = payload.get("tool_response") or payload.get("tool_result") or ""

        # Check if output contains an error
        resp_str = str(tool_response)
        has_error = any(err in resp_str.lower() for err in ("error", "failed", "traceback", "exception", "exit code 1", "exit status 1"))

        step_idx = state.get("step_counter", 0)
        state["step_counter"] = step_idx + 1

        prior_history = list(state.get("history", []))

        step_record = {
            "step_index": step_idx,
            "reasoning": "",
            "action_name": tool_name,
            "action_args": tool_input if isinstance(tool_input, dict) else {"raw": str(tool_input)},
            "tool_response": resp_str,
        }

        # Update recent actions tracker for loop detection
        signature = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        state.setdefault("recent_actions", []).append({"signature": signature, "failed": has_error})
        if len(state["recent_actions"]) > 10:
            state["recent_actions"].pop(0)

        step_res = guard.on_step(
            step_record=step_record,
            history=prior_history,
            metadata={"source": agent_source, "session_id": session_id},
        )

        state["history"].append(step_record)

        # Check if a subgoal transition occurred (new phase boundary)
        if step_res.get("subgoal_transition"):
            state["phase_check_pending"] = True

        match_det = step_res.get("match_details") or {}
        cat = step_res.get("category") or match_det.get("category")
        conf = step_res.get("confidence")
        if conf is None:
            conf = match_det.get("confidence", 0.0)
        suggestions = step_res.get("suggestions")
        if not suggestions and match_det.get("safe_alternative"):
            suggestions = [match_det["safe_alternative"]]
        elif not suggestions:
            suggestions = []

        _append_session_log(session_id, {
            "type": "step",
            "session_id": session_id,
            "step_index": step_idx,
            "action_name": tool_name,
            "flagged": step_res.get("flagged", False),
            "category": cat,
            "confidence": conf,
            "warning": step_res.get("warning"),
            "suggestions": suggestions,
            "timestamp": datetime.datetime.now().isoformat(),
        }, log_dir)

        # Check if guard halted execution on critical flag
        if step_res.get("continue_execution") is False:
            state["halt_next_tool"] = True
            cat_str = cat or "other"
            warn_str = step_res.get("warning") or "Critical guard intervention"
            state["halt_reason"] = f"Critical flag on step {step_idx} [{cat_str}]: {warn_str}"

        _save_session_state(state, log_dir)

        # If flagged, alert the user and inject steer guidance into LLM's context window!
        if step_res.get("flagged"):
            warning = step_res.get("warning", "Potential issue detected")
            cat_display = cat or "other"
            sugg_str = f" Suggestion: {suggestions[0]}" if suggestions else ""

            sys.stderr.write(
                f"\n\033[93m⚠️  [LONGHORIZON GUARD ALERT]\033[0m Step {step_idx} "
                f"[{cat_display} conf={conf:.2f}]:\n    {warning}{sugg_str}\n\n"
            )
            sys.stderr.flush()

            guidance = f"\n[LONGHORIZON GUARD ADVISORY] {warning}.{sugg_str}\n"
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": guidance,
                }
            }

        return {"continue": True}

    # 4. TURN OR SESSION END
    elif event_name in ("Stop", "SessionEnd"):
        if state.get("history"):
            summary = guard.on_run_end(
                metadata={"session_id": session_id, "task": state.get("task_description", "")},
                trajectory={"steps": state.get("history", [])},
            )
            _append_session_log(session_id, {
                "type": "summary",
                "session_id": session_id,
                "total_steps": len(state.get("history", [])),
                "root_cause_source": summary.get("root_cause_source"),
                "root_cause_error_type": summary.get("root_cause_error_type"),
                "root_cause_step_index": summary.get("root_cause_step_index"),
                "subgoals_summary": summary.get("subgoals_summary"),
                "drift_summary": summary.get("drift_summary"),
                "timestamp": datetime.datetime.now().isoformat(),
            }, log_dir)

            sys.stderr.write(
                f"\n\033[92m✅  [LONGHORIZON GUARD]\033[0m Session {session_id[:8]} completed ({len(state.get('history', []))} steps).\n"
            )
            sys.stderr.flush()

        return {"continue": True}

    return {"continue": True}


def main() -> None:
    """Main CLI entrypoint for stdio hook execution."""
    try:
        raw_input = sys.stdin.read().strip()
        if not raw_input:
            print(json.dumps({"continue": True}))
            sys.exit(0)

        payload = json.loads(raw_input)
        response = handle_hook(payload)
        print(json.dumps(response))
        sys.exit(0)
    except Exception as exc:
        # Fail-open: Never block user workflow if hook encounters an exception
        logger.warning("Hook execution encountered error (failing open): %s", exc)
        print(json.dumps({"continue": True}))
        sys.exit(0)


if __name__ == "__main__":
    main()
