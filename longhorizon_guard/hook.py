"""WorkBuddy and CodeBuddy Native Lifecycle Hook Integration for LongHorizon Guard.

Reads hook event payloads from stdin, tracks multi-step agent trajectories,
flags drift/loops in real time, intercepts harmful actions (PreToolUse deny),
and injects course-correction context (PostToolUse additionalContext).
"""

import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Ensure repository root is on sys.path regardless of execution directory
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.storage.adapters.antigravity_session import sanitize_data

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
                return json.load(f)
        except Exception as exc:
            logger.debug("Failed to load state file %s: %s", state_file, exc)

    return {
        "session_id": session_id,
        "step_counter": 0,
        "task_description": "",
        "history": [],
        "recent_actions": [],
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


def handle_hook(payload: Dict[str, Any], guard: Optional[GuardInterface] = None, log_dir: str = DEFAULT_HOOK_LOG_DIR) -> Dict[str, Any]:
    """Process a single lifecycle hook event payload."""
    if guard is None:
        guard = GuardInterface()

    event_name = payload.get("hook_event_name", "")
    session_id = str(payload.get("session_id") or "default")
    state = _load_session_state(session_id, log_dir)

    # 1. USER PROMPT SUBMISSION (Initial Task and Plan Proposal)
    if event_name == "UserPromptSubmit":
        prompt = payload.get("prompt", "") or ""
        state["task_description"] = prompt
        _save_session_state(state, log_dir)

        plan_res = guard.on_plan_proposed(
            task_description=prompt,
            proposed_plan="",
            metadata={"source": "workbuddy_hook", "session_id": session_id},
        )

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

        step_record = {
            "step_index": step_idx,
            "reasoning": "",
            "action_name": tool_name,
            "action_args": tool_input if isinstance(tool_input, dict) else {"raw": str(tool_input)},
            "tool_response": resp_str,
        }
        state["history"].append(step_record)

        # Update recent actions tracker for loop detection
        signature = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        state.setdefault("recent_actions", []).append({"signature": signature, "failed": has_error})
        if len(state["recent_actions"]) > 10:
            state["recent_actions"].pop(0)

        step_res = guard.on_step(
            step_record=step_record,
            history=state["history"],
            metadata={"source": "workbuddy_hook", "session_id": session_id},
        )

        _append_session_log(session_id, {
            "type": "step",
            "session_id": session_id,
            "step_index": step_idx,
            "action_name": tool_name,
            "flagged": step_res.get("flagged", False),
            "category": step_res.get("category"),
            "confidence": step_res.get("confidence", 0.0),
            "warning": step_res.get("warning"),
            "suggestions": step_res.get("suggestions", []),
            "timestamp": datetime.datetime.now().isoformat(),
        }, log_dir)

        _save_session_state(state, log_dir)

        # If flagged, alert the user and inject steer guidance into LLM's context window!
        if step_res.get("flagged"):
            warning = step_res.get("warning", "Potential issue detected")
            cat = step_res.get("category", "drift")
            conf = step_res.get("confidence", 0.0)
            suggestions = step_res.get("suggestions", [])
            sugg_str = f" Suggestion: {suggestions[0]}" if suggestions else ""

            sys.stderr.write(
                f"\n\033[93m⚠️  [LONGHORIZON GUARD ALERT]\033[0m Step {step_idx} "
                f"[{cat} conf={conf:.2f}]:\n    {warning}{sugg_str}\n\n"
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
