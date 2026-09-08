"""OpenAI-compatible LLM client middleware wrapper for longhorizon_guard.

Provides transparent trajectory observation by intercepting chat.completions.create()
calls on openai.OpenAI() or any OpenAI-compatible client.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from longhorizon_guard.interface import GuardInterface

logger = logging.getLogger(__name__)


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


class _WrappedCompletions:
    """Proxies chat.completions to observe create() calls."""

    def __init__(self, completions: Any, parent: "WrappedOpenAIClient") -> None:
        self._completions = completions
        self._parent = parent

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept chat completion calls to track steps and plans."""
        # 1. Pre-call observation (plan detection & tool response pairing)
        try:
            messages = kwargs.get("messages") or (args[0] if args else [])
            self._parent._process_messages_before_call(messages)
        except Exception as exc:
            if not self._parent.fail_open:
                raise
            logger.exception("Guard observation before create() failed open: %s", exc)

        # 2. Real API call — completely transparent, unaltered
        response = self._completions.create(*args, **kwargs)

        # 3. Post-call observation (assistant thought & tool call capture)
        try:
            self._parent._process_response_after_call(response)
        except Exception as exc:
            if not self._parent.fail_open:
                raise
            logger.exception("Guard observation after create() failed open: %s", exc)

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _WrappedChat:
    """Proxies client.chat to wrap completions."""

    def __init__(self, chat: Any, parent: "WrappedOpenAIClient") -> None:
        self._chat = chat
        self._parent = parent
        self.completions = _WrappedCompletions(getattr(chat, "completions"), parent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class WrappedOpenAIClient:
    """Transparent proxy around an OpenAI or OpenAI-compatible client."""

    def __init__(
        self,
        client: Any,
        guard: Optional[GuardInterface] = None,
        on_flag: Optional[Callable[[Dict[str, Any]], None]] = None,
        fail_open: bool = True,
    ) -> None:
        self._client = client
        self.guard: GuardInterface = guard if guard is not None else GuardInterface()
        self.on_flag: Optional[Callable[[Dict[str, Any]], None]] = on_flag
        self.fail_open: bool = fail_open

        self.history: List[Dict[str, Any]] = []
        self.last_run_summary: Optional[Dict[str, Any]] = None
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._processed_tool_ids: Set[str] = set()
        self._step_counter: int = 0
        self._plan_proposed_called: bool = False
        self._task_description: str = ""

        # Intercept chat completions
        self.chat = _WrappedChat(getattr(client, "chat"), self)

    def _process_messages_before_call(self, messages: Any) -> None:
        """Inspect messages for initial task description and tool responses."""
        if not messages or not isinstance(messages, (list, tuple)):
            return

        # Heuristic for Hook 1: on_plan_proposed on first interaction
        # Limitation: Without explicit task boundaries, we treat the first user message
        # as the task description. If system/user message contains a numbered plan, we pass it.
        if not self._plan_proposed_called:
            task_desc = ""
            plan_text = ""
            for msg in messages:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                text_content = str(content or "")
                if role == "user" and not task_desc:
                    task_desc = text_content
                elif role == "system" and not task_desc and "task" in text_content.lower():
                    task_desc = text_content

                # Extract plan text if present in user or system prompt
                if not plan_text and any(marker in text_content for marker in ("1.", "Step 1", "Plan:", "subgoal", "\n- ")):
                    plan_text = text_content

            if task_desc:
                self._task_description = task_desc
                self.guard.on_plan_proposed(
                    task_description=task_desc,
                    proposed_plan=plan_text,
                    metadata={"source": "openai_client_wrapper"},
                )
                self._plan_proposed_called = True

        # Process any new tool responses sent in this turn
        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role in ("tool", "function"):
                tool_call_id = (
                    msg.get("tool_call_id") if isinstance(msg, dict)
                    else getattr(msg, "tool_call_id", None)
                )
                # Fallback to function name for legacy function calling
                if not tool_call_id:
                    tool_call_id = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", None)

                if not tool_call_id or tool_call_id in self._processed_tool_ids:
                    continue

                tool_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")

                # Correlate with pending tool call from prior turn
                pending = self._pending_tool_calls.pop(str(tool_call_id), None)
                if not pending and self._pending_tool_calls:
                    # Best-effort fallback if IDs differ
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
                    metadata={"source": "openai_client_wrapper"},
                )

                self.history.append(step_record)
                self._processed_tool_ids.add(str(tool_call_id))
                self._step_counter += 1

                if step_res.get("flagged"):
                    if self.on_flag:
                        self.on_flag(step_res)
                    else:
                        logger.warning(
                            "[LongHorizonGuard] Step %d flagged: %s",
                            step_record["step_index"],
                            step_res.get("warning"),
                        )

    def _process_response_after_call(self, response: Any) -> None:
        """Capture assistant reasoning and tool_calls from API response."""
        if not response or not hasattr(response, "choices"):
            return

        choices = getattr(response, "choices", [])
        if not choices:
            return

        choice = choices[0]
        message = getattr(choice, "message", None)
        if not message:
            return

        content = getattr(message, "content", "") or ""
        reasoning = getattr(message, "reasoning_content", None) or content

        # If tracker was in fallback, check if assistant's initial response defines a plan
        if (
            getattr(self.guard, "subgoal_tracker", None)
            and getattr(self.guard.subgoal_tracker, "is_fallback", False)
            and self._step_counter == 0
        ):
            candidate_plan = reasoning or content
            if any(marker in candidate_plan for marker in ("1.", "Step 1", "Plan:", "Phase 1", "\n- ")):
                self.guard.on_plan_proposed(
                    task_description=self._task_description,
                    proposed_plan=candidate_plan,
                    metadata={"source": "openai_client_wrapper", "turn": "assistant_plan"},
                )

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                tc_id = getattr(tc, "id", None) or str(id(tc))
                func = getattr(tc, "function", None)
                fname = getattr(func, "name", "tool") if func else "tool"
                raw_args = getattr(func, "arguments", {}) if func else {}
                parsed_args = _parse_args(raw_args)

                self._pending_tool_calls[str(tc_id)] = {
                    "reasoning": reasoning,
                    "name": fname,
                    "args": parsed_args,
                }

    def finalize_run(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Explicitly finalize trajectory and call guard.on_run_end().

        Note: Because LLM client sessions do not have a universal 'done' signal,
        the host must call finalize_run() when the task completes.
        """
        meta = {"task": self._task_description}
        if metadata:
            meta.update(metadata)

        self.last_run_summary = self.guard.on_run_end(
            metadata=meta,
            trajectory={"steps": self.history},
        )
        return self.last_run_summary

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_guard(
    client: Any,
    guard: Optional[GuardInterface] = None,
    on_flag: Optional[Callable[[Dict[str, Any]], None]] = None,
    fail_open: bool = True,
) -> WrappedOpenAIClient:
    """Wrap an OpenAI or OpenAI-compatible client with LongHorizon Guard monitoring.

    Args:
        client: An instance of openai.OpenAI() or compatible client.
        guard: Optional GuardInterface instance (created with defaults if None).
        on_flag: Optional callback called when a step is flagged.
        fail_open: If True (default), guard errors will never break client calls.

    Returns:
        WrappedOpenAIClient that observes chat completions transparently.
    """
    return WrappedOpenAIClient(
        client=client,
        guard=guard,
        on_flag=on_flag,
        fail_open=fail_open,
    )


# Backward-compatibility alias
OpenAIClientWrapper = WrappedOpenAIClient

