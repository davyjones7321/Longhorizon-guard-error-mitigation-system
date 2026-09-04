"""LangChain / LangGraph callback handler for longhorizon_guard.

Provides zero-code-change trajectory monitoring by translating LangChain agent lifecycle
events into GuardInterface step records and hooks.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from longhorizon_guard.interface import GuardInterface

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler  # type: ignore[no-redef]
    except ImportError:
        class BaseCallbackHandler:  # type: ignore[no-redef]
            """Fallback stub when langchain is not installed."""
            pass


class LongHorizonGuardCallback(BaseCallbackHandler):
    """Callback handler that translates LangChain events into GuardInterface hooks.

    Hooks called:
    - on_chain_start -> guard.on_plan_proposed()
    - on_agent_action + on_tool_end / on_tool_error -> guard.on_step()
    - on_chain_end -> guard.on_run_end()
    """

    def __init__(
        self,
        guard: Optional[GuardInterface] = None,
        on_flag: Optional[Callable[[Dict[str, Any]], None]] = None,
        fail_open: bool = True,
    ) -> None:
        super().__init__()
        self.guard: GuardInterface = guard if guard is not None else GuardInterface()
        self.on_flag: Optional[Callable[[Dict[str, Any]], None]] = on_flag
        self.fail_open: bool = fail_open

        self.history: List[Dict[str, Any]] = []
        self.last_run_summary: Optional[Dict[str, Any]] = None
        self._pending_actions: Dict[str, Dict[str, Any]] = {}
        self._latest_action: Optional[Dict[str, Any]] = None
        self._step_counter: int = 0
        self._task_description: str = ""
        self._proposed_plan: str = ""
        self._chain_depth: int = 0

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain starts. Top-level chain triggers on_plan_proposed."""
        try:
            self._chain_depth += 1
            if self._chain_depth == 1 or parent_run_id is None:
                # Extract task description and plan heuristics from inputs
                task_desc = ""
                plan_text = ""
                if isinstance(inputs, dict):
                    task_desc = str(
                        inputs.get("input")
                        or inputs.get("query")
                        or inputs.get("task")
                        or inputs.get("prompt")
                        or ""
                    )
                    plan_text = str(inputs.get("plan") or inputs.get("proposed_plan") or "")
                elif isinstance(inputs, str):
                    task_desc = inputs

                self._task_description = task_desc
                self._proposed_plan = plan_text

                if self.guard is not None:
                    self.guard.on_plan_proposed(
                        task_description=task_desc,
                        proposed_plan=plan_text,
                        metadata={"run_id": str(run_id) if run_id else "unknown"},
                    )
        except Exception as exc:
            if not self.fail_open:
                raise
            logger.exception("LongHorizonGuardCallback.on_chain_start failed open: %s", exc)

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when an agent decides on an action (tool + thought)."""
        try:
            tool_name = getattr(action, "tool", "") or ""
            tool_input = getattr(action, "tool_input", {})
            reasoning = getattr(action, "log", "") or ""

            action_data = {
                "action_name": tool_name,
                "action_args": tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)},
                "reasoning": reasoning,
            }

            key = str(run_id) if run_id else "default"
            self._pending_actions[key] = action_data
            self._latest_action = action_data
        except Exception as exc:
            if not self.fail_open:
                raise
            logger.exception("LongHorizonGuardCallback.on_agent_action failed open: %s", exc)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Fallback tool tracking in case on_agent_action was not emitted."""
        try:
            tool_name = (serialized or {}).get("name", "")
            key = str(run_id) if run_id else "default"
            if key not in self._pending_actions and not self._latest_action:
                self._latest_action = {
                    "action_name": tool_name,
                    "action_args": {"input": str(input_str)},
                    "reasoning": "",
                }
        except Exception as exc:
            if not self.fail_open:
                raise
            logger.exception("LongHorizonGuardCallback.on_tool_start failed open: %s", exc)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool execution finishes. Translates into on_step()."""
        try:
            parent_key = str(parent_run_id) if parent_run_id else None
            key = str(run_id) if run_id else None

            action_data = (
                self._pending_actions.pop(parent_key, None)
                or self._pending_actions.pop(key, None)
                or self._latest_action
                or {
                    "action_name": "unknown_tool",
                    "action_args": {},
                    "reasoning": "",
                }
            )

            step_record = {
                "step_index": self._step_counter,
                "reasoning": action_data.get("reasoning", ""),
                "action_name": action_data.get("action_name", ""),
                "action_args": action_data.get("action_args", {}),
                "tool_response": str(output),
            }

            step_res = self.guard.on_step(
                step_record=step_record,
                history=self.history,
                metadata={"run_id": str(run_id) if run_id else "unknown"},
            )

            self.history.append(step_record)
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

            return step_res
        except Exception as exc:
            if not self.fail_open:
                raise
            logger.exception("LongHorizonGuardCallback.on_tool_end failed open: %s", exc)
            return None

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool execution fails with an error."""
        return self.on_tool_end(
            output=f"Error: {error}",
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain completes. Top-level chain triggers on_run_end."""
        try:
            self._chain_depth = max(0, self._chain_depth - 1)
            if self._chain_depth == 0 or parent_run_id is None:
                self.last_run_summary = self.guard.on_run_end(
                    metadata={
                        "task": self._task_description,
                        "run_id": str(run_id) if run_id else "unknown",
                    },
                    trajectory={"steps": self.history},
                )
        except Exception as exc:
            if not self.fail_open:
                raise
            logger.exception("LongHorizonGuardCallback.on_chain_end failed open: %s", exc)
