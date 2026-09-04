"""Comprehensive tests for LongHorizon Guard integrations (LangChain callback & OpenAI wrapper)."""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.integrations.langchain_callback import LongHorizonGuardCallback
from longhorizon_guard.integrations.client_wrapper import wrap_guard, WrappedOpenAIClient


# =========================================================================
# Mock Objects for OpenAI SDK compatibility
# =========================================================================

@dataclass
class FakeFunction:
    name: str
    arguments: str

@dataclass
class FakeToolCall:
    id: str
    type: str
    function: FakeFunction

@dataclass
class FakeMessage:
    role: str
    content: Optional[str]
    tool_calls: Optional[List[FakeToolCall]] = None
    reasoning_content: Optional[str] = None

@dataclass
class FakeChoice:
    index: int
    message: FakeMessage
    finish_reason: str = "stop"

@dataclass
class FakeChatCompletion:
    id: str
    choices: List[FakeChoice]
    model: str = "gpt-4o"


class FakeCompletions:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs})
        if not self._responses:
            raise RuntimeError("No more fake responses queued")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, responses: List[Any]):
        self.completions = FakeCompletions(responses)
        self.chat = FakeChat(self.completions)
        self.custom_property = "unwrapped_client_value"

    def custom_method(self) -> str:
        return "custom_method_output"


# =========================================================================
# Mock Objects for LangChain AgentAction compatibility
# =========================================================================

class FakeAgentAction:
    def __init__(self, tool: str, tool_input: Any, log: str):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log


# =========================================================================
# PATH A TESTS: LangChain Callback Handler
# =========================================================================

class TestLangChainCallbackHandler:
    """Verify LongHorizonGuardCallback lifecycle and schema translation."""

    def test_instantiation_default_and_custom(self):
        cb_default = LongHorizonGuardCallback()
        assert isinstance(cb_default.guard, GuardInterface)
        assert cb_default.history == []

        custom_guard = GuardInterface()
        cb_custom = LongHorizonGuardCallback(guard=custom_guard, fail_open=False)
        assert cb_custom.guard is custom_guard
        assert cb_custom.fail_open is False

    def test_full_lifecycle_agent_run_translation(self):
        """Simulate on_chain_start -> on_agent_action -> on_tool_end -> on_chain_end."""
        guard = GuardInterface()
        callback = LongHorizonGuardCallback(guard=guard)

        # 1. Start chain
        run_id_chain = uuid.uuid4()
        callback.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={
                "input": "Find latest stock price for Apple",
                "plan": "1. Search web for stock price\n2. Return value",
            },
            run_id=run_id_chain,
        )

        assert callback._task_description == "Find latest stock price for Apple"
        assert callback._proposed_plan == "1. Search web for stock price\n2. Return value"

        # 2. Step 1: Agent decides to search
        action_run_id = uuid.uuid4()
        tool_run_id = uuid.uuid4()
        action1 = FakeAgentAction(
            tool="web_search",
            tool_input={"query": "Apple stock price today"},
            log="I should search for Apple's current stock price.",
        )
        callback.on_agent_action(action1, run_id=action_run_id)
        step_res1 = callback.on_tool_end(
            output="AAPL is trading at $225.50",
            run_id=tool_run_id,
            parent_run_id=action_run_id,
        )

        assert step_res1 is not None
        assert "continue_execution" in step_res1
        assert len(callback.history) == 1

        # Verify exact schema of step_record passed into GuardInterface
        record1 = callback.history[0]
        assert record1["step_index"] == 0
        assert record1["reasoning"] == "I should search for Apple's current stock price."
        assert record1["action_name"] == "web_search"
        assert record1["action_args"] == {"query": "Apple stock price today"}
        assert record1["tool_response"] == "AAPL is trading at $225.50"

        # 3. Step 2: Tool failure event
        action_run_id_2 = uuid.uuid4()
        tool_run_id_2 = uuid.uuid4()
        action2 = FakeAgentAction(
            tool="finance_api",
            tool_input={"ticker": "AAPL"},
            log="Verifying with finance API.",
        )
        callback.on_agent_action(action2, run_id=action_run_id_2)
        callback.on_tool_error(
            error=ConnectionError("API rate limit exceeded"),
            run_id=tool_run_id_2,
            parent_run_id=action_run_id_2,
        )

        assert len(callback.history) == 2
        record2 = callback.history[1]
        assert record2["step_index"] == 1
        assert record2["action_name"] == "finance_api"
        assert "Error: API rate limit exceeded" in record2["tool_response"]

        # 4. Chain finishes
        callback.on_chain_end(
            outputs={"output": "Apple is at $225.50"},
            run_id=run_id_chain,
        )

        assert callback.last_run_summary is not None
        assert callback.last_run_summary["processed"] is True
        assert "subgoals_summary" in callback.last_run_summary

    def test_on_flag_callback_invocation(self):
        """Verify on_flag callback is executed when a step is flagged."""
        flag_events = []

        def handle_flag(res):
            flag_events.append(res)

        guard = GuardInterface()
        callback = LongHorizonGuardCallback(guard=guard, on_flag=handle_flag)

        # Trigger a flagged action (e.g. repeated Nothing happens loop)
        action_id_1 = uuid.uuid4()
        callback.on_agent_action(
            FakeAgentAction(tool="'go", tool_input={}, log="Walking"),
            run_id=action_id_1,
        )
        callback.on_tool_end("Nothing happens.", run_id=uuid.uuid4(), parent_run_id=action_id_1)

        action_id_2 = uuid.uuid4()
        callback.on_agent_action(
            FakeAgentAction(tool="'go", tool_input={}, log="Walking again"),
            run_id=action_id_2,
        )
        callback.on_tool_end("Nothing happens.", run_id=uuid.uuid4(), parent_run_id=action_id_2)

        action_id_3 = uuid.uuid4()
        callback.on_agent_action(
            FakeAgentAction(tool="'go", tool_input={}, log="Walking third time"),
            run_id=action_id_3,
        )
        callback.on_tool_end("Nothing happens.", run_id=uuid.uuid4(), parent_run_id=action_id_3)

        # 3rd identical 'Nothing happens' triggers structural loop rule
        assert len(flag_events) > 0
        assert flag_events[-1]["flagged"] is True

    def test_fail_open_guarantee(self, caplog):
        """Verify internal exceptions never escape when fail_open=True."""
        guard = GuardInterface()
        callback = LongHorizonGuardCallback(guard=guard, fail_open=True)

        with patch.object(guard, "on_step", side_effect=RuntimeError("Simulated guard failure")):
            with caplog.at_level(logging.ERROR):
                res = callback.on_tool_end("test output", run_id=uuid.uuid4())
                assert res is None  # Fails open without raising


# =========================================================================
# PATH B TESTS: OpenAI Client Wrapper Middleware
# =========================================================================

class TestOpenAIClientWrapper:
    """Verify wrap_guard() middleware intercepts and observes multi-turn chats."""

    def test_passthrough_attributes_and_methods(self):
        fake_client = FakeOpenAIClient([])
        wrapped = wrap_guard(fake_client)

        assert wrapped.custom_property == "unwrapped_client_value"
        assert wrapped.custom_method() == "custom_method_output"
        assert isinstance(wrapped.guard, GuardInterface)

    def test_multi_turn_tool_calling_observation(self):
        """Simulate realistic multi-turn tool calling:
        Turn 1: User prompt -> Assistant calls tool 'search_database'
        Turn 2: Tool result passed -> Assistant answers final answer
        """
        # Prepare mock responses
        turn1_response = FakeChatCompletion(
            id="chatcmpl-1",
            choices=[
                FakeChoice(
                    index=0,
                    message=FakeMessage(
                        role="assistant",
                        content="I need to query the employee database.",
                        reasoning_content="Querying the database for user 42.",
                        tool_calls=[
                            FakeToolCall(
                                id="call_abc123",
                                type="function",
                                function=FakeFunction(
                                    name="db_query",
                                    arguments=json.dumps({"user_id": 42, "fields": ["name", "salary"]}),
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        turn2_response = FakeChatCompletion(
            id="chatcmpl-2",
            choices=[
                FakeChoice(
                    index=0,
                    message=FakeMessage(
                        role="assistant",
                        content="Employee 42 is Alice with salary $120,000.",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )

        fake_client = FakeOpenAIClient([turn1_response, turn2_response])
        wrapped = wrap_guard(fake_client)

        # ---- Turn 1: Host sends initial prompt ----
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fetch details for employee 42."},
        ]

        resp1 = wrapped.chat.completions.create(model="gpt-4o", messages=messages)

        # Assert response is unmodified from real client
        assert resp1 is turn1_response
        assert resp1.choices[0].message.tool_calls[0].function.name == "db_query"
        # At turn 1, step record is pending until tool observation returns
        assert len(wrapped.history) == 0

        # ---- Turn 2: Host sends tool execution result ----
        tool_call = resp1.choices[0].message.tool_calls[0]
        messages.extend([
            {
                "role": "assistant",
                "content": resp1.choices[0].message.content,
                "tool_calls": resp1.choices[0].message.tool_calls,
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"name": "Alice", "salary": 120000}),
            },
        ])

        resp2 = wrapped.chat.completions.create(model="gpt-4o", messages=messages)

        # Assert response 2 is unmodified
        assert resp2 is turn2_response
        assert resp2.choices[0].message.content == "Employee 42 is Alice with salary $120,000."

        # Verify step_record was automatically constructed and passed to GuardInterface
        assert len(wrapped.history) == 1
        record = wrapped.history[0]
        assert record["step_index"] == 0
        assert record["reasoning"] == "Querying the database for user 42."
        assert record["action_name"] == "db_query"
        assert record["action_args"] == {"user_id": 42, "fields": ["name", "salary"]}
        assert record["tool_response"] == json.dumps({"name": "Alice", "salary": 120000})

        # ---- Finalize Run ----
        summary = wrapped.finalize_run(metadata={"task": "Employee query"})
        assert summary["processed"] is True
        assert summary["subgoals_summary"] is not None
        assert wrapped.last_run_summary is summary

    def test_client_api_exception_propagates_cleanly(self):
        """Real client exceptions (e.g. rate limit / network) must propagate unaltered to the caller."""
        api_error = ConnectionResetError("Connection lost to API")
        fake_client = FakeOpenAIClient([api_error])
        wrapped = wrap_guard(fake_client)

        with pytest.raises(ConnectionResetError, match="Connection lost to API"):
            wrapped.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    def test_fail_open_when_guard_crashes(self, caplog):
        """If GuardInterface throws inside create(), the real API response is STILL returned."""
        turn_response = FakeChatCompletion(
            id="chatcmpl-3",
            choices=[FakeChoice(index=0, message=FakeMessage(role="assistant", content="Hello!"))],
        )
        fake_client = FakeOpenAIClient([turn_response])
        guard = GuardInterface()
        wrapped = wrap_guard(fake_client, guard=guard, fail_open=True)

        with patch.object(guard, "on_plan_proposed", side_effect=RuntimeError("Matcher internal crash")):
            with caplog.at_level(logging.ERROR):
                resp = wrapped.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hello"}],
                )
                # The real API response must be returned regardless of guard failure
                assert resp is turn_response
