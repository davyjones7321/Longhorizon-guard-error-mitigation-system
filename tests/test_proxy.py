"""Unit and integration tests for LongHorizon Guard HTTP API proxy server."""

import http.server
import json
import os
import threading
import urllib.request
from typing import Any, Dict, List, Optional
import pytest

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.proxy import run_proxy, GuardProxyHandler, SessionState


class MockUpstreamHandler(http.server.BaseHTTPRequestHandler):
    """Mock upstream LLM provider server."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [{"id": "gpt-4o"}]}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        req_body = self.rfile.read(content_length)
        parsed = json.loads(req_body.decode("utf-8"))

        messages = parsed.get("messages", [])
        last_msg = messages[-1] if messages else {}

        # If user asked to run a command, mock tool call
        if last_msg.get("role") == "user":
            resp_body = {
                "id": "chatcmpl-mock-1",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I will execute the command.",
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "pytest tests/"})
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
        elif last_msg.get("role") in ("tool", "function"):
            resp_body = {
                "id": "chatcmpl-mock-2",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "All tests passed successfully."
                    },
                    "finish_reason": "stop"
                }]
            }
        else:
            resp_body = {
                "id": "chatcmpl-mock-3",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done."}
                }]
            }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_body).encode("utf-8"))


@pytest.fixture(scope="module")
def mock_upstream_server():
    """Start mock upstream server on ephemeral port."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockUpstreamHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture
def proxy_server(mock_upstream_server):
    """Start Guard proxy server pointing to mock upstream."""
    flags_captured: List[Dict[str, Any]] = []

    def capture_flag(report: Dict[str, Any]) -> None:
        flags_captured.append(report)

    # Reset sessions before test
    GuardProxyHandler.sessions = {}
    guard = GuardInterface()
    server = run_proxy(
        host="127.0.0.1",
        port=0,
        upstream=mock_upstream_server,
        guard=guard,
        on_flag=capture_flag,
        fail_open=True,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {
        "url": f"http://127.0.0.1:{port}",
        "flags": flags_captured,
        "guard": guard,
    }
    server.shutdown()


def test_proxy_health_check(proxy_server):
    """Verify health check endpoint returns 200 OK."""
    req = urllib.request.Request(f"{proxy_server['url']}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("status") == "healthy"


def test_proxy_models_passthrough(proxy_server):
    """Verify GET /v1/models passes through to upstream."""
    req = urllib.request.Request(f"{proxy_server['url']}/v1/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "data" in data
        assert data["data"][0]["id"] == "gpt-4o"


def test_proxy_multi_turn_tool_calling_and_observation(proxy_server):
    """Verify proxy intercepts plan, tracks tool execution, and records history."""
    proxy_url = proxy_server["url"]
    session_id = "test-session-01"

    # Turn 1: User gives task description and numbered plan
    payload_turn_1 = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Task: Build CLI\n1. Write code\n2. Run pytest"}
        ]
    }
    req1 = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(payload_turn_1).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Session-ID": session_id},
        method="POST",
    )
    with urllib.request.urlopen(req1) as resp:
        assert resp.status == 200
        data1 = json.loads(resp.read().decode("utf-8"))
        assert "choices" in data1
        tool_call = data1["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "bash"

    # Check session state after Turn 1
    session = GuardProxyHandler.sessions[session_id]
    assert session._plan_proposed_called is True
    assert "call_123" in session._pending_tool_calls

    # Turn 2: Agent sends back the tool output
    payload_turn_2 = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Task: Build CLI\n1. Write code\n2. Run pytest"},
            {"role": "assistant", "content": "I will execute the command.", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_123", "name": "bash", "content": "1 passed in 0.05s"}
        ]
    }
    req2 = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(payload_turn_2).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Session-ID": session_id},
        method="POST",
    )
    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        data2 = json.loads(resp.read().decode("utf-8"))
        assert "choices" in data2

    # Verify that turn 2 created a step in Guard history
    assert len(session.history) == 1
    assert session.history[0]["action_name"] == "bash"
    assert session.history[0]["tool_response"] == "1 passed in 0.05s"


def test_proxy_action_repetition_flagging(proxy_server):
    """Verify that repeated identical actions trigger the on_flag callback."""
    session = SessionState(session_id="rep-test", guard=proxy_server["guard"], on_flag=lambda r: proxy_server["flags"].append(r))

    # Simulate 3 identical failing bash commands
    for i in range(3):
        session._pending_tool_calls[f"tc_{i}"] = {
            "reasoning": "retrying command",
            "name": "bash",
            "args": {"cmd": "npm test"},
        }
        session.process_messages_before_call([
            {"role": "tool", "tool_call_id": f"tc_{i}", "name": "bash", "content": "Error: exit code 1"}
        ])

    # 3 repetitions must trigger action repetition detector
    assert len(session.history) == 3
    assert len(proxy_server["flags"]) >= 1
    flagged = proxy_server["flags"][-1]
    assert flagged["flagged"] is True
    assert "repeated" in flagged["warning"].lower()


def test_proxy_fail_open_guarantee(mock_upstream_server):
    """Verify that if internal guard analysis raises an exception, the proxy still returns 200 OK."""
    class BrokenGuard:
        def on_plan_proposed(self, *args, **kwargs):
            raise RuntimeError("Corrupted pattern library database")
        def on_step(self, *args, **kwargs):
            raise RuntimeError("Drift monitor crashed")

    GuardProxyHandler.sessions = {}
    server = run_proxy(
        host="127.0.0.1",
        port=0,
        upstream=mock_upstream_server,
        guard=BrokenGuard(),
        fail_open=True,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "choices" in data
    finally:
        server.shutdown()


def test_proxy_session_logging_to_disk(tmp_path):
    """Verify that proxy automatically writes sanitized session JSONL records to disk."""
    log_dir = tmp_path / "test_logs"
    guard = GuardInterface()
    session = SessionState(
        session_id="log_disk_test",
        guard=guard,
        log_dir=str(log_dir),
    )

    # 1. Process plan with credential in user prompt
    session.process_messages_before_call([
        {"role": "user", "content": "Deploy app with api_key='sk-proj-123456789012345678901234567890'"}
    ])

    # 2. Process step
    session._pending_tool_calls["tc_1"] = {
        "reasoning": "checking files",
        "name": "ls",
        "args": {"path": "."},
    }
    session.process_messages_before_call([
        {"role": "tool", "tool_call_id": "tc_1", "name": "ls", "content": "file1.py"}
    ])

    # 3. Finalize session
    summary = session.finalize_session()
    assert summary is not None

    # Verify log file was written
    assert session.log_file_path is not None
    assert os.path.exists(session.log_file_path)

    with open(session.log_file_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 3
    assert lines[0]["type"] == "plan"
    assert lines[1]["type"] == "step"
    assert lines[2]["type"] == "summary"

    # Verify credential redaction
    assert "sk-proj-123456789012345678901234567890" not in open(session.log_file_path).read()
    assert "***REDACTED***" in lines[0]["task_description"]
