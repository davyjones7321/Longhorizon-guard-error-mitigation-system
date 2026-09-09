"""Unit and integration tests for LongHorizon Guard HTTP API proxy server."""

import gzip
import http.server
import json
import os
import threading
import time
import urllib.request
import zlib
from typing import Any, Dict, List, Optional
import pytest

from longhorizon_guard.interface import GuardInterface
from longhorizon_guard.proxy import (
    run_proxy,
    GuardProxyHandler,
    GuardProxyServer,
    SessionState,
    _decompress_response_body,
    _extract_filename_before,
    _extract_candidates_from_line,
    _is_placeholder_task,
    _is_substantive_task,
    _extract_task_description,
    _parse_text_edits,
    SSEAccumulator,
)


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

        raw_resp = json.dumps(resp_body).encode("utf-8")
        accept_encoding = self.headers.get("Accept-Encoding", "")
        if "gzip" in accept_encoding.lower():
            compressed = gzip.compress(raw_resp)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(compressed)))
            self.end_headers()
            self.wfile.write(compressed)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw_resp)))
            self.end_headers()
            self.wfile.write(raw_resp)


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

    # Simulate 4 identical failing bash commands (3 repetitions in history)
    for i in range(4):
        session._pending_tool_calls[f"tc_{i}"] = {
            "reasoning": "retrying command",
            "name": "bash",
            "args": {"cmd": "npm test"},
        }
        session.process_messages_before_call([
            {"role": "tool", "tool_call_id": f"tc_{i}", "name": "bash", "content": "Error: exit code 1"}
        ])

    # 3 repetitions in history must trigger action repetition detector
    assert len(session.history) == 4
    assert len(proxy_server["flags"]) >= 1
    flagged = proxy_server["flags"][-1]
    assert flagged["flagged"] is True
    assert "repeated" in flagged["warning"].lower()


def test_decompress_response_body_unit():
    """Unit test _decompress_response_body with gzip, deflate, and uncompressed payloads."""
    raw_text = b'{"status": "ok", "message": "hello world"}'

    # 1. Gzip compressed with header
    gzipped = gzip.compress(raw_text)
    assert _decompress_response_body(gzipped, content_encoding="gzip") == raw_text
    assert _decompress_response_body(gzipped, content_encoding="x-gzip") == raw_text

    # 2. Gzip compressed without header (magic bytes 0x1f 0x8b)
    assert _decompress_response_body(gzipped, content_encoding=None) == raw_text
    assert _decompress_response_body(gzipped, content_encoding="") == raw_text

    # 3. Deflate (zlib header)
    deflated = zlib.compress(raw_text)
    assert _decompress_response_body(deflated, content_encoding="deflate") == raw_text

    # 4. Raw deflate (no zlib header)
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw_deflated = compressor.compress(raw_text) + compressor.flush()
    assert _decompress_response_body(raw_deflated, content_encoding="deflate") == raw_text

    # 5. Plaintext / uncompressed
    assert _decompress_response_body(raw_text, content_encoding=None) == raw_text
    assert _decompress_response_body(raw_text, content_encoding="identity") == raw_text

    # 6. Empty body
    assert _decompress_response_body(b"") == b""


def test_proxy_gzip_compressed_response_handling(proxy_server, caplog):
    """Regression test: verify proxy observes gzip-compressed upstream responses without fail-open.

    Reproduces the bug where upstreams (e.g. Groq) send gzip bodies, causing
    UnicodeDecodeError during utf-8 decoding of raw bytes, silently skipping
    session.process_response_after_call and dropping all tool observation.
    """
    import logging
    caplog.set_level(logging.WARNING)

    proxy_url = proxy_server["url"]
    session_id = "test-gzip-session-01"

    # Turn 1: Client sends request with Accept-Encoding: gzip
    payload_turn_1 = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Run the test suite\n1. Run pytest"}
        ]
    }
    req1 = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(payload_turn_1).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
            "X-Session-ID": session_id,
        },
        method="POST",
    )

    with urllib.request.urlopen(req1) as resp:
        assert resp.status == 200
        # Client MUST receive the original gzip encoding and compressed bytes
        assert resp.headers.get("Content-Encoding") == "gzip"
        raw_client_body = resp.read()
        assert raw_client_body.startswith(b"\x1f\x8b")

        # Client decompressor succeeds
        client_decompressed = json.loads(gzip.decompress(raw_client_body).decode("utf-8"))
        assert "choices" in client_decompressed
        tool_call = client_decompressed["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "bash"

    # Verify Guard internally decoded the gzip body without hitting fail-open exception
    assert "Guard post-call observation failed open" not in caplog.text
    session = GuardProxyHandler.sessions[session_id]
    assert "call_123" in session._pending_tool_calls
    assert session._pending_tool_calls["call_123"]["name"] == "bash"

    # Turn 2: Agent executes the tool and submits tool response
    payload_turn_2 = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Run the test suite\n1. Run pytest"},
            {"role": "assistant", "content": "I will execute the command.", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_123", "name": "bash", "content": "120 passed in 10s"}
        ]
    }
    req2 = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(payload_turn_2).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
            "X-Session-ID": session_id,
        },
        method="POST",
    )

    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Encoding") == "gzip"
        raw_client_body2 = resp.read()
        client_decompressed2 = json.loads(gzip.decompress(raw_client_body2).decode("utf-8"))
        assert "choices" in client_decompressed2

    # Verify that Turn 2 successfully matched the pending tool call and recorded the step!
    assert len(session.history) == 1
    assert session.history[0]["action_name"] == "bash"
    assert session.history[0]["tool_response"] == "120 passed in 10s"
    assert session._step_counter == 1
    assert "Guard post-call observation failed open" not in caplog.text


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


def test_proxy_plan_heuristic_avoids_few_shot_boilerplate(tmp_path):
    """Verify that task description extraction skips Aider system few-shot examples."""
    guard = GuardInterface()
    session = SessionState(session_id="few_shot_test", guard=guard, log_dir=str(tmp_path))

    messages = [
        {
            "role": "system",
            "content": "You are an expert developer. To edit files, output SEARCH/REPLACE blocks.",
        },
        {
            "role": "user",
            "content": "Change the greeting to be more casual.",
        },
        {
            "role": "assistant",
            "content": "greeting.py\n<<<<<<< SEARCH\nprint('hello')\n=======\nprint('hey')\n>>>>>>> REPLACE",
        },
        {
            "role": "user",
            "content": "Build a Flask CRUD API with endpoints for GET and POST /items.",
        },
    ]

    session.process_messages_before_call(messages)
    assert session._task_description == "Build a Flask CRUD API with endpoints for GET and POST /items."
    assert "greeting" not in session._task_description


def test_proxy_plain_text_diff_observation(tmp_path):
    """Verify that proxy captures plain-text Aider SEARCH/REPLACE diffs as steps and logs summary."""
    log_dir = tmp_path / "aider_logs"
    guard = GuardInterface()
    session = SessionState(session_id="aider_test", guard=guard, log_dir=str(log_dir))

    # Turn 1: user task
    session.process_messages_before_call([
        {"role": "user", "content": "Create a Flask app with tests."}
    ])

    # Realistic Aider response containing multi-file SEARCH/REPLACE diff blocks
    aider_response = {
        "id": "chatcmpl-aider-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "I will create `app.py` and `tests/test_app.py`.\n\n"
                    "app.py\n"
                    "<<<<<<< SEARCH\n"
                    "=======\n"
                    "from flask import Flask, jsonify\n"
                    "app = Flask(__name__)\n\n"
                    "@app.route('/items')\n"
                    "def get_items():\n"
                    "    return jsonify([])\n"
                    ">>>>>>> REPLACE\n\n"
                    "tests/test_app.py\n"
                    "<<<<<<< SEARCH\n"
                    "=======\n"
                    "import pytest\n"
                    "from app import app\n\n"
                    "def test_get_items():\n"
                    "    client = app.test_client()\n"
                    "    assert client.get('/items').status_code == 200\n"
                    ">>>>>>> REPLACE\n"
                ),
            },
            "finish_reason": "stop",
        }],
    }

    # Process response
    session.process_response_after_call(aider_response)

    assert session.mode == "text_edits"
    assert len(session.history) == 2
    assert session.history[0]["action_name"] == "edit_file"
    assert session.history[0]["action_args"]["path"] == "app.py"
    assert session.history[0]["action_args"]["format"] == "diff"
    assert session.history[1]["action_name"] == "edit_file"
    assert session.history[1]["action_args"]["path"] == "tests/test_app.py"
    assert session.history[1]["action_args"]["format"] == "diff"

    # Verify log records: during the turn, no premature summary is written
    assert os.path.exists(session.log_file_path)
    with open(session.log_file_path, "r", encoding="utf-8") as f:
        records_during_turn = [json.loads(line) for line in f if line.strip()]

    assert len(records_during_turn) == 3
    assert records_during_turn[0]["type"] == "plan"
    assert records_during_turn[1]["type"] == "step"
    assert records_during_turn[1]["action_args"]["path"] == "app.py"
    assert records_during_turn[2]["type"] == "step"
    assert records_during_turn[2]["action_args"]["path"] == "tests/test_app.py"

    # Explicit session finalization produces the summary record
    session.finalize_session()
    with open(session.log_file_path, "r", encoding="utf-8") as f:
        records_after_finalize = [json.loads(line) for line in f if line.strip()]

    assert len(records_after_finalize) == 4
    assert records_after_finalize[3]["type"] == "summary"
    assert records_after_finalize[3]["total_steps"] == 2


def test_proxy_plain_text_whole_file_observation(tmp_path):
    """Verify that proxy captures plain-text whole file blocks as steps."""
    log_dir = tmp_path / "whole_logs"
    guard = GuardInterface()
    session = SessionState(session_id="whole_test", guard=guard, log_dir=str(log_dir))

    session.process_messages_before_call([
        {"role": "user", "content": "Write server.py"}
    ])

    response = {
        "id": "chatcmpl-whole-1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "Here is the complete file for `server.py`:\n\n"
                    "server.py\n"
                    "```python\n"
                    "import http.server\n"
                    "print('running')\n"
                    "```\n"
                ),
            },
            "finish_reason": "stop",
        }],
    }

    session.process_response_after_call(response)
    assert session.mode == "text_edits"
    assert len(session.history) == 1
    assert session.history[0]["action_name"] == "edit_file"
    assert session.history[0]["action_args"]["path"] == "server.py"
    assert session.history[0]["action_args"]["format"] == "whole"


def test_proxy_e2e_aider_flow_through_http(tmp_path):
    """End-to-end integration test of an Aider session proxied through HTTP."""
    class AiderMockHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_len)

            resp_payload = {
                "id": "chatcmpl-aider-e2e",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "I will update app.py:\n\n"
                            "app.py\n"
                            "<<<<<<< SEARCH\n"
                            "def old(): pass\n"
                            "=======\n"
                            "def new(): return True\n"
                            ">>>>>>> REPLACE\n"
                        ),
                    },
                    "finish_reason": "stop",
                }],
            }
            body = json.dumps(resp_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), AiderMockHandler)
    upstream_port = upstream_server.server_port
    t_upstream = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    t_upstream.start()

    log_dir = tmp_path / "e2e_aider_logs"
    proxy_server = run_proxy(
        host="127.0.0.1",
        port=0,
        upstream=f"http://127.0.0.1:{upstream_port}/v1",
        log_dir=str(log_dir),
    )
    proxy_port = proxy_server.server_port
    t_proxy = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    t_proxy.start()

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=json.dumps({
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "Diff editing format instructions."},
                    {"role": "user", "content": "Change greeting (few shot)"},
                    {"role": "assistant", "content": "greeting.py\n<<<<<<< SEARCH\n..."},
                    {"role": "user", "content": "Implement new function in app.py"},
                ],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Session-ID": "aider_http_session"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "choices" in data

        # Check session log file during turn: plan + step (no premature summary)
        log_files = list(log_dir.glob("session_*_aider_http_session.jsonl"))
        assert len(log_files) == 1
        with open(log_files[0], "r", encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]

        assert len(entries) == 2
        assert entries[0]["type"] == "plan"
        assert entries[0]["task_description"] == "Implement new function in app.py"
        assert entries[1]["type"] == "step"
        assert entries[1]["action_args"]["path"] == "app.py"

        # Explicitly finalize via HTTP endpoint
        finalize_req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/session/finalize",
            headers={"X-Session-ID": "aider_http_session"},
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(finalize_req) as fin_resp:
            assert fin_resp.status == 200
            fin_data = json.loads(fin_resp.read().decode("utf-8"))
            assert fin_data.get("status") == "finalized"

        with open(log_files[0], "r", encoding="utf-8") as f:
            final_entries = [json.loads(l) for l in f if l.strip()]

        assert len(final_entries) == 3
        assert final_entries[2]["type"] == "summary"
        assert final_entries[2]["total_steps"] == 1
    finally:
        proxy_server.shutdown()
        upstream_server.shutdown()


def test_gap1_sentence_embedded_filename_extraction():
    """Gap 1 Regression: Verify filename extraction succeeds when the filename is embedded mid-sentence.

    Previously, scanning only candidate words or anchoring with $ at the end of the line caused:
    'Let's go ahead and modify the file app.py to fix this issue:' to return unknown_file.
    """
    # 1. Filename embedded mid-sentence before trailing prose
    text1 = "Let's go ahead and modify the file app.py to fix this issue:"
    candidates1 = _extract_candidates_from_line(text1)
    assert any(c[0] == "app.py" for c in candidates1)
    extracted1 = _extract_filename_before(text1 + "\n<<<<<<< SEARCH", len(text1) + 1)
    assert extracted1 == "app.py"

    # 2. Delimiter preference: colon/end-of-line candidate preferred over mid-sentence candidate
    text2 = "Compare old_version.py with new_version.py:"
    extracted2 = _extract_filename_before(text2 + "\n<<<<<<< SEARCH", len(text2) + 1)
    assert extracted2 == "new_version.py"

    # 3. Backtick-wrapped path
    text3 = "Here is the diff for `src/models/user.py` to add validation."
    extracted3 = _extract_filename_before(text3 + "\n<<<<<<< SEARCH", len(text3) + 1)
    assert extracted3 == "src/models/user.py"

    # 4. Ignore false positives: Python version 3.10, e.g., i.e.
    text4 = "Using Python 3.10 (e.g. standard library), modify `config.py`:"
    extracted4 = _extract_filename_before(text4 + "\n<<<<<<< SEARCH", len(text4) + 1)
    assert extracted4 == "config.py"

    # 5. Full parse_text_edits test with embedded sentence
    block = (
        "Let's go ahead and modify the file app.py to fix this issue:\n"
        "<<<<<<< SEARCH\n"
        "def run(): return False\n"
        "=======\n"
        "def run(): return True\n"
        ">>>>>>> REPLACE"
    )
    edits = _parse_text_edits(block)
    assert len(edits) == 1
    assert edits[0]["path"] == "app.py"
    assert edits[0]["format"] == "diff"


def test_gap2_placeholder_handshake_delatching(tmp_path):
    """Gap 2 Regression: Verify session updates task description when upgrading from placeholder handshake.

    Previously, _plan_proposed_called latched permanently True on Request 1 containing
    Aider's placeholder 'I am not sharing any files that you can edit yet.', ignoring the
    substantive task that arrived in Request 2.
    """
    log_dir = tmp_path / "gap2_logs"
    guard = GuardInterface()
    session = SessionState(session_id="gap2_session", guard=guard, log_dir=str(log_dir))

    # Turn 1: Aider handshake without files
    handshake_messages = [
        {"role": "user", "content": "I am not sharing any files that you can edit yet."}
    ]
    assert _is_placeholder_task("I am not sharing any files that you can edit yet.") is True
    session.process_messages_before_call(handshake_messages)

    # Handshake did not lock in substantive plan proposed
    assert session._plan_proposed_called is False
    assert session._task_description == "I am not sharing any files that you can edit yet."

    # Turn 2: Substantive task arrives
    real_task = "Build a Flask CRUD API with endpoints for GET and POST /items."
    turn2_messages = [
        {"role": "user", "content": "I am not sharing any files that you can edit yet."},
        {"role": "assistant", "content": "Understood. Please let me know what you'd like me to work on."},
        {"role": "user", "content": real_task},
    ]
    session.process_messages_before_call(turn2_messages)

    # The plan should have updated to the real task
    assert session._plan_proposed_called is True
    assert session._task_description == real_task

    # Check JSONL log file: verify plan entry records the substantive task
    assert session.log_file_path is not None
    assert os.path.exists(session.log_file_path)
    with open(session.log_file_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    plan_records = [r for r in records if r["type"] == "plan"]
    assert len(plan_records) == 2
    assert plan_records[1]["task_description"] == real_task


def test_gap3_multi_turn_does_not_finalize_per_turn(tmp_path):
    """Gap 3 Regression: Verify multi-turn sessions do not write premature summary records per turn.

    Previously, finish_reason == 'stop' triggered finalize_session() and on_run_end() after
    every turn, polluting the transcript with premature intermediate summary entries.
    """
    log_dir = tmp_path / "gap3_logs"
    guard = GuardInterface()
    session = SessionState(session_id="gap3_session", guard=guard, log_dir=str(log_dir), idle_timeout=0.0)

    session.process_messages_before_call([
        {"role": "user", "content": "Implement user service"}
    ])

    # Turn 1
    resp1 = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "user.py\n<<<<<<< SEARCH\npass\n=======\nclass User: pass\n>>>>>>> REPLACE",
            },
            "finish_reason": "stop",
        }]
    }
    session.process_response_after_call(resp1)

    # Verify no summary written after Turn 1
    with open(session.log_file_path, "r", encoding="utf-8") as f:
        turn1_entries = [json.loads(l) for l in f if l.strip()]
    assert len(turn1_entries) == 2  # 1 plan + 1 step
    assert not any(e["type"] == "summary" for e in turn1_entries)

    # Turn 2
    resp2 = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "test_user.py\n<<<<<<< SEARCH\npass\n=======\ndef test_user(): pass\n>>>>>>> REPLACE",
            },
            "finish_reason": "stop",
        }]
    }
    session.process_response_after_call(resp2)

    # Verify no summary written after Turn 2
    with open(session.log_file_path, "r", encoding="utf-8") as f:
        turn2_entries = [json.loads(l) for l in f if l.strip()]
    assert len(turn2_entries) == 3  # 1 plan + 2 steps
    assert not any(e["type"] == "summary" for e in turn2_entries)

    # Now finalize explicitly (e.g. on shutdown, idle timeout, or explicit API call)
    summary = session.finalize_session()
    assert summary is not None

    with open(session.log_file_path, "r", encoding="utf-8") as f:
        final_entries = [json.loads(l) for l in f if l.strip()]

    assert len(final_entries) == 4  # 1 plan + 2 steps + 1 summary
    summary_entries = [e for e in final_entries if e["type"] == "summary"]
    assert len(summary_entries) == 1
    assert summary_entries[0]["total_steps"] == 2

    # Verify idempotent finalize (does not write duplicate summary entries)
    session.finalize_session()
    with open(session.log_file_path, "r", encoding="utf-8") as f:
        re_entries = [json.loads(l) for l in f if l.strip()]
    assert len(re_entries) == 4


def test_sse_accumulator_unit():
    """Unit test SSEAccumulator across multi-chunk delta events."""
    accumulator = SSEAccumulator()

    # Feeding partial line fragments across feed_bytes calls
    accumulator.feed_bytes(b": keepalive comment\n")
    accumulator.feed_bytes(b'data: {"id": "chatcmpl-unit-1", "model": "llama-3.3-70b", ')
    accumulator.feed_bytes(b'"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]}\n\n')
    accumulator.feed_bytes(b'data: {"choices": [{"index": 0, "delta": {"content": " world", "reasoning_content": "think step 1"}, "finish_reason": null}]}\n\n')
    accumulator.feed_bytes(b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n')
    accumulator.feed_bytes(b'data: [DONE]\n\n')

    reconstructed = accumulator.finalize()
    assert reconstructed["id"] == "chatcmpl-unit-1"
    assert reconstructed["model"] == "llama-3.3-70b"
    choice = reconstructed["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello world"
    assert choice["message"]["reasoning_content"] == "think step 1"


def test_proxy_streaming_sse_realtime_forwarding_and_observation(tmp_path):
    """Verify that proxy forwards SSE chunks to client in real time and observes edits.

    Regression test for: proxy buffered or ignored responses when Content-Type was
    text/event-stream, causing real coding assistants like Aider with stream: true to
    have 0 steps observed.
    """
    class StreamingUpstreamHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Multi-chunk Aider-style SSE events with intentional delay
            events = [
                b'data: {"id":"chatcmpl-stream-aider","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-stream-aider","choices":[{"index":0,"delta":{"content":"I will create hello.py:\\n\\n"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-stream-aider","choices":[{"index":0,"delta":{"content":"hello.py\\n<<<<<<< SEARCH\\n"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-stream-aider","choices":[{"index":0,"delta":{"content":"=======\\nprint(\\"hello world\\")\\n"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-stream-aider","choices":[{"index":0,"delta":{"content":">>>>>>> REPLACE\\n"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-stream-aider","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            for ev in events:
                chunk = f"{len(ev):X}\r\n".encode("ascii") + ev + b"\r\n"
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.03)

            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    upstream_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamingUpstreamHandler)
    upstream_port = upstream_srv.server_port
    t_up = threading.Thread(target=upstream_srv.serve_forever, daemon=True)
    t_up.start()

    log_dir = tmp_path / "stream_logs"
    proxy_server = run_proxy(
        host="127.0.0.1",
        port=0,
        upstream=f"http://127.0.0.1:{upstream_port}/v1",
        log_dir=str(log_dir),
        idle_timeout=0.0,
    )
    proxy_port = proxy_server.server_port
    t_px = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    t_px.start()

    try:
        session_id = "aider_streaming_session"
        req_body = json.dumps({
            "model": "gpt-4o",
            "stream": True,
            "messages": [
                {"role": "user", "content": "Create hello.py that prints hello world"},
            ]
        }).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=req_body,
            headers={
                "Content-Type": "application/json",
                "X-Session-ID": session_id,
            },
            method="POST",
        )

        received_lines = []
        t0 = time.time()
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")
            while True:
                line = resp.readline()
                if not line:
                    break
                received_lines.append((time.time() - t0, line))

        # (a) Verify real-time incremental arrival:
        # Non-empty lines should be delivered across time, not all at t=0
        data_lines = [l for l in received_lines if l[1].startswith(b"data:")]
        assert len(data_lines) == 7
        total_stream_time = data_lines[-1][0] - data_lines[0][0]
        assert total_stream_time >= 0.10, f"Stream completed too fast ({total_stream_time}s); buffering suspected"

        # (b) Verify client received exact original streaming bytes
        assert any(b"hello.py" in l[1] for l in data_lines)
        assert data_lines[-1][1].strip() == b"data: [DONE]"

        # Wait briefly for proxy server thread to finalize observation
        import time as _t
        _t.sleep(0.1)

        # (c) Verify guard session recorded the reconstructed edit
        session = GuardProxyHandler.sessions.get(session_id)
        assert session is not None
        assert session.mode == "text_edits"
        assert len(session.history) == 1
        step = session.history[0]
        assert step["action_name"] == "edit_file"
        assert step["action_args"]["path"] == "hello.py"
        assert step["action_args"]["format"] == "diff"

        # (d) Verify log file on disk
        log_files = list(log_dir.glob(f"session_*_{session_id}.jsonl"))
        assert len(log_files) == 1
        with open(log_files[0], "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) >= 2
        assert entries[0]["type"] == "plan"
        assert entries[1]["type"] == "step"
        assert entries[1]["action_args"]["path"] == "hello.py"
    finally:
        proxy_server.shutdown()
        upstream_srv.shutdown()


def test_proxy_streaming_sse_tool_calls_observation(tmp_path):
    """Verify that proxy reconstructs streaming tool_calls chunks into a complete tool step."""
    class StreamingToolCallUpstream(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            events = [
                b'data: {"id":"chatcmpl-tc-stream","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"write_file","arguments":""}}]},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-tc-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\": "}}]},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-tc-stream","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"main.py\\", \\"content\\": \\"import os\\"}"}}]},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-tc-stream","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            for ev in events:
                chunk = f"{len(ev):X}\r\n".encode("ascii") + ev + b"\r\n"
                self.wfile.write(chunk)
                self.wfile.flush()

            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    upstream_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamingToolCallUpstream)
    upstream_port = upstream_srv.server_port
    t_up = threading.Thread(target=upstream_srv.serve_forever, daemon=True)
    t_up.start()

    log_dir = tmp_path / "tool_stream_logs"
    proxy_server = run_proxy(
        host="127.0.0.1",
        port=0,
        upstream=f"http://127.0.0.1:{upstream_port}/v1",
        log_dir=str(log_dir),
        idle_timeout=0.0,
    )
    proxy_port = proxy_server.server_port
    t_px = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    t_px.start()

    try:
        session_id = "tool_streaming_session"
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=json.dumps({"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "Write code"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Session-ID": session_id},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            resp.read()

        import time as _t
        _t.sleep(0.1)

        session = GuardProxyHandler.sessions.get(session_id)
        assert session is not None
        assert session.mode == "tool_calls"
        assert "call_abc" in session._pending_tool_calls
        pending = session._pending_tool_calls["call_abc"]
        assert pending["name"] == "write_file"
        assert pending["args"]["path"] == "main.py"
        assert pending["args"]["content"] == "import os"
    finally:
        proxy_server.shutdown()
        upstream_srv.shutdown()


def test_proxy_streaming_sse_gzip_compressed_stream(tmp_path):
    """Verify that proxy correctly forwards and observes SSE streams that are gzip-compressed by upstream."""
    class GzipStreamingUpstream(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            events = [
                b'data: {"id":"chatcmpl-gzip-stream","choices":[{"index":0,"delta":{"content":"server.py\\n```python\\nprint(\'ok\')\\n```"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-gzip-stream","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
            for ev in events:
                comp = compressor.compress(ev)
                if comp:
                    chunk = f"{len(comp):X}\r\n".encode("ascii") + comp + b"\r\n"
                    self.wfile.write(chunk)
                    self.wfile.flush()
            tail = compressor.flush()
            if tail:
                chunk = f"{len(tail):X}\r\n".encode("ascii") + tail + b"\r\n"
                self.wfile.write(chunk)
                self.wfile.flush()

            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    upstream_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), GzipStreamingUpstream)
    upstream_port = upstream_srv.server_port
    t_up = threading.Thread(target=upstream_srv.serve_forever, daemon=True)
    t_up.start()

    log_dir = tmp_path / "gzip_stream_logs"
    proxy_server = run_proxy(
        host="127.0.0.1",
        port=0,
        upstream=f"http://127.0.0.1:{upstream_port}/v1",
        log_dir=str(log_dir),
        idle_timeout=0.0,
    )
    proxy_port = proxy_server.server_port
    t_px = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    t_px.start()

    try:
        session_id = "gzip_streaming_session"
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=json.dumps({"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "Write server.py"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Session-ID": session_id, "Accept-Encoding": "gzip"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            # Client receives compressed body
            assert resp.headers.get("Content-Encoding") == "gzip"
            raw_data = resp.read()
            decompressed = gzip.decompress(raw_data).decode("utf-8")
            assert "server.py" in decompressed

        import time as _t
        _t.sleep(0.1)

        session = GuardProxyHandler.sessions.get(session_id)
        assert session is not None
        assert session.mode == "text_edits"
        assert len(session.history) == 1
        assert session.history[0]["action_args"]["path"] == "server.py"
        assert session.history[0]["action_args"]["format"] == "whole"
    finally:
        proxy_server.shutdown()
        upstream_srv.shutdown()


def test_sse_accumulator_multibyte_utf8_split():
    """Regression test: verify feed_bytes preserves multi-byte UTF-8 sequences split across chunks.

    Previously, feed_bytes called chunk.decode('utf-8', errors='replace') on each raw byte chunk
    independently. If a socket read split in the middle of a 2-byte, 3-byte, or 4-byte character,
    both fragments decoded as '\ufffd' replacement characters.
    """
    # 1. 2-byte UTF-8 character 'é' (0xC3 0xA9)
    acc_2byte = SSEAccumulator()
    raw_2byte = b'data: {"choices":[{"delta":{"content":"caf\xc3\xa9 done"}}]}\n\n'
    split_pos_2 = raw_2byte.index(b"\xc3") + 1  # Between 0xC3 and 0xA9
    part_2a = raw_2byte[:split_pos_2]
    part_2b = raw_2byte[split_pos_2:]

    acc_2byte.feed_bytes(part_2a)
    acc_2byte.feed_bytes(part_2b)
    res_2byte = acc_2byte.finalize()
    content_2byte = res_2byte["choices"][0]["message"]["content"]
    assert content_2byte == "café done"
    assert "\ufffd" not in content_2byte

    # 2. 3-byte UTF-8 character '—' (em-dash: 0xE2 0x80 0x94)
    acc_3byte = SSEAccumulator()
    raw_3byte = "data: {\"choices\":[{\"delta\":{\"content\":\"start \u2014 end\"}}]}\n\n".encode("utf-8")
    split_pos_3 = raw_3byte.index(b"\xe2") + 2  # After 0xE2 0x80, before 0x94
    part_3a = raw_3byte[:split_pos_3]
    part_3b = raw_3byte[split_pos_3:]

    acc_3byte.feed_bytes(part_3a)
    acc_3byte.feed_bytes(part_3b)
    res_3byte = acc_3byte.finalize()
    content_3byte = res_3byte["choices"][0]["message"]["content"]
    assert content_3byte == "start — end"
    assert "\ufffd" not in content_3byte

    # 3. 4-byte UTF-8 character '🚀' (emoji: 0xF0 0x9F 0x9A 0x80)
    acc_4byte = SSEAccumulator()
    raw_4byte = "data: {\"choices\":[{\"delta\":{\"content\":\"Launch \U0001f680 now\"}}]}\n\n".encode("utf-8")
    split_pos_4 = raw_4byte.index(b"\xf0") + 2  # After 0xF0 0x9F, before 0x9A 0x80
    part_4a = raw_4byte[:split_pos_4]
    part_4b = raw_4byte[split_pos_4:]

    acc_4byte.feed_bytes(part_4a)
    acc_4byte.feed_bytes(part_4b)
    res_4byte = acc_4byte.finalize()
    content_4byte = res_4byte["choices"][0]["message"]["content"]
    assert content_4byte == "Launch 🚀 now"
    assert "\ufffd" not in content_4byte


