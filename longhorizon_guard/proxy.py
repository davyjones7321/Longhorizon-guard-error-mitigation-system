"""HTTP API Proxy Server for LongHorizon Guard.

Transparently proxies chat completion requests between coding assistants
(OpenCode, Claude Code, Cursor, Aider, Codex) and upstream LLM providers,
intercepting tool calls and evaluating trajectories in real time.
"""

import argparse
import http.server
import json
import logging
import sys
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Set

from longhorizon_guard.interface import GuardInterface

logger = logging.getLogger("longhorizon_guard.proxy")


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


class SessionState:
    """Tracks trajectory state for an active agent session."""

    def __init__(self, session_id: str, guard: GuardInterface, on_flag: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.session_id = session_id
        self.guard = guard
        self.on_flag = on_flag
        self.history: List[Dict[str, Any]] = []
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._processed_tool_ids: Set[str] = set()
        self._step_counter: int = 0
        self._plan_proposed_called: bool = False
        self._task_description: str = ""

    def process_messages_before_call(self, messages: List[Dict[str, Any]]) -> None:
        """Inspect request messages for initial plan and prior tool execution outputs."""
        if not messages or not isinstance(messages, (list, tuple)):
            return

        # 1. First interaction plan detection
        if not self._plan_proposed_called:
            task_desc = ""
            plan_text = ""
            for msg in messages:
                role = msg.get("role") if isinstance(msg, dict) else None
                content = msg.get("content") if isinstance(msg, dict) else ""
                if role == "user" and not task_desc:
                    task_desc = str(content or "")
                elif role == "system" and not task_desc and "task" in str(content or "").lower():
                    task_desc = str(content or "")

            if task_desc:
                self._task_description = task_desc
                self.guard.on_plan_proposed(
                    task_description=task_desc,
                    proposed_plan=plan_text,
                    metadata={"source": "guard_proxy", "session_id": self.session_id},
                )
                self._plan_proposed_called = True

        # 2. Correlate tool responses from previous agent action
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role in ("tool", "function"):
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
                    metadata={"source": "guard_proxy", "session_id": self.session_id},
                )

                self.history.append(step_record)
                self._processed_tool_ids.add(str(tool_call_id))
                self._step_counter += 1

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
        """Capture assistant thought and proposed tool calls from response."""
        choices = response_body.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        reasoning = message.get("reasoning_content") or content

        tool_calls = message.get("tool_calls", [])
        if tool_calls and isinstance(tool_calls, list):
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


class GuardProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler that routes requests to upstream and inspects agent trajectories."""

    upstream_base_url: str = "https://api.openai.com/v1"
    guard: Optional[GuardInterface] = None
    on_flag: Optional[Callable[[Dict[str, Any]], None]] = None
    fail_open: bool = True
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
                self.sessions[session_id] = SessionState(session_id=session_id, guard=g, on_flag=self.on_flag)
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
        is_chat = path.endswith("/chat/completions")

        parsed_json: Optional[Dict[str, Any]] = None
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
        if upstream_base.endswith("/v1") and path.startswith("/v1/"):
            target_url = upstream_base + path[3:]
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
                        if "application/json" in content_type:
                            resp_json = json.loads(resp_body.decode("utf-8"))
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


def run_proxy(
    host: str = "127.0.0.1",
    port: int = 8000,
    upstream: str = "https://api.openai.com/v1",
    guard: Optional[GuardInterface] = None,
    on_flag: Optional[Callable[[Dict[str, Any]], None]] = None,
    fail_open: bool = True,
) -> http.server.ThreadingHTTPServer:
    """Launch the LongHorizon Guard HTTP API proxy server.

    Args:
        host: Host interface to bind to (default '127.0.0.1').
        port: Port to listen on (default 8000).
        upstream: Upstream LLM provider base URL (default 'https://api.openai.com/v1').
        guard: Optional GuardInterface instance.
        on_flag: Optional callback function triggered on detected error flags.
        fail_open: Whether to allow requests through if internal guard analysis fails.

    Returns:
        The running ThreadingHTTPServer instance.
    """
    GuardProxyHandler.upstream_base_url = upstream
    GuardProxyHandler.guard = guard if guard is not None else GuardInterface()
    GuardProxyHandler.on_flag = on_flag
    GuardProxyHandler.fail_open = fail_open

    server = http.server.ThreadingHTTPServer((host, port), GuardProxyHandler)
    return server


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
        "--no-fail-open",
        action="store_true",
        help="Raise internal guard exceptions instead of failing open",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"🛡️  LongHorizon Guard Real-Time API Proxy Running")
    print(f"   Listening on: http://{args.host}:{args.port}")
    print(f"   Upstream LLM: {args.upstream}")
    print(f"   Fail-Open:    {not args.no_fail_open}")
    print("=" * 60)
    print(f"\nTo monitor OpenCode, Cursor, Aider, or Claude Code, configure:")
    print(f"   export OPENAI_BASE_URL=\"http://{args.host}:{args.port}/v1\"")
    print("\nWaiting for agent requests... (Press Ctrl+C to stop)\n")

    server = run_proxy(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        fail_open=not args.no_fail_open,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LongHorizon Guard proxy...")
        server.shutdown()


if __name__ == "__main__":
    main()
