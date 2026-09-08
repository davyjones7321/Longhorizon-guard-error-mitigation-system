"""Unit tests for WorkBuddy/CodeBuddy hook integration."""

import json
import os
import shutil
import tempfile
import unittest

from longhorizon_guard.hook import handle_hook


class TestHookIntegration(unittest.TestCase):
    """Test suite verifying hook observation, loop interception, and steering."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="guard_test_hook_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_user_prompt_submit(self):
        payload = {
            "session_id": "test-session-1",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Build a secure priority queue in Python with unit tests",
        }
        resp = handle_hook(payload, log_dir=self.test_dir)
        self.assertTrue(resp.get("continue"))

        log_path = os.path.join(self.test_dir, "session_test-session-1.jsonl")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["type"], "plan")

    def test_pre_tool_use_normal(self):
        payload = {
            "session_id": "test-session-2",
            "hook_event_name": "PreToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": "queue.py", "content": "class PriorityQueue: pass"},
        }
        resp = handle_hook(payload, log_dir=self.test_dir)
        self.assertTrue(resp.get("continue"))

    def test_pre_tool_use_intercepts_loop(self):
        session_id = "test-session-loop"
        tool_name = "Bash"
        tool_input = {"command": "python broken_script.py"}

        # Simulate 3 failed executions of identical command
        for _ in range(3):
            handle_hook({
                "session_id": session_id,
                "hook_event_name": "PreToolUse",
                "tool_name": tool_name,
                "tool_input": tool_input,
            }, log_dir=self.test_dir)
            handle_hook({
                "session_id": session_id,
                "hook_event_name": "PostToolUse",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_response": "Traceback (most recent call last): ModuleNotFoundError: No module named 'foo'",
            }, log_dir=self.test_dir)

        # 4th PreToolUse attempt MUST be denied/intercepted
        interception = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
        }, log_dir=self.test_dir)

        self.assertIn("hookSpecificOutput", interception)
        decision = interception["hookSpecificOutput"].get("permissionDecision")
        self.assertEqual(decision, "deny")
        self.assertIn("Loop Detected", interception["hookSpecificOutput"].get("permissionDecisionReason", ""))

    def test_post_tool_use_flags_and_steers(self):
        session_id = "test-session-steer"
        # Submit prompt
        handle_hook({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Create database tables",
        }, log_dir=self.test_dir)

        # Tool execution with error
        resp = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "sqlite3 db.sqlite < schema.sql"},
            "tool_response": "Error: near line 1: syntax error in CREATE TABLE",
        }, log_dir=self.test_dir)

        # Verify step recorded
        log_path = os.path.join(self.test_dir, f"session_{session_id}.jsonl")
        self.assertTrue(os.path.exists(log_path))

    def test_turn_end_summary(self):
        session_id = "test-session-summary"
        handle_hook({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Run tests",
        }, log_dir=self.test_dir)
        handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest"},
            "tool_response": "1 passed",
        }, log_dir=self.test_dir)
        handle_hook({
            "session_id": session_id,
            "hook_event_name": "Stop",
        }, log_dir=self.test_dir)

        log_path = os.path.join(self.test_dir, f"session_{session_id}.jsonl")
        with open(log_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        types = [r["type"] for r in records]
        self.assertIn("plan", types)
        self.assertIn("step", types)
        self.assertIn("summary", types)


if __name__ == "__main__":
    unittest.main()
