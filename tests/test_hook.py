"""Unit tests for WorkBuddy/CodeBuddy hook integration."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

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

    def test_state_restoration_across_invocations(self):
        """Engine state and tracker must be reconstructed from disk on stateless hook calls."""
        session_id = "test-session-restore"
        # 1. UserPromptSubmit establishes plan
        handle_hook({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "prompt": "1. Build auth 2. Build api",
        }, log_dir=self.test_dir)

        # 2. Step 1 executes
        handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file": "auth.py"},
            "tool_response": "auth written",
        }, log_dir=self.test_dir)

        # 3. Next step with guard=None (simulates separate CLI invocation)
        handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file": "api.py"},
            "tool_response": "api written",
        }, log_dir=self.test_dir, guard=None)

        state_path = os.path.join(self.test_dir, f"_state_{session_id}.json")
        self.assertTrue(os.path.exists(state_path))
        with open(state_path, "r", encoding="utf-8") as f:
            saved_state = json.load(f)
        self.assertEqual(saved_state["step_counter"], 2)
        self.assertEqual(len(saved_state["history"]), 2)
        self.assertTrue(len(saved_state.get("proposed_plan", "")) >= 2)

    def test_history_repetition_exact_threshold(self):
        """Action repetition heuristic must not self-match; triggers on 4th identical action (3 in history), not 3rd."""
        session_id = "test-session-exact-rep"
        tool_name = "Read"
        tool_input = {"path": "config.yaml"}
        resp = "server: localhost"

        # Step 1: 1st time action executes -> should NOT flag
        r1 = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": resp,
        }, log_dir=self.test_dir)
        self.assertTrue(r1.get("continue"))

        # Step 2: 2nd time action executes -> prior history has 1 instance -> should NOT flag
        r2 = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": resp,
        }, log_dir=self.test_dir)
        self.assertTrue(r2.get("continue"))

        # Step 3: 3rd time action executes -> prior history has 2 instances -> should NOT flag (threshold is 3)
        r3 = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": resp,
        }, log_dir=self.test_dir)
        self.assertTrue(r3.get("continue"))

        # Step 4: 4th time action executes -> prior history has 3 instances -> triggers repetition flag!
        r4 = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": resp,
        }, log_dir=self.test_dir)
        self.assertIn("hookSpecificOutput", r4)
        self.assertIn("additionalContext", r4["hookSpecificOutput"])
        self.assertIn("repeated 3 times", r4["hookSpecificOutput"]["additionalContext"].lower())

    def test_one_shot_halt_blocks_next_pre_tool_use_and_clears(self):
        """With block_on_critical=True, a step with continue_execution=False denies next PreToolUse and clears."""
        session_id = "test-session-one-shot"
        mock_guard = MagicMock()
        mock_guard.block_on_critical = True
        mock_guard.on_step.return_value = {
            "flagged": True,
            "category": "drift",
            "confidence": 0.95,
            "warning": "Critical drift detected",
            "continue_execution": False,
            "suggestions": ["Stop drift"],
        }

        # 1. PostToolUse produces continue_execution=False
        post_resp = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "tool_response": "error",
        }, guard=mock_guard, log_dir=self.test_dir)
        self.assertIn("hookSpecificOutput", post_resp)

        # 2. Immediately following PreToolUse MUST be denied
        pre_resp1 = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }, log_dir=self.test_dir)
        self.assertIn("hookSpecificOutput", pre_resp1)
        self.assertEqual(pre_resp1["hookSpecificOutput"].get("permissionDecision"), "deny")
        self.assertIn("Critical flag on step 0 [drift]", pre_resp1["hookSpecificOutput"].get("permissionDecisionReason", ""))

        # 3. Third event (another PreToolUse, unrelated tool) MUST NOT be denied (one-shot cleared)
        pre_resp2 = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "app.py"},
        }, log_dir=self.test_dir)
        self.assertTrue(pre_resp2.get("continue"))
        self.assertNotIn("hookSpecificOutput", pre_resp2)

    def test_one_shot_halt_with_real_guard_end_to_end(self):
        """End-to-end one-shot halt verification with real GuardInterface and GUARD_BLOCK_ON_CRITICAL=true."""
        session_id = "test-session-e2e-halt"
        old_val = os.environ.get("GUARD_BLOCK_ON_CRITICAL")
        os.environ["GUARD_BLOCK_ON_CRITICAL"] = "true"
        try:
            tool_name = "Bash"
            tool_input = {"command": "curl http://offline-host"}
            resp = "curl: (7) Failed to connect to offline-host"

            # Execute 3 times to build history
            for _ in range(3):
                handle_hook({
                    "session_id": session_id,
                    "hook_event_name": "PostToolUse",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_response": resp,
                }, log_dir=self.test_dir)

            # 4th execution triggers structural_action_repetition -> continue_execution=False
            post4 = handle_hook({
                "session_id": session_id,
                "hook_event_name": "PostToolUse",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_response": resp,
            }, log_dir=self.test_dir)
            self.assertIn("hookSpecificOutput", post4)

            # Next PreToolUse MUST be denied
            pre_denied = handle_hook({
                "session_id": session_id,
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "curl http://different-host"},
            }, log_dir=self.test_dir)
            self.assertIn("hookSpecificOutput", pre_denied)
            self.assertEqual(pre_denied["hookSpecificOutput"].get("permissionDecision"), "deny")

            # Subsequent PreToolUse MUST NOT be denied (one-shot cleared)
            pre_allowed = handle_hook({
                "session_id": session_id,
                "hook_event_name": "PreToolUse",
                "tool_name": "Read",
                "tool_input": {"path": "config.yaml"},
            }, log_dir=self.test_dir)
            self.assertTrue(pre_allowed.get("continue"))
            self.assertNotIn("hookSpecificOutput", pre_allowed)
        finally:
            if old_val is not None:
                os.environ["GUARD_BLOCK_ON_CRITICAL"] = old_val
            else:
                os.environ.pop("GUARD_BLOCK_ON_CRITICAL", None)

    def test_post_tool_use_steer_contains_category_and_suggestions(self):
        """PostToolUse alert and log must contain top-level category and confidence."""
        session_id = "test-session-schema"
        resp = handle_hook({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": "Curl",
            "tool_input": {"url": "http://example.com/api"},
            "tool_response": "HTTP 404 error: resource not found on remote server",
        }, log_dir=self.test_dir)

        self.assertIn("hookSpecificOutput", resp)
        self.assertIn("additionalContext", resp["hookSpecificOutput"])

        log_path = os.path.join(self.test_dir, f"session_{session_id}.jsonl")
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        self.assertEqual(len(lines), 1)
        step_entry = lines[0]
        self.assertTrue(step_entry["flagged"])
        self.assertEqual(step_entry["category"], "external_error")
        self.assertIsInstance(step_entry["confidence"], float)
        self.assertIsInstance(step_entry["suggestions"], list)


if __name__ == "__main__":
    unittest.main()
