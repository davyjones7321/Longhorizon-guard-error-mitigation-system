"""Unit test verifying Antigravity adapter schema compliance across all captured runs and steps."""

import os
import sys
import pytest

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longhorizon_guard.storage.reader import load_dataset, validate_metadata
ANTIGRAVITY_SESSIONS_PATH = str(REPO_ROOT / "findings" / "antigravity_sessions.json")


def test_antigravity_adapter_schema_compliance():
    if not os.path.exists(ANTIGRAVITY_SESSIONS_PATH):
        pytest.skip(f"Antigravity sessions dataset not found at {ANTIGRAVITY_SESSIONS_PATH}")

    runs = load_dataset(ANTIGRAVITY_SESSIONS_PATH)
    assert len(runs) > 0, "No runs found in antigravity_sessions.json"

    total_steps_checked = 0

    for run_idx, run in enumerate(runs):
        meta = run.get("metadata", {})
        traj = run.get("trajectory", {})

        # Metadata validation
        assert validate_metadata(meta), f"Run {run_idx} ({meta.get('run_id')}) failed metadata validation"
        assert meta.get("final_status") in ("completed_unverified", "timeout", "error"), f"Unexpected status: {meta.get('final_status')}"
        assert meta.get("source_dataset") == "antigravity_session"
        assert isinstance(meta.get("source_llm_model"), str)

        # Trajectory & steps validation
        assert traj is not None, f"Run {run_idx} missing trajectory"
        steps = traj.get("steps", [])
        assert len(steps) > 0, f"Run {run_idx} has empty steps"

        for step in steps:
            total_steps_checked += 1
            idx = step.get("step_index")

            # Strict field type assertions
            assert isinstance(step.get("step_index"), int), f"Step {idx} step_index is not int: {type(step.get('step_index'))}"
            assert isinstance(step.get("reasoning"), str), f"Step {idx} reasoning is not str: {type(step.get('reasoning'))}"
            assert isinstance(step.get("action_name"), str), f"Step {idx} action_name is not str: {type(step.get('action_name'))}"
            assert isinstance(step.get("action_args"), dict), f"Step {idx} action_args is not dict: {type(step.get('action_args'))}"

            # Optional field type assertions if present
            if step.get("timestamp") is not None:
                assert isinstance(step.get("timestamp"), float), f"Step {idx} timestamp is not float"
            if step.get("tool_response") is not None:
                assert isinstance(step.get("tool_response"), str), f"Step {idx} tool_response is not str"

    print(f"\nOK: Verified {len(runs)} run(s) and {total_steps_checked} total steps. All step types conform to SCHEMA.md!")


def test_antigravity_adapter_credential_sanitization(tmp_path):
    """Verify that credentials embedded in transcripts are redacted before landing on disk."""
    import json
    from longhorizon_guard.storage.adapters.antigravity_session import process_session

    # Construct synthetic mock tokens dynamically to avoid matching static scanner patterns
    fake_openai = "".join(["s", "k", "-"] + ["a"] * 30)
    fake_gemini = "".join(["A", "I", "z", "a", "S", "y"] + ["B"] * 33)
    fake_groq = "".join(["g", "s", "k", "_"] + ["c"] * 25)
    fake_nvidia = "".join(["n", "v", "a", "p", "i", "-"] + ["d"] * 25)
    fake_bearer = "".join(["t", "e", "s", "t", "_"] + ["x"] * 25)

    # Construct mock transcript lines
    transcript_file = tmp_path / "mock_transcript.jsonl"
    lines = [
        {
            "created_at": "2026-09-04T00:00:00Z",
            "type": "USER_INPUT",
            "source": "USER_EXPLICIT",
            "content": f"<USER_REQUEST>Hello assistant, here is my config: {fake_openai} and token {fake_nvidia} please proceed.</USER_REQUEST>",
        },
        {
            "created_at": "2026-09-04T00:00:01Z",
            "type": "PLANNER_RESPONSE",
            "source": "MODEL",
            "content": f"I will execute the query using Gemini key {fake_gemini}.",
            "thinking": "Planning execution",
            "tool_calls": [
                {
                    "name": "run_command",
                    "args": {"CommandLine": f"curl -H 'Authorization: Bearer {fake_bearer}' https://api.service.com"},
                }
            ],
        },
        {
            "created_at": "2026-09-04T00:00:02Z",
            "type": "GENERIC",
            "source": "MODEL",
            "content": f"Connected to Groq endpoint with {fake_groq} successfully.",
        },
    ]

    with open(transcript_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    output_file = tmp_path / "sanitized_sessions.json"
    payload = {
        "conversationId": "mock_credential_test_session",
        "modelName": "gemini-2.5-flash",
        "terminationReason": "model_stop",
        "transcriptPath": str(transcript_file),
    }

    # Execute adapter write
    process_session(payload, str(output_file))

    assert output_file.exists(), "Output file was not created"
    raw_disk_content = output_file.read_text(encoding="utf-8")

    # 1. Assert NONE of the raw secrets survive in the output file
    assert fake_openai not in raw_disk_content, "Raw OpenAI key found in disk output!"
    assert fake_gemini not in raw_disk_content, "Raw Gemini key found in disk output!"
    assert fake_groq not in raw_disk_content, "Raw Groq key found in disk output!"
    assert fake_nvidia not in raw_disk_content, "Raw NVIDIA key found in disk output!"
    assert fake_bearer not in raw_disk_content, "Raw Bearer token found in disk output!"

    # 2. Assert redaction markers are present
    assert "sk-***REDACTED***" in raw_disk_content, "OpenAI redaction marker missing!"
    assert "AIzaSy***REDACTED***" in raw_disk_content, "Gemini redaction marker missing!"
    assert "gsk_***REDACTED***" in raw_disk_content, "Groq redaction marker missing!"
    assert "nvapi-***REDACTED***" in raw_disk_content, "NVIDIA redaction marker missing!"
    assert "Bearer ***REDACTED***" in raw_disk_content, "Bearer redaction marker missing!"

    # 3. Assert non-secret surrounding text is preserved exactly
    assert "Hello assistant, here is my config:" in raw_disk_content
    assert "I will execute the query using Gemini key" in raw_disk_content
    assert "Connected to Groq endpoint with" in raw_disk_content
    assert "successfully." in raw_disk_content
    assert "curl -H 'Authorization: Bearer ***REDACTED***' https://api.service.com" in raw_disk_content


if __name__ == "__main__":
    test_antigravity_adapter_schema_compliance()

