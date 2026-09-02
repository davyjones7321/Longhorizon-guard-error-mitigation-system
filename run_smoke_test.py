#!/usr/bin/env python3
"""Smoke test script for verifying agent_scaffold and evalharness integration.

Usage:
  python run_smoke_test.py [--provider gemini|openrouter|anthropic]

This script:
1. Accepts --provider CLI argument (default: gemini).
2. Validates that the provider API key environment variable is set (fails loudly if missing).
3. Validates that ToolRegistry has registered tools (fails loudly if empty).
4. Loads exactly 1 trivial task.
5. Executes the task through the adapter using the chosen provider's LLM call.
6. Persists trajectory JSON and metadata JSON under eval/output/smoke_test_math_1/trial_1/.
7. Prints the final status, final answer, step count, and saved file locations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure harness and eval are on sys.path for direct script execution
ROOT_DIR = Path(__file__).parent.resolve()
HARNESS_DIR = ROOT_DIR / "harness"
EVAL_DIR = ROOT_DIR / "eval"

if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_scaffold.tools import ToolRegistry, default_tools
from evalharness.adapter import create_scaffold_adapter
from evalharness.llm import (
    create_anthropic_llm_call,
    create_gemini_llm_call,
    create_openrouter_llm_call,
    validate_api_key,
)
from evalharness.logger import save_metadata, save_trajectory
from evalharness.runner import run_task
from evalharness.schema import Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke test for agent_scaffold and evalharness integration.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openrouter", "anthropic"],
        default="gemini",
        help="LLM provider to use for smoke test (default: gemini)",
    )
    parser.add_argument(
        "--api-key-var",
        default=None,
        help="Environment variable name for API key (overrides default env var for chosen provider)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for chosen provider (overrides default model)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  agent_scaffold + evalharness Smoke Test")
    print("=" * 60)

    # 1. Resolve Provider & Endpoint
    if args.provider == "gemini":
        env_var = args.api_key_var or "GEMINI_API_KEY"
        model_name = args.model or "gemini-3.6-flash"
        endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        llm_factory = lambda: create_gemini_llm_call(api_key_env_var=env_var, model=model_name)
    elif args.provider == "openrouter":
        env_var = args.api_key_var or "OPENROUTER_API_KEY"
        model_name = args.model or "dots-studio/dots-3-note-preview:free"
        endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
        llm_factory = lambda: create_openrouter_llm_call(api_key_env_var=env_var, model=model_name)
    elif args.provider == "anthropic":
        env_var = args.api_key_var or "ANTHROPIC_API_KEY"
        model_name = args.model or "claude-3-5-sonnet-20241022"
        endpoint_url = "https://api.anthropic.com/v1/messages"
        llm_factory = lambda: create_anthropic_llm_call(api_key_env_var=env_var, model=model_name)

    print(f"\n[1/4] Using provider: {args.provider} (model: {model_name}, env_var: {env_var}) -> {endpoint_url}")
    try:
        api_key = validate_api_key(env_var)
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"      API key detected: {masked_key}")
    except ValueError as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Check Tool Registry
    print("\n[2/4] Initializing Tool Registry...")
    tools = default_tools()
    registry = ToolRegistry(tools)
    if not registry.names():
        print("FATAL ERROR: Tool registry has no tools registered!", file=sys.stderr)
        sys.exit(1)
    print(f"      Registered tools: {', '.join(registry.names())}")

    # 3. Create Trivial Task & Adapter
    print(f"\n[3/4] Creating adapter with real LLM call ({args.provider})...")
    try:
        real_llm = llm_factory()
        adapter = create_scaffold_adapter(
            llm_call=real_llm,
            registry=registry,
            max_steps=5,
        )
    except Exception as exc:
        print(f"FATAL ERROR: Failed to construct adapter/LLM call: {exc}", file=sys.stderr)
        sys.exit(1)

    task = Task(
        task_id="smoke_test_math_1",
        description="What is 2 + 2? Use the calculator tool to find the answer.",
        horizon_level=1,
        expected_output="4",
        max_steps=5,
    )
    print(f"      Task ID: {task.task_id}")
    print(f"      Description: {task.description}")

    # 4. Run Task once & Persist Trajectory / Metadata
    print("\n[4/4] Running task through adapter and persisting trajectory...")
    try:
        trajectory, metadata = run_task(adapter, task)
        metadata.trial_number = 1

        eval_output_dir = EVAL_DIR / "output"
        traj_path = save_trajectory(trajectory, eval_output_dir, task.task_id, trial=1)
        meta_path = save_metadata(metadata, eval_output_dir, task.task_id, trial=1)

        root_output_dir = ROOT_DIR / "output"
        if root_output_dir.resolve() != eval_output_dir.resolve():
            save_trajectory(trajectory, root_output_dir, task.task_id, trial=1)
            save_metadata(metadata, root_output_dir, task.task_id, trial=1)
    except Exception as exc:
        print(f"\nFATAL ERROR: Execution failed with exception: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Smoke Test Execution Summary")
    print("=" * 60)
    print(f"Final Status:       {metadata.final_status.value}")
    print(f"Total Steps Taken:  {metadata.total_steps_taken}")

    final_answer = "(no final answer)"
    if trajectory.steps:
        last_step = trajectory.steps[-1]
        print(f"Last Action Name:   {last_step.action_name}")
        if last_step.action_args and "answer" in last_step.action_args:
            final_answer = last_step.action_args["answer"]
        elif last_step.state_snapshot and "final_answer" in last_step.state_snapshot:
            final_answer = last_step.state_snapshot["final_answer"]

    print(f"Final Answer:       {final_answer}")
    print(f"Duration Seconds:   {metadata.duration_seconds}s")
    print(f"Trajectory JSON:    {traj_path}")
    print(f"Metadata JSON:      {meta_path}")
    print("=" * 60)

    if metadata.final_status.value in ("success", "done"):
        print("\nSmoke test PASSED successfully!")
        sys.exit(0)
    else:
        print(f"\nSmoke test FAILED with status: {metadata.final_status.value}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
