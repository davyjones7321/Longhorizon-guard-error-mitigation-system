#!/usr/bin/env python3
"""Batch experiment runner for evalharness.

Usage:
  python run_experiment.py [--provider gemini|openrouter|anthropic] [--tasks eval/tasks/tasks_arithmetic_chain.json] [--trials 3] [--delay 2]

Features:
- CLI argument --provider (choices: gemini, openrouter, anthropic; default: gemini).
- CLI argument --api-key-var to override the environment variable name for the chosen provider.
- CLI argument --model to override the model name.
- Startup message displaying chosen provider, model, API key env var, and target endpoint URL.
- Loads tasks from JSON config file (default: eval/tasks/tasks_arithmetic_chain.json).
- Runs specified trials (default: 3) per task using create_scaffold_adapter().
- Persists trajectory and metadata JSON files under eval/output/<task_id>/trial_<n>/.
- Consolidates all trial logs into eval/output/all_experiment_logs_combined.json.
- Grades each trial (compares final_answer against task's expected_output).
- Prints live progress line per trial.
- Rate-limit aware with configurable delay between trials.
- Prints a final summary table grouped by horizon_level.
- Exception handling per trial so failed trials do not crash the batch.
- Strictly ensures expected_output is NEVER included in LLM prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure harness and eval are on sys.path
ROOT_DIR = Path(__file__).parent.resolve()
HARNESS_DIR = ROOT_DIR / "harness"
EVAL_DIR = ROOT_DIR / "eval"

if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from agent_scaffold.tools import ToolRegistry, default_tools
    from evalharness.adapter import create_scaffold_adapter
    from evalharness.llm import (
        create_anthropic_llm_call,
        create_gemini_llm_call,
        create_openrouter_llm_call,
        validate_api_key,
    )
    from evalharness.logger import save_metadata, save_trajectory
    from evalharness.runner import load_tasks, run_task
    from evalharness.schema import FinalStatus, RunMetadata, Task, Trajectory
except ImportError as exc:
    print("\n[NOTE] 'run_experiment.py' is an internal batch benchmark runner.", file=sys.stderr)
    print("To test LongHorizon Guard with zero external dependencies, run:", file=sys.stderr)
    print("    python run_smoke_test.py", file=sys.stderr)
    print("To monitor your own agent harness in real time, run:", file=sys.stderr)
    print("    longhorizon-guard proxy --port 8000", file=sys.stderr)
    print("To evaluate any custom trajectory from your own harness, run:", file=sys.stderr)
    print("    python run_smoke_test.py --trajectory path/to/your_run.json\n", file=sys.stderr)
    sys.exit(1)


def resolve_provider_llm_call(
    provider: str,
    api_key_var: str | None = None,
    model: str | None = None,
) -> Tuple[Any, str, str, str]:
    """Resolve the LLM call function, env var name, model name, and endpoint URL based on --provider."""
    if provider == "gemini":
        env_var = api_key_var or "GEMINI_API_KEY"
        model_name = model or "gemini-2.5-flash"
        endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        llm_call = create_gemini_llm_call(api_key_env_var=env_var, model=model_name)
    elif provider == "openrouter":
        env_var = api_key_var or "OPENROUTER_API_KEY"
        model_name = model or "dots-studio/dots-3-note-preview:free"
        endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
        llm_call = create_openrouter_llm_call(api_key_env_var=env_var, model=model_name)
    elif provider == "anthropic":
        env_var = api_key_var or "ANTHROPIC_API_KEY"
        model_name = model or "claude-3-5-sonnet-20241022"
        endpoint_url = "https://api.anthropic.com/v1/messages"
        llm_call = create_anthropic_llm_call(api_key_env_var=env_var, model=model_name)
    else:
        raise ValueError(f"Unknown provider: '{provider}'. Supported: gemini, openrouter, anthropic")

    return llm_call, env_var, model_name, endpoint_url


def extract_final_answer(trajectory: Trajectory) -> str:
    """Extract final answer string from the last step's state_snapshot or action_args."""
    if not trajectory.steps:
        return ""
    last_step = trajectory.steps[-1]
    if last_step.state_snapshot and "final_answer" in last_step.state_snapshot:
        ans = last_step.state_snapshot["final_answer"]
        if ans is not None:
            return str(ans).strip()
    if last_step.action_args and "answer" in last_step.action_args:
        ans = last_step.action_args["answer"]
        if ans is not None:
            return str(ans).strip()
    return ""


def grade_trial(final_answer: str, expected_output: str | None) -> Tuple[bool, str]:
    """Grade trial answer against expected_output (exact match after stripping whitespace)."""
    if expected_output is None:
        return True, "No expected_output provided"

    clean_actual = str(final_answer).strip()
    clean_expected = str(expected_output).strip()

    is_match = (clean_actual == clean_expected)
    status_str = "PASS" if is_match else "FAIL"
    note = f"grade={status_str}, got='{clean_actual}', expected='{clean_expected}'"
    return is_match, note


def assert_expected_output_not_in_prompts(trajectory: Trajectory, expected_output: str | None, task_id: str) -> None:
    """Explicitly verify that expected_output is NEVER included in any prompt sent to the LLM."""
    if not expected_output:
        return
    clean_expected = str(expected_output).strip()
    for step in trajectory.steps:
        prompt_sent = ""
        if step.state_snapshot and "prompt_sent" in step.state_snapshot:
            prompt_sent = str(step.state_snapshot["prompt_sent"])

        if f'"expected_output": "{clean_expected}"' in prompt_sent or f"expected_output: {clean_expected}" in prompt_sent:
            raise RuntimeError(
                f"SAFETY FAILURE: task '{task_id}' leaked expected_output='{clean_expected}' into LLM prompt!"
            )


def consolidate_logs(output_base: Path) -> Path:
    """Consolidate all persisted trial trajectories and metadata into a single combined JSON file."""
    combined = []
    for task_dir in sorted(output_base.iterdir()):
        if not task_dir.is_dir() or task_dir.name in ("report",):
            continue
        for trial_dir in sorted(task_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue
            meta_file = trial_dir / "run_metadata.json"
            traj_file = trial_dir / "trajectory.json"

            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            traj = json.loads(traj_file.read_text(encoding="utf-8")) if traj_file.exists() else {}

            combined.append({
                "task_id": task_dir.name,
                "trial_number": trial_dir.name,
                "metadata": meta,
                "trajectory": traj.get("steps", []),
            })

    out_file = output_base / "all_experiment_logs_combined.json"
    out_file.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch evaluation experiment.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openrouter", "anthropic"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
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
    parser.add_argument(
        "--tasks",
        default=str(EVAL_DIR / "tasks" / "tasks_arithmetic_chain.json"),
        help="Path to tasks JSON file (default: eval/tasks/tasks_arithmetic_chain.json)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per task (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between trials to avoid rate limits (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVAL_DIR / "output"),
        help="Directory to persist output trajectories and metadata (default: eval/output)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  evalharness Batch Experiment Runner")
    print("=" * 70)

    # 1. Resolve Provider & Construct LLM Call
    try:
        llm_call, api_key_var, model_name, endpoint_url = resolve_provider_llm_call(
            provider=args.provider,
            api_key_var=args.api_key_var,
            model=args.model,
        )
    except Exception as exc:
        print(f"\nFATAL ERROR initializing provider '{args.provider}': {exc}", file=sys.stderr)
        sys.exit(1)

    # Startup print statement displaying provider, model, env var, and target endpoint
    print(f"\nUsing provider: {args.provider} (model: {model_name}, env_var: {api_key_var}) -> {endpoint_url}")

    # Validate API key
    try:
        api_key = validate_api_key(api_key_var)
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"API Key detected for {api_key_var}: {masked_key}")
    except ValueError as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Load Tasks
    task_file = Path(args.tasks)
    print(f"\nLoading tasks from {task_file}...")
    if not task_file.exists():
        fallback_file = EVAL_DIR / "tests" / "tasks_arithmetic_chain.json"
        if fallback_file.exists():
            task_file = fallback_file
        else:
            print(f"FATAL ERROR: Task file not found: {task_file}", file=sys.stderr)
            sys.exit(1)

    try:
        tasks = load_tasks(task_file)
    except Exception as exc:
        print(f"FATAL ERROR: Failed to load tasks from {task_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(tasks)} tasks to run across {args.trials} trials each.")

    # 3. Setup Tool Registry & Output Base
    registry = ToolRegistry(default_tools())
    output_base = Path(args.output_dir)

    print(f"\nExecuting trials (delay={args.delay}s between runs)...\n")
    print("-" * 70)

    stats_by_horizon: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})

    total_tasks = len(tasks)
    trial_counter = 0
    total_expected_runs = total_tasks * args.trials

    for task_idx, task in enumerate(tasks, 1):
        if task.expected_output and task.expected_output in task.description:
            print(
                f"Warning: expected_output '{task.expected_output}' appeared in description of task {task.task_id}"
            )

        for trial in range(1, args.trials + 1):
            trial_counter += 1

            adapter = create_scaffold_adapter(
                llm_call=llm_call,
                registry=registry,
                max_steps=task.max_steps,
            )

            start_time = time.time()
            try:
                trajectory, metadata = run_task(adapter, task)
                metadata.trial_number = trial
                duration = time.time() - start_time

                final_answer = extract_final_answer(trajectory)
                passed, grade_note = grade_trial(final_answer, task.expected_output)
                metadata.grader_notes = grade_note

                assert_expected_output_not_in_prompts(trajectory, task.expected_output, task.task_id)

            except Exception as exc:
                duration = time.time() - start_time
                trajectory = Trajectory()
                final_answer = f"ERROR: {exc}"
                passed = False
                metadata = RunMetadata(
                    run_id=str(uuid.uuid4()),
                    task_id=task.task_id,
                    trial_number=trial,
                    horizon_level=task.horizon_level,
                    total_steps_taken=0,
                    final_status=FinalStatus.ERROR,
                    duration_seconds=round(duration, 4),
                    grader_notes=f"Trial execution error: {exc}",
                )

            try:
                save_trajectory(trajectory, output_base, task.task_id, trial)
                save_metadata(metadata, output_base, task.task_id, trial)

                root_output = ROOT_DIR / "output"
                if root_output.resolve() != output_base.resolve():
                    save_trajectory(trajectory, root_output, task.task_id, trial)
                    save_metadata(metadata, root_output, task.task_id, trial)
            except Exception as exc:
                print(f"Warning: Failed to persist files for {task.task_id} trial {trial}: {exc}")

            stats_by_horizon[task.horizon_level]["total"] += 1
            if passed:
                stats_by_horizon[task.horizon_level]["success"] += 1

            match_str = "PASS" if passed else "FAIL"
            status_val = metadata.final_status.value
            ans_display = (final_answer[:25] + "...") if len(final_answer) > 28 else final_answer
            print(
                f"[{trial_counter:02d}/{total_expected_runs:02d}] "
                f"task={task.task_id:<14} (L{task.horizon_level}) | "
                f"trial={trial}/{args.trials} | "
                f"status={status_val:<7} | "
                f"ans='{ans_display}' | "
                f"match={match_str}"
            )

            if trial_counter < total_expected_runs and args.delay > 0:
                time.sleep(args.delay)

    combined_log_path = consolidate_logs(output_base)
    root_output = ROOT_DIR / "output"
    if root_output.resolve() != output_base.resolve():
        consolidate_logs(root_output)

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY BY HORIZON LEVEL")
    print("=" * 70)
    print(f" {'Horizon':<10} | {'Total Trials':<14} | {'Succeeded':<12} | {'Success Rate':<14}")
    print("-" * 70)

    grand_total = 0
    grand_success = 0

    for level in sorted(stats_by_horizon.keys()):
        tot = stats_by_horizon[level]["total"]
        succ = stats_by_horizon[level]["success"]
        rate = (succ / tot * 100.0) if tot > 0 else 0.0
        grand_total += tot
        grand_success += succ
        print(f" Horizon {level:<2} | {tot:<14} | {succ:<12} | {rate:>12.1f}%")

    print("-" * 70)
    overall_rate = (grand_success / grand_total * 100.0) if grand_total > 0 else 0.0
    print(f" {'Overall':<10} | {grand_total:<14} | {grand_success:<12} | {overall_rate:>12.1f}%")
    print("=" * 70)
    print(f"\nIndividual trial outputs: {output_base.resolve()}")
    print(f"Consolidated single-file log: {combined_log_path.resolve()}")


if __name__ == "__main__":
    main()
