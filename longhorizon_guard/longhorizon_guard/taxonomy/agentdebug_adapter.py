"""Adapter to convert agenterrorbench_converted.json records into AgentDebug input format."""

import json
from typing import Any, Dict, List


def _extract_task_description(meta: Dict[str, Any], raw_steps: List[Dict[str, Any]]) -> str:
    """Task text isn't in metadata for this dataset — reconstruct it from
    the trajectory. ALFWorld/WebShop/most GAIA: step 0's tool_response
    contains the system-delivered task prompt. Remaining GAIA cases (~12%):
    step 0's tool_response is empty, but the agent's own reasoning reliably
    restates the task before acting on it."""
    if not raw_steps:
        return ""
    first_response = raw_steps[0].get("tool_response") or ""
    if first_response.strip():
        return first_response
    # Fallback: use the agent's own first-step reasoning (contains the
    # restated task, plus its own planning text mixed in — noisier, but
    # the only available source for this ~12% of GAIA records).
    return raw_steps[0].get("reasoning") or ""


def convert_to_agentdebug_format(run: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one agenterrorbench_converted.json record into the dict shape

    ErrorTypeDetector.analyze_trajectory() expects, bypassing their own
    parse_trajectory()/file-based loader entirely.
    """
    meta = run.get("metadata", {}) or {}
    traj = run.get("trajectory", {}) or {}
    raw_steps = traj.get("steps", [])

    steps = []
    prev_tool_response = ""
    for s in raw_steps:
        reasoning = s.get("reasoning") or ""
        action_name = s.get("action_name") or "none"
        action_args = s.get("action_args", {})

        # Synthesize an <action> tag wrapping action_name + args, appended
        # to the existing reasoning content (which already contains
        # <memory>/<reflection>/<plan> tags in this dataset's raw format).
        if isinstance(action_args, dict) and action_args:
            action_str = f"{action_name}[{json.dumps(action_args)}]"
        elif action_args:
            action_str = f"{action_name}[{action_args}]"
        else:
            action_str = action_name

        content = f"{reasoning}\n<action>{action_str}</action>"

        steps.append({
            "step": s.get("step_index", 0) + 1,  # 0-indexed source -> 1-indexed target
            "content": content,
            "env_response": s.get("tool_response") or "",
            "current_input": prev_tool_response,
        })
        prev_tool_response = s.get("tool_response") or ""

    return {
        "task_id": meta.get("run_id", "unknown"),
        "task_description": _extract_task_description(meta, raw_steps),
        "success": meta.get("final_status") == "success",
        "steps": steps,
        "total_steps": len(steps),
        "environment": meta.get("task_id", "alfworld"),  # task_id IS the environment name in this dataset
    }
