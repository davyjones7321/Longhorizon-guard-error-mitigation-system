# Long-Horizon Guard Data Contract (`SCHEMA.md`)

**Schema Version**: `1.0`

This document defines the standardized plain JSON / dictionary contract for run metadata and trajectory log files consumed and produced by `longhorizon_guard`. Any evaluation harness or agent platform can integrate with `longhorizon_guard` as long as its output files satisfy this contract.

> **Taxonomy Standard Note**: The standardized error taxonomy for `longhorizon_guard` consists of **7 error categories**:
> 1. `planning_error` — Plan was flawed from the start (wrong approach or missed constraint).
> 2. `memory_error` — Agent misremembered or lost track of context/facts from earlier steps.
> 3. `tool_use_error` — Action execution failed (bad tool parameters, malformed JSON, syntax error in tool call).
> 4. `reflection_error` — Misjudged progress (e.g. thought task was complete when it wasn't).
> 5. `external_error` — External environment or API failure (e.g. 429 rate limit, 5xx server error, timeout).
> 6. `grader_error` — Grader mistake or regex mismatch on a correct answer.
> 7. `other` — Unclassified or ambiguous failure.
>
> *Note: This 7-category taxonomy is the standardized classification for `longhorizon_guard` and supersedes earlier 5-category taxonomy drafts.*

---

## 1. Run Metadata Schema (`run_metadata.json`)

A single JSON object representing the execution metadata of one trial.

```json
{
  "schema_version": "1.0",
  "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "task_id": "chain_L4_a",
  "trial_number": 1,
  "horizon_level": 4,
  "total_steps_taken": 5,
  "final_status": "success",
  "duration_seconds": 12.45,
  "grader_notes": "grade=PASS, got='49', expected='49'",
  "created_at": "2026-08-25T14:00:00",
  "root_cause_step_index": null,
  "root_cause_error_type": null,
  "tag_confidence": null,
  "tag_source": null
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `string` | Unique UUID or identifier for the run. |
| `task_id` | `string` | Task identifier (e.g. `chain_L4_a`). |
| `trial_number` | `integer` | 1-based trial attempt number. |
| `horizon_level` | `integer` | Complexity / horizon length level (e.g., 1–4). |
| `total_steps_taken` | `integer` | Total number of execution steps taken by the agent. |
| `final_status` | `string` | Final outcome status: `"success"`, `"fail"`, `"timeout"`, `"error"`, or `"completed_unverified"`. |

### Optional / Tagging Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `string` | Contract schema version (default `"1.0"`). |
| `duration_seconds` | `float` | Total wall-clock span in seconds (last_timestamp - first_timestamp). |
| `active_duration_seconds` | `float` or `null` | Active execution time in seconds, excluding idle gaps > 300s between steps. |
| `grader_notes` | `string` | Grader notes or detailed error message. |
| `created_at` | `string` | ISO timestamp string of run execution. |
| `root_cause_step_index` | `integer` or `null` | 0-based step index where the root-cause failure occurred. |
| `root_cause_error_type` | `string` or `null` | Tag classification from the 7 standardized categories. |
| `tag_confidence` | `float` or `null` | Confidence score of the assigned tag (0.0 to 1.0). |
| `tag_source` | `string` or `null` | Source of the tag (`"human"`, `"llm_judge"`, `"agenterrorbench_import"`, etc.). |
| `task_description` | `string` or `null` | Full human-readable task description or prompt text. |
| `source_dataset` | `string` or `null` | Origin dataset or platform name (e.g. `"agenterrorbench"`, `"antigravity_session"`). |
| `source_llm_model` | `string` or `null` | LLM model identifier used during the session (e.g. `"gemini-3.6-flash"`, `"GPT-4o"`). |

---

## 2. Trajectory Schema (`trajectory.json`)

A single JSON object containing an ordered array of execution steps taken by the agent.

```json
{
  "schema_version": "1.0",
  "steps": [
    {
      "step_index": 0,
      "timestamp": 1787572519.628,
      "reasoning": "Thought: I need to calculate (((3+7)*2-5+9)/3)*4...",
      "action_name": "calculator",
      "action_args": {
        "expression": "(((3+7)*2-5+9)/3)*4"
      },
      "tool_response": "OK: executed calculator({'expression': '(((3+7)*2-5+9)/3)*4'})",
      "state_snapshot": null,
      "error_tag": null
    }
  ]
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `array` | List of step objects in chronological order. |
| `step.step_index` | `integer` | 0-based step index. |
| `step.reasoning` | `string` | Agent's raw chain-of-thought text / reasoning. |
| `step.action_name` | `string` | Name of tool called, `"done"`, or `"error"`. |
| `step.action_args` | `dict` | Parameters passed to the action. |

### Optional / Tagging Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `string` | Contract schema version (default `"1.0"`). |
| `step.timestamp` | `float` | Epoch timestamp in seconds. |
| `step.tool_response` | `string` or `null` | Environment / tool execution response string. |
| `step.state_snapshot` | `dict` or `null` | State history or prompt context snapshot. |
| `step.error_tag` | `string` or `null` | Per-step error annotation from the 7 standardized categories. |
