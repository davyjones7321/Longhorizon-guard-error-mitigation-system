# LongHorizon Guard: Long-Horizon Agent Error Mitigation & Trajectory Monitoring

LongHorizon Guard is a Python library and CLI tool designed to detect, track, and mitigate failure propagation in long-horizon autonomous LLM agent workflows.

The system monitors agent execution trajectories step-by-step using a two-layer failure detection engine, tracks subgoal progress, measures trajectory drift, and triggers plan reflection to prevent error cascading across extended task horizons.

---

## Technical Foundations and Academic Citations

LongHorizon Guard builds upon foundational research in LLM agent failure taxonomy, error propagation, and self-reflection:

1. **AgentErrorBench: Evaluating and Mitigating Failure Propagation in Language Agent Workflows**
   - Paper: [arXiv:2410.15836](https://arxiv.org/abs/2410.15836)
   - Dataset: [AgentErrorBench on GitHub](https://github.com/THUDM/AgentErrorBench) | [HuggingFace Datasets](https://huggingface.co/datasets/THUDM/AgentErrorBench)
   - Implemented Components: Primary taxonomy of root-cause error categories (`planning_error`, `reflection_error`, `memory_error`, `tool_use_error`, `external_error`), trajectory structure schema, and benchmark evaluation methodology across ALFWORLD, WebShop, and GAIA environments.

2. **AgentDebug: Fine-Grained Error Detection and Localization for LLM Agents**
   - Paper: [arXiv:2409.11727](https://arxiv.org/abs/2409.11727)
   - Implemented Components: Fine-grained structural detectors for action repetition, nothing-happens loop detection, and early step localization of root-cause errors.

3. **Reflexion: Language Agents with Verbal Reinforcement Learning**
   - Paper: [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
   - Implemented Components: Periodic verbal plan reflection and invalidation triggers based on stalled subgoal counts and trajectory drift thresholds.

---

## Core System Architecture

The package implements a modular, non-blocking 4-hook interface (`GuardInterface`):

```text
  Host Agent Execution Loop
             |
             +---> 1. on_plan_proposed(task_description, proposed_plan)
             |
             +---> 2. on_step(step_record, history)
             |          |
             |          +---> Layer A: Broad-Corpus TF-IDF Pattern Matcher
             |          +---> Layer B: Structural & Heuristic Rule Detectors
             |          +---> SubgoalTracker (State & Rule S3 Status)
             |          +---> DriftMonitor (Severity: LOW / MEDIUM / HIGH)
             |          +---> PlanReflector (Dual Triggers & Coincidence Guard)
             |
             +---> 3. on_subgoal_boundary(subgoal_id, status)
             |
             +---> 4. on_run_end(metadata, trajectory) -> Failure Summary
```

### Key Modules Implemented

1. **Two-Layer Failure Matching Engine**:
   - **Layer A (Broad-Corpus TF-IDF Pattern Matcher)**: Matches step reasoning and action text against 11 mined failure pattern centroids using a pre-computed 5,454-term broad-corpus IDF table built across 5,013 trajectory steps.
   - **Layer B (Structural & Rule Detectors)**: Fallback rule detectors for structural failure modes including action repetition (`_detect_action_repetition`), repeated observation stagnation (`_detect_nothing_happens_loop`), mechanical search (`planning_exhaustive_search`), malformed tool formatting, and step limit exhaustion.
   - **Input Scoping**: Layer A vectorizes only novel step text (`reasoning + action_name + action_args`), explicitly excluding `tool_response` from TF-IDF vectorization to prevent environment boilerplate false positives (such as repeated ALFWORLD cabinet and countertop descriptions). Layer B structural detectors retain full access to raw `tool_response` text.

2. **Subgoal Tracker (`subgoals/tracker.py`)**:
   - Parses agent-declared plans into structured subgoals.
   - Tracks state transitions: `not_started`, `in_progress`, `completed`, `failed`.
   - Implements Rule S3: assigns `stalled_advanced` status when an agent is forced to advance to a subsequent subgoal without completing the current one.

3. **Drift Monitor (`drift_monitor/monitor.py`)**:
   - Assesses trajectory drift severity (`LOW`, `MEDIUM`, `HIGH`).
   - Triggers elevated drift when accumulated failure thresholds are met (2 or more failed subgoals, 2 or more stalled subgoals, or a slow progress ratio).
   - Degrades gracefully when pattern or subgoal data is absent.

4. **Plan Reflector (`reflector/reflector.py`)**:
   - Evaluates whether the active plan remains viable.
   - Invalidates plans when 2 or more subgoals fail or when drift severity reaches `HIGH`.
   - Uses dual triggers (subgoal boundaries and step intervals) coupled with a coincidence guard to prevent duplicate evaluations.

5. **Fail-Open System Contract**:
   - Wraps execution in safe exception handlers with a 2.0-second maximum timeout per hook.
   - Guarantees that internal monitor failures or unhandled exceptions log a warning and return `flagged=False`, preventing host agent execution crashes.

---

## Installation

### Prerequisites
- Python 3.9 or higher

### Cloning the Repository
```bash
git clone https://github.com/your-org/error-prop.git
cd error-prop/longhorizon_guard
```

### Installing the Package
Install in editable mode for local development:
```bash
pip install -e .
```

---

## Usage

### 1. Command Line Interface (CLI)

Check library status and verified pattern statistics:
```bash
longhorizon-guard info
```

Evaluate a JSON trajectory file:
```bash
longhorizon-guard evaluate --trajectory path/to/trajectory.json
```

### 2. Python API Integration

Incorporate `GuardInterface` into an agent execution loop:

```python
from longhorizon_guard import GuardInterface

# Initialize GuardInterface (automatically loads embedded broad-corpus IDF)
guard = GuardInterface()

# Hook 1: Register initial task description and plan
task_description = "Locate item in WebShop and complete purchase"
proposed_plan = "1. Search for item\n2. Select options\n3. Click buy now"
guard.on_plan_proposed(task_description, proposed_plan)

# Hook 2: Process execution steps inside the agent loop
history = []
step_record = {
    "step_index": 0,
    "reasoning": "Search for blue cotton sweater in size medium",
    "action_name": "search",
    "action_args": "blue cotton sweater medium",
    "tool_response": "Found 15 items"
}

res = guard.on_step(step_record, history)

if res.get("flagged"):
    print(f"Flagged category: {res['match_details']['category']}")
    print(f"Confidence: {res['match_details']['confidence']}")

history.append(step_record)

# Hook 3: Fire on subgoal completion or transition
guard.on_subgoal_boundary(subgoal_id="subgoal_1", status="completed")

# Hook 4: Finalize run and retrieve root-cause attribution
summary = guard.on_run_end(
    metadata={"task_id": "webshop_01"}, 
    trajectory={"steps": history}
)

print(f"Root Cause Category: {summary['root_cause_error_type']}")
print(f"Root Cause Step Index: {summary['root_cause_step_index']}")
```

---

## Running Verification Tests

Run the full pytest suite (69 tests):
```bash
pytest tests/ -v
```

---

## Dataset Resources

- **AgentErrorBench Benchmark Dataset**: [GitHub Repository](https://github.com/THUDM/AgentErrorBench) | [HuggingFace Dataset](https://huggingface.co/datasets/THUDM/AgentErrorBench)
- **ALFWORLD Environment**: [ALFWORLD GitHub](https://github.com/alfworld/alfworld)
- **WebShop Environment**: [WebShop GitHub](https://github.com/princeton-nlp/webshop)
- **GAIA Benchmark**: [GAIA Benchmark on HuggingFace](https://huggingface.co/datasets/gaia-benchmark/GAIA)
