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
             |          +---> DriftMonitor (Severity: none / low / medium / high / critical)
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
   - Tracks state transitions across the full `SubgoalStatus` lifecycle: `not_started`, `in_progress`, `completed`, `stalled_advanced`, `failed`, and `abandoned`.
   - Implements Rule S3: assigns `stalled_advanced` status when an agent exceeds the maximum step limit (default: 10 steps, configurable via `max_subgoal_steps`) and is forced to advance to a subsequent subgoal without completing the current one.

3. **Drift Monitor (`drift_monitor/monitor.py`)**:
   - Assesses trajectory drift severity across 5 discrete levels: `none`, `low`, `medium`, `high`, and `critical` (mapped from numeric drift score [0.0, 1.0]).
   - Flags drift when severity score meets or exceeds `drift_threshold` (default: 0.35, configurable via `GuardConfig`), triggered by accumulated signals (repeated stalled subgoals, slow progress ratio, accumulated subgoal failures, or pattern repetition).
   - Degrades gracefully when pattern or subgoal data is absent.

4. **Plan Reflector (`reflector/reflector.py`)**:
   - Evaluates whether the active plan remains viable.
   - Invalidates plans when 2 or more subgoals fail or when drift severity reaches `high` or `critical`.
   - Uses dual triggers (subgoal boundaries and step intervals) coupled with a coincidence guard to prevent duplicate evaluations.

5. **Fail-Open System Contract**:
   - Wraps execution in safe exception handlers with a 2.0-second maximum timeout per hook.
   - Guarantees that internal monitor failures or unhandled exceptions log a warning and return `flagged=False`, preventing host agent execution crashes.

---

## Installation

### Prerequisites
- Python 3.9 or higher

### Direct One-Line Installation (Pip from GitHub)
Install directly from GitHub with zero subdirectory syntax required:
```bash
pip install git+https://github.com/davyjones7321/Longhorizon-guard-error-mitigation-system.git
```

### Local Installation
```bash
git clone https://github.com/davyjones7321/Longhorizon-guard-error-mitigation-system.git
cd Longhorizon-guard-error-mitigation-system
pip install .
```

For active development, install in editable mode:
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

Run the real-time API proxy to monitor coding assistants (OpenCode, Cursor, Aider, Claude Code):
```bash
longhorizon-guard proxy --port 8000 --upstream https://api.openai.com/v1
```

Once running, configure your coding assistant's API base URL once:
- **OpenCode / Codex / Aider / Cursor**: Set environment variable:
  ```bash
  export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
  ```
- **Claude Code**: Set environment variable:
  ```bash
  export ANTHROPIC_BASE_URL="http://127.0.0.1:8000/v1"
  ```
The proxy intercepts all tool calls, reasoning steps, and plans transparently in the background, printing real-time warnings if the assistant enters an infinite loop, repeats failed commands, or drifts from its subgoals.

### 2. Python API Integration

Incorporate `GuardInterface` into an agent execution loop:

```python
from longhorizon_guard import GuardInterface

# Initialize GuardInterface (automatically loads embedded broad-corpus IDF)
guard = GuardInterface()

# Hook 1: Register initial task description and plan
task_description = "Locate item in WebShop and complete purchase"
proposed_plan = "1. Search for item\n2. Select options\n3. Click buy now"
plan_res = guard.on_plan_proposed(task_description, proposed_plan, metadata={"task_id": "webshop_01"})

# Non-blocking contract: 'approved' means no internal hard error, 'flagged' means guard found an issue
if plan_res["flagged"]:
    print(f"Plan Warnings: {plan_res['flags']}")
    print(f"Plan Suggestions: {plan_res['suggestions']}")

# Hook 2: Process execution steps inside the agent loop
history = []
step_record = {
    "step_index": 0,
    "reasoning": "Search for blue cotton sweater in size medium",
    "action_name": "search",
    "action_args": {"query": "blue cotton sweater medium"},
    "tool_response": "Found 15 items"
}

res = guard.on_step(step_record, history, metadata={"task_id": "webshop_01"})

# Primary recommended field for simple integrations:
if res["flagged"]:
    print(f"Warning: {res['warning']}")

# Advanced inspection path:
if res.get("match_details"):
    print(f"Pattern Match: {res['match_details']['category']} ({res['match_details']['confidence']:.2f})")
if res.get("drift_detected"):
    print(f"Drift Signals: {res['drift_assessment']['triggered_signals']}")
if res.get("reflection_result", {}).get("revision_suggested"):
    print(f"Plan Invalidation: {res['reflection_result']['revision_reasoning']}")

history.append(step_record)

# Hook 3: Fire on subgoal completion or transition
guard.on_subgoal_boundary(
    subgoal_id="subgoal_1",
    subgoal_status="completed",
    step_history=history,
    metadata={"task_id": "webshop_01"},
)

# Hook 4: Finalize run and retrieve root-cause attribution
summary = guard.on_run_end(
    metadata={"task_id": "webshop_01"}, 
    trajectory={"steps": history}
)

print(f"Root Cause Source: {summary['root_cause_source']}")  # 'pattern_match' | 'drift_monitor' | 'reflector' | 'none'
print(f"Root Cause Category: {summary['root_cause_error_type']}")
print(f"Root Cause Step Index: {summary['root_cause_step_index']}")
```

#### Centralized Configuration (`GuardConfig`)

`GuardInterface` accepts an optional `config: GuardConfig` parameter for structured parameter tuning. When no `config` object is passed, `GuardInterface()` automatically defaults to `GuardConfig.from_env()`, which reads active environment variables (e.g., `GUARD_MAX_SUBGOAL_STEPS`, `GUARD_DRIFT_THRESHOLD`, `GUARD_FAIL_OPEN`) or falls back to built-in system defaults.

Individual keyword arguments passed to `GuardInterface()` (`pattern_library_path`, `max_subgoal_steps`, `drift_threshold`, `category_thresholds`, `match_timeout`) take direct precedence and override values from `config`:

```python
from longhorizon_guard import GuardInterface, GuardConfig

# Approach 1: Structured configuration object
config = GuardConfig(
    max_subgoal_steps=12,          # S3 step ceiling before marking stalled_advanced (default: 10)
    drift_threshold=0.30,          # Severity threshold [0.0, 1.0] to flag drift_detected (default: 0.35)
    reflection_step_interval=4,    # Cadence for periodic plan validity checks (default: 5)
    fail_open=True,                # Catch internal errors without breaking agent (default: True)
)
guard = GuardInterface(config=config)

# Approach 2: Direct constructor keyword argument overrides
guard = GuardInterface(
    max_subgoal_steps=8,
    drift_threshold=0.40,
)
```

### 3. Automatic Integrations (Zero-Change & Middleware)

If you use LangChain / LangGraph or standard OpenAI-compatible client libraries, you do not need to manually instrument your agent loops. LongHorizon Guard provides two native adapters:

#### Path A: LangChain / LangGraph Callback Handler
Add `LongHorizonGuardCallback` to your agent executor or chain. It automatically intercepts `on_chain_start`, `on_agent_action`, `on_tool_end` / `on_tool_error`, and `on_chain_end`, translating events into `GuardInterface` step records and executing the lifecycle hooks transparently:

```python
from langchain.agents import create_agent  # or AgentExecutor
from longhorizon_guard import LongHorizonGuardCallback

# 1. Instantiate callback (non-blocking logging by default)
guard_callback = LongHorizonGuardCallback()

# Optional: supply an on_flag hook if your host application wants to actively intervene
def handle_guard_alert(step_res):
    print(f"Intervention needed! Guard flagged: {step_res['warning']}")

guard_callback = LongHorizonGuardCallback(on_flag=handle_guard_alert)

# 2. Pass directly to your LangChain agent — ZERO changes to your agent loop
agent = create_agent(..., callbacks=[guard_callback])
result = agent.invoke({"input": "Find the latest ACME report and extract numbers."})

# 3. Retrieve final trajectory summary
print(guard_callback.last_run_summary)
```

#### Path B: OpenAI-Compatible Client Middleware (`wrap_guard`)
For custom agents using `openai.OpenAI()` or any OpenAI-compatible client (such as local Ollama, vLLM, DeepSeek, or Groq), `wrap_guard()` intercepts `client.chat.completions.create()`:

```python
import openai
from longhorizon_guard import wrap_guard

# 1. Wrap your client once
client = wrap_guard(openai.OpenAI())

# 2. Run your normal multi-turn tool-calling loop unmodified
# The wrapper observes tool calls, correlates tool responses, and calls on_step() automatically
messages = [{"role": "user", "content": "Search for Python docs and summarize."}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=[...],
)

# ... your normal tool execution loop ...

# 3. Explicitly finalize when your task is complete
run_summary = client.finalize_run()
print(f"Root cause step: {run_summary['root_cause_step_index']}")
```

#### Honest Integration Boundaries: What is Automatic vs. What Still Requires Setup
- **LangChain / LangGraph (`LongHorizonGuardCallback`)**:
  - **Truly Automatic**: Plan extraction from initial inputs, step-by-step reasoning/tool/observation tracking, trajectory history maintenance, and `on_run_end` execution on chain finish.
  - **Requires Setup**: Active intervention. By default, the guard is non-blocking (logs warnings). If you want the agent to abort or re-prompt on flags, you must provide an `on_flag` callable.
- **OpenAI Client Wrapper (`wrap_guard`)**:
  - **Truly Automatic**: Multi-turn tool call correlation (pairs assistant `tool_calls` with subsequent `role: "tool"` messages), schema translation into `step_record`, and transparent pass-through of all API responses and errors.
  - **Requires Setup**:
    1. **Calling `finalize_run()`**: An LLM client has no universal concept of when an agent's multi-step task is complete (some finish in 1 call, some loop 20 times). You must call `client.finalize_run()` when your loop terminates.
    2. **Task/Plan Heuristic**: The wrapper treats the initial user message as the task description. If your agent uses a separate formal planning phase, explicit registration via `client.guard.on_plan_proposed(...)` gives higher precision than the heuristic.

---

## LLM-as-a-Judge Evaluation & Provider Configuration

LongHorizon Guard includes an automated, multi-provider LLM judge (`longhorizon_guard.taxonomy.judge`) to evaluate completed trajectories and benchmark root-cause attribution.

### 1. Code-Free Provider Setup (`providers.yaml` & `.env`)
You can configure or switch providers without writing any code:
1. Copy the secrets template to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
2. (Optional) Copy the provider template to `providers.yaml` to customize models, temperatures, or add local endpoints:
   ```bash
   cp providers.example.yaml providers.yaml
   ```

### 2. Supported Providers Out of the Box
- **Local / Self-Hosted Models**:
  - **Ollama**: Pre-configured (`--provider local_ollama` or `--base-url http://localhost:11434/v1`)
  - **vLLM / LMStudio**: Pre-configured (`--provider local_vllm`)
- **OpenAI-Compatible Cloud Gateways**:
  - **DeepSeek**: Pre-configured (`--provider deepseek`)
  - **OpenAI Official**: Pre-configured (`--provider openai` with `OPENAI_API_KEY`)
- **Native Direct APIs**:
  - **Google Gemini**, **Groq**, **Cloudflare Workers AI**, **OpenRouter**, **NVIDIA NIM**, **TokenRouter**.

### 3. Running the Judge CLI
```bash
# Run with local Ollama (zero API keys needed):
python -m longhorizon_guard.taxonomy.judge --provider local_ollama

# Run with an OpenAI-compatible endpoint on the fly:
python -m longhorizon_guard.taxonomy.judge \
    --provider openai-compatible \
    --base-url http://localhost:11434/v1 \
    --model qwen2.5:14b

# Run with Groq or Gemini using a custom model override:
python -m longhorizon_guard.taxonomy.judge --provider groq --model llama-3.3-70b-versatile
python -m longhorizon_guard.taxonomy.judge --provider gemini --model gemini-2.5-flash
```

---

## Known Limitations

### Semantic / Logical Constraint Checking
The current detection engine (combining TF-IDF centroid pattern matching, keyword rule matching, and structural loop heuristics) detects known error patterns, action repetitions, unresponsive tool responses, subgoal progression stalls, and plan divergence. 

However, **it cannot detect arbitrary semantic or logical constraint violations where execution is syntactically valid and free of error keywords**.

* **Concrete Example:**
  Suppose a user requests:
  > *"Book a flight arriving strictly before 10:00 AM on Monday with zero layovers."*

  If the agent executes an action:
  ```json
  {
    "action_name": "select_flight",
    "action_args": {"flight_id": "FL-402", "arrival_time": "11:45 AM", "day": "Monday", "layovers": 1},
    "tool_response": "Flight FL-402 reserved successfully."
  }
  ```
  The syntax is valid, tool execution succeeds without errors, and no failure keywords or known centroid patterns match. The system cannot currently verify that `11:45 AM` contradicts the task constraint `before 10:00 AM`, or that `1 layover` violates `zero layovers`.
* **Roadmap:**
  Semantic constraint verification is a known gap reserved for a future verification layer (e.g. an LLM-based formal constraint checker or AST state invariant validator), not implemented in this version.

---

## Verification

Run the test suite:
```bash
pytest tests/ -v
```
