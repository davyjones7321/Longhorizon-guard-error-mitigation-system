# LongHorizon Guard: Long-Horizon Agent Error Mitigation & Causal Memory

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-105%20Passed%20(100%25)-success.svg)](file:///d:/edge-downloades/d/intership-projects/error-prop/tests)
[![Architecture](https://img.shields.io/badge/Architecture-3--Tier%20HippoRAG%20Causal%20Memory-purple.svg)](file:///d:/edge-downloades/d/intership-projects/error-prop/longhorizon_guard/memory)

**LongHorizon Guard** is a real-time error mitigation, trajectory governance, and causal memory framework designed for autonomous LLM coding agents (**OpenAI Codex**, **Claude Code**, **Cursor**, **Aider**, and **OpenCode**). 

In complex, multi-step tasks, autonomous agents rarely fail because of a single catastrophic bug; instead, **small, unnoticed mistakes at Step 2 propagate into trajectory drift by Step 6 and total task collapse by Step 12**. LongHorizon Guard intercepts agent execution step-by-step, validates action prerequisites, tracks multi-horizon drift, and uses a neurobiologically inspired **HippoRAG causal knowledge graph** to discover reachable recovery actions before mistakes cascade.

---

## Academic Foundations & Literature Citations

LongHorizon Guard synthesizes four foundational lines of academic research across LLM agent failure taxonomy, structural debugging, reinforcement learning, and neurobiological associative memory:

### 1. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models
* **Authors:** Bernal Gutiérrez, et al. (Ohio State University & Stanford University, 2024)
* **Citation:** NeurIPS 2024 | [arXiv:2405.14831](https://arxiv.org/abs/2405.14831)
* **What We Adopted:**
  * **Dual-Memory Neurobiological Architecture**: Mimicking the mammalian brain's division between the neocortex (short-term working buffer) and hippocampus (structured relational index). LongHorizon Guard splits state into an in-process **`WorkingMemory`** and an indexed **`CausalErrorGraph`**.
  * **Personalized PageRank (PPR) Associative Diffusion**: Rather than relying strictly on dense vector embedding similarity (which misses multi-hop causal chains), LongHorizon Guard implements HippoRAG's graph diffusion algorithm. It injects personalized teleportation energy at active action/tool nodes and diffuses probability mass across typed causal edges to calculate downstream error risk and find reachable recovery strategies in **<0.5ms** (well below the 15ms agent timeout budget).

### 2. AgentErrorBench: Evaluating and Mitigating Failure Propagation in Language Agent Workflows
* **Authors:** Tsinghua University & Zhipu AI (2024)
* **Citation:** [arXiv:2410.15836](https://arxiv.org/abs/2410.15836) | [GitHub](https://github.com/THUDM/AgentErrorBench)
* **What We Adopted:**
  * Root-cause error taxonomy: Standardized five-category taxonomy (`planning_error`, `reflection_error`, `memory_error`, `tool_use_error`, `external_error`).
  * Benchmark evaluation methodology across ALFWORLD, WebShop, and GAIA trajectories.

### 3. AgentDebug: Fine-Grained Error Detection and Localization for LLM Agents
* **Authors:** Fudan University, et al. (2024)
* **Citation:** [arXiv:2409.11727](https://arxiv.org/abs/2409.11727)
* **What We Adopted:**
  * Fine-grained structural detectors for action repetition loops, observation stagnation ("nothing happens" cycles), and early step localization of initial root causes.

### 4. Reflexion: Language Agents with Verbal Reinforcement Learning
* **Authors:** Shinn, et al. (Northeastern University, MIT, Princeton, 2023)
* **Citation:** [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
* **What We Adopted:**
  * Dynamic plan invalidation triggers and verbal reflection summaries driven by accumulated subgoal stalls and trajectory drift acceleration.

---

## 3-Tier HippoRAG Causal Memory Architecture

Standard RAG systems retrieve documents using vector similarity, but **vector search cannot understand causality or multi-hop dependency chains** (e.g., *“Deploying to staging failed because database migrations were never executed during the build phase”*). LongHorizon Guard implements a 3-tier memory system designed for agent causality:

```text
                           ┌────────────────────────┐
                           │     GuardInterface     │
                           └──────────┬─────────────┘
                                      │
                                      ▼
                           ┌────────────────────────┐
                           │      MemoryGuard       │
                           │     (Coordinator)      │
                           └──────┬───────────┬─────┘
                                  │           │
            ┌─────────────────────┴───┐       │
            ▼                         ▼       ▼
 ┌───────────────────────┐ ┌─────────────────────────┐ ┌───────────────────────┐
 │     WorkingMemory     │ │    CausalErrorGraph     │ │   LocalConceptIndex   │
 │  - Sliding Step Window│ │  - NetworkX DiGraph     │ │  - Sparse TF-IDF Cosine │
 │  - Drift Velocity/Acc │ │  - Prerequisite Edges   │ │  - Task Similarity      │
 │  - Loop Signature DB  │ │  - Multi-Hop Cascades   │ │  - Zero External API    │
 └───────────────────────┘ └───────────┬─────────────┘ └───────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────────┐
                           │ AssociativeMemoryEngine │
                           │ - HippoRAG Personalized │
                           │   PageRank (PPR <0.5ms) │
                           │ - Multi-Hop Causal Risk │
                           │ - Recovery Path Search  │
                           └─────────────────────────┘
```

### Tier 1: Ephemeral Working Memory (`WorkingMemory`)
Maintained in-process during active execution:
* **Sliding Action Window**: Retains recent action signatures and arguments.
* **Repetition Counter**: Detects repeated action calls with identical arguments that failed, catching loops before token exhaustion.
* **Kinematic Drift Monitoring**: Calculates first-derivative (drift velocity) and second-derivative (drift acceleration) across subgoal intervals.
* **Advisory State**: Maintains unacknowledged warnings and steering signals.

### Tier 2: Causal Error Knowledge Graph (`CausalErrorGraph`)
A directed multigraph backed by NetworkX (`nx.DiGraph`), persisted locally to JSON (`findings/memory/causal_graph.json`) or in-memory (`:memory:`):
* **No External Database Required**: Operates completely embedded with zero Neo4j, Redis, or Docker daemons.
* **Graph Node Entities**:
  * `TaskConceptNode`: High-level user tasks and requirements.
  * `SubgoalNode`: Plan phases with topological prerequisite constraints.
  * `ActionPatternNode`: Normalized tool patterns and parameter templates.
  * `ErrorSignatureNode`: Classified error taxonomy categories and error regexes.
  * `RecoveryNode`: Prescribed corrective actions and safe tool alternatives.
* **Relational Causal Edges**:
  * `PREREQUISITE_OF`: Topological dependencies between milestones (e.g., `run_tests` → `deploy_service`).
  * `TRIGGERS_ERROR`: Direct causal edge linking an action to an observed failure signature.
  * `PROPAGATES_TO`: Multi-step cascade edge modeling error evolution (e.g., `planning_error` → `tool_use_error` → `drift`).
  * `REMEDIED_BY`: Edge mapping an error node to an effective recovery action.

### Tier 3: HippoRAG Associative Diffusion Engine (`AssociativeMemoryEngine`)
Implements **Personalized PageRank (PPR)** over the causal graph:
1. When an agent proposes an action, the engine seeds personalized teleportation probability on matching `ActionPatternNode` entities.
2. Probability mass diffuses across all causal and cascade edges.
3. The engine computes associative error risk scores and retrieves reachable `RecoveryNode` safe alternatives in **<0.5ms**.
4. **Capability Normalization (`capability_classifier.py`)**: Real tool names (`shell`, `Bash`, `exec_command`, `powershell`, `cmd`) and natural task phrasings are automatically normalized into canonical capabilities, ensuring checks never fail due to harness-specific naming differences.

---

## Core System Architecture

LongHorizon Guard operates through a modular, fail-open 4-hook lifecycle interface:

```text
  Host AI Agent (Codex / Claude Code / Cursor / Aider)
             │
             +---> 1. on_plan_proposed(task_description, proposed_plan)
             │          - Parses subgoals & verifies prerequisite order
             │          - HippoRAG checks plan against known failure patterns
             │
             +---> 2. on_step(step_record, history)
             │          │
             │          +---> Layer A: Broad-Corpus TF-IDF Pattern Matcher (5,454 IDF terms)
             │          +---> Layer B: Structural & Heuristic Rule Detectors (Loop/Repetition)
             │          +---> SubgoalTracker: Enforces Rule S3 & phase transitions
             │          +---> DriftMonitor: Calculates drift severity (none/low/med/high/crit)
             │          +---> PlanReflector: Triggers plan reflection on high drift
             │
             +---> 3. on_subgoal_boundary(subgoal_id, status)
             │          - Evaluates milestone completion & updates drift velocity
             │
             +---> 4. on_run_end(metadata, trajectory)
                        - Correlates root cause errors with downstream drift
                        - Writes error cascades to persistent Causal Graph
                        - Emits complete audit summary
```

### Safety & Reliability Contract (Fail-Open Guarantee)
Every hook invocation is wrapped in resilient exception boundaries with a strict **2.0-second timeout limit**. If any internal monitor, graph lookup, or matcher raises an exception, the guard logs a diagnostic warning and **fails open** (`flagged=False`), guaranteeing that **LongHorizon Guard will never crash your primary AI coding assistant**.

---

## Live Agent Integrations

LongHorizon Guard natively supports modern terminal coding agents via lifecycle command hooks (`UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`).

### 1. OpenAI Codex (First-Class Default)

OpenAI Codex is the default native environment for LongHorizon Guard.

#### Automatic 1-Click Setup
Run this single command from your Python environment:
```bash
longhorizon-guard setup-codex
```
* Automatically locates `~/.codex/config.toml`.
* Cleans up legacy/conflicting hook configurations.
* Registers all four lifecycle hooks with the active Python binary.

#### Manual Configuration
Add the following tables to `~/.codex/config.toml`:
```toml
[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = "python -m longhorizon_guard.hook"
timeout = 30
statusMessage = "LongHorizon Guard is checking the task"

[[hooks.PreToolUse]]
matcher = ".*"
[[hooks.PreToolUse.hooks]]
type = "command"
command = "python -m longhorizon_guard.hook"
timeout = 30
statusMessage = "LongHorizon Guard is checking the action"

[[hooks.PostToolUse]]
matcher = ".*"
[[hooks.PostToolUse.hooks]]
type = "command"
command = "python -m longhorizon_guard.hook"
timeout = 30
statusMessage = "LongHorizon Guard is reviewing the result"

[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = "python -m longhorizon_guard.hook"
timeout = 30
statusMessage = "LongHorizon Guard is finalizing the run"
```

#### Run Codex with Guard Active
```bash
codex
```
*(On first run, press down-arrow to highlight **"2. Trust all and continue"** and press Enter. Codex will remember your choice).*

---

### 2. Claude Code Integration

Claude Code hooks use the identical wire protocol as Codex and work seamlessly with the same engine.

#### Automatic 1-Click Setup
```bash
# Configure current project (.claude/settings.json):
longhorizon-guard setup-claude

# OR configure globally (~/.claude/settings.json):
longhorizon-guard setup-claude --global
```

#### Manual Configuration
Add this hook block to `.claude/settings.json` (or `~/.claude/settings.json`):
```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "python -m longhorizon_guard.hook",
        "timeout": 30
      }
    ],
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python -m longhorizon_guard.hook",
            "timeout": 30
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python -m longhorizon_guard.hook",
            "timeout": 30
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m longhorizon_guard.hook",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

#### Run Claude Code
```bash
claude
```
Select **"Always Allow"** when Claude Code prompts for hook authorization on the first turn.

---

### 3. Real-Time API Proxy (Cursor, Aider, OpenCode)

For IDE-based agents without native lifecycle hooks, LongHorizon Guard provides a non-intrusive reverse proxy:

```bash
longhorizon-guard proxy --port 8000 --upstream https://api.openai.com/v1
```

Point your agent to the proxy:
```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
```
The proxy intercepts reasoning steps and tool calls transparently in the background, writing audit logs to `findings/proxy_sessions/`.

---

## Installation

### Prerequisites
* Python 3.9 or higher

### Direct Install from GitHub
```bash
pip install git+https://github.com/davyjones7321/Longhorizon-guard-error-mitigation-system.git
```

### Local / Development Install
```bash
git clone https://github.com/davyjones7321/Longhorizon-guard-error-mitigation-system.git
cd Longhorizon-guard-error-mitigation-system
pip install -e .
```

---

## Python API Usage

Incorporate `GuardInterface` into custom agent loops or evaluation pipelines:

```python
from longhorizon_guard import GuardConfig, GuardInterface

# Initialize with Causal Graph Memory enabled
config = GuardConfig(
    enable_memory=True,
    memory_storage_path="findings/memory/causal_graph.json",
    drift_threshold=0.35,
)
guard = GuardInterface(config=config)

# 1. Propose Plan
plan_res = guard.on_plan_proposed(
    task_description="Build priority queue and run tests",
    proposed_plan="1. Implement queue.py\n2. Run automated tests",
)
if not plan_res["approved"]:
    print(f"Plan Warning: {plan_res['flags']}")

# 2. Execute Steps
step_record = {
    "step_index": 0,
    "action_name": "Bash",
    "action_args": {"command": "python -m unittest"},
    "tool_response": "Ran 5 tests in 0.01s... OK",
}
step_res = guard.on_step(step_record, history=[])
if step_res["flagged"]:
    print(f"Step Flagged: {step_res['warning']}")

# 3. Complete Trajectory
summary = guard.on_run_end(
    metadata={"task_id": "task_01"},
    trajectory={"steps": [step_record]},
)
print(f"Run Outcome: {summary.get('run_status')}, Drift: {summary.get('drift_score')}")
```

---

## CLI Command Reference

| Command | Description |
| :--- | :--- |
| `longhorizon-guard setup-codex` | Automatically configures `~/.codex/config.toml` for OpenAI Codex. |
| `longhorizon-guard setup-claude` | Automatically configures `.claude/settings.json` for Claude Code. |
| `longhorizon-guard info` | Displays loaded pattern centroids, IDF terms, and memory status. |
| `longhorizon-guard evaluate -t <file.json>` | Evaluates an offline JSON trajectory file for failure propagation. |
| `longhorizon-guard proxy -p 8000` | Starts the real-time HTTP monitoring proxy. |

---

## Verification & Testing

Execute the complete test suite across the memory graph, associative engine, classifier, and hook integrations:

```bash
python -m pytest tests/ -v
```

```text
============================= 105 passed in 7.81s =============================
```
