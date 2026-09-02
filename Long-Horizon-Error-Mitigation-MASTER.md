# Long-Horizon Error Mitigation — Master Reference
### From Taxonomy to Fine-Tuning: Everything We're Building, In One File

**What this file is:** the single consolidated reference for the whole project — every paper we're drawing from, every layer we're building, how they connect, how it should be packaged for reuse, and the current operational status/build prompt for the phase we're on right now. If you only keep one file from this whole effort, keep this one.

**Papers this draws from** (see the original methodology reference for full detail):
- **Paper 2 (AgentDebug)** — root-cause tagging of failed trajectories → Layer 1
- **Paper 3 (PreFlect)** — pre-execution + periodic plan checking against known error patterns → Layers 2 & 3
- **Paper 4 (Subgoal-Driven Planning)** — breaking tasks into checkpoints to contain errors locally → Layer 4
- **Distillation** (general technique, not a single paper) — compressing everything learned into a small, cheap checker model → Layer 5

---

## Table of Contents

- **PART I** — Current Status & What's Actually Built So Far
- **PART II** — The Five Layers: Architecture & Implementation (Taxonomy → Fine-Tuning)
  - 0. The big picture
  - 1. Layer 1 — Logging & Root-Cause Tagging (AgentDebug)
  - 2. Layer 2 — Pre-Flight Plan Checker (PreFlect)
  - 3. Layer 3 — In-Flight Drift Monitor (PreFlect, periodic)
  - 4. Layer 4 — Subgoal Decomposition
  - 5. Layer 5 — Distillation / Fine-Tuning the Small Checker Model
  - 6. How the five layers connect, end to end
  - 7. Suggested repo structure (feature-complete system)
  - 8. Suggested rollout order
  - 9. Glossary
- **PART III** — Packaging as a Standalone Library (multi-harness, multi-model reuse)
- **PART IV** — Operational Build Playbook (the exact next phases: data cleanup → tagger → scaffold)
- **PART V** — Open Decisions & Known Bottlenecks (read before building anything)

---

## PART I — Current Status & What's Actually Built So Far

*(Update this section yourself as things change — it's meant to be a living snapshot, not a static record.)*

**Built and confirmed working:**
- Trajectory + metadata logging (Layer 1, Step 1.1) — recording agent runs correctly across horizon levels L1–L4, using an eval harness with `chain_l*` tasks.

**Built but not yet run / not yet wired in:**
- A manual tagging CLI (`tagger.py`) exists with 6 categories (`planning_error`, `memory_error`, `tool_use_error`, `external_error`, `grader_error`, `other`) but has never been invoked — `tags.csv` doesn't exist anywhere yet.

**Not built yet:**
- Automated LLM-as-judge tagger (Layer 1, Steps 1.3–1.5)
- Schema fields for root-cause tracking (Layer 1, Step 1.6)
- Everything in Layers 2–5
- The standalone library packaging (Part III)

**Known issue blocking progress, and the current plan to work around it:**
- A full audit (110 runs across `eval/output/`, `output/`, `findings/`) found **zero tagged failures anywhere**, and found that L4's 72% error rate (and the smaller error rates at L1–L3) is **100% OpenRouter free-tier 429 rate-limit noise**, not genuine agent reasoning failures. There are currently zero confirmed genuine reasoning failures in the dataset.
- Current plan: consolidate existing genuine runs (across all folders, including older recorded batches) into one clean set, filtering out all rate-limit/infra noise, to get to a working calibration set (target ~50–100 genuine runs). See **Part IV** for the exact build prompt for this.
- Longer-term plan: once the guard is packaged as a library (Part III), deploy it inside Claude Code / Antigravity sessions as the primary source of realistic long-horizon trajectories going forward, since that sidesteps the shared free-tier quota entirely. The synthetic arithmetic-chain benchmark remains useful specifically for tagger calibration (Layer 1, Step 1.2/1.3), since it has ground-truth graders that real coding sessions don't.

---

## PART II — The Five Layers: Architecture & Implementation (Taxonomy → Fine-Tuning)

**Purpose:** the step-by-step build plan for all five layers, in build order, with plain-English explanations before implementation details for non-technical teammates.

## 0. The big picture, in plain English

Imagine an AI agent doing a long task with 20 steps. If it makes a small mistake on step 3, that mistake doesn't just sit there — it snowballs, and by step 20 the whole thing is broken. This is "error propagation."

We're building a system with **four layers of defense**, plus a final step that compresses everything we learn into a small, cheap model:

| Layer | Question it answers | Analogy |
|---|---|---|
| 1. Logging & Root-Cause Tagging | "What went wrong, and where exactly did it start?" | A flight recorder / black box |
| 2. Pre-Flight Plan Check | "Before the agent acts, does this plan look like a mistake we've seen before?" | A pilot's pre-flight checklist |
| 3. In-Flight Drift Monitor | "While the agent is running, is it starting to go off track?" | Air traffic control checking in periodically |
| 4. Subgoal Decomposition | "Can we break the big task into small checkpoints, so one bad step doesn't ruin everything?" | Checkpoints in a video game instead of one long life |

Later, once we've collected enough data from Layers 1–4, we **train a small, cheap model** whose only job is to sit in front of our main agent and say "this plan looks risky" or "this plan looks fine" — like a spell-checker, but for agent plans.

**Nothing in this system exists yet — Layers 1 through 5 all need to be built.** Layer 1 (logging + root-cause tagging) comes first because every other layer depends on its data, but it is not done. This guide covers all five layers, in build order, with plain-English explanations and concrete steps for each.

---

## 1. Layer 1 — Build the Logging & Root-Cause Tagging System (AgentDebug, Paper 2)

### 1.1 Plain-English explanation

Before we can catch or prevent errors, we need a reliable way to **record exactly what the agent did, step by step, on every run** — and then, for runs that failed, **figure out which single step actually caused the failure** (as opposed to every small mistake that happened along the way).

Think of it as installing a flight recorder (black box) on the agent, plus a crash investigator who reviews the recording afterward and points to the one moment things went wrong — not just "something went wrong somewhere in the last ten minutes."

This is the foundation everything else in this guide is built on. Layers 2–4 are not separate tools bolted onto the agent — they are all **consumers of this same trajectory data**, plus **producers of new data in the same format**. So the single most important rule for the whole project is:

> **Every component we build (Layer 1's tagger, the planner-checker, drift monitor, subgoal tracker) must read from and write back to the same trajectory/metadata store.** That's what lets us eventually train the small model in Layer 5 — it needs one consistent dataset, not several disconnected logs.

### 1.2 The three things to build

**A. Trajectory & metadata logging** — instrument the agent harness so every run writes out two files:
- `metadata.json` — one entry per trial: task ID, horizon level, trial number, final status (success/fail), duration, any grading notes.
- `trajectory.json` — the step-by-step play-by-play of that one run: every action the agent took, every tool call it made, and the result of each.

**B. The Error Taxonomy** — a fixed, agreed-upon list of error categories, so that when something goes wrong, everyone (and every tool downstream) uses the same labels. Use the five categories from the AgentDebug paper:
- `memory` — the agent misremembered or lost track of a fact from earlier in the run.
- `reflection` — the agent misjudged its own progress (e.g., thought it was done when it wasn't).
- `planning` — the plan itself was flawed from the start (wrong approach, missing a constraint).
- `action` — the plan was fine, but the agent executed a step incorrectly (wrong tool call, bad input).
- `system` — the failure was external (a tool/API error, a timeout, an environment issue) rather than the agent's own reasoning.

**C. The Root-Cause Debugger** — a 3-step automated process that runs on failed trajectories:
1. **Tag every step** in the trajectory with one of the five error-taxonomy labels above (or `null` if that step looks fine).
2. **Find the single earliest step** that was tagged with an error and that plausibly started the chain of failure — this becomes the `root_cause_step_index`.
3. **Re-run the agent from that step**, giving it specific corrective feedback about the mistake, and repeat a few times if it still fails, refining the feedback each time.

### 1.3 Step-by-step build plan

**Step 1.1 — Instrument the harness to write trajectory + metadata files**

- For every step the agent takes (each tool call, each intermediate reasoning output, each result), append an entry to that run's `trajectory.json`. At minimum, capture: `step_index`, `action`, `tool_call`, `result`.
- At the end of the run, write `metadata.json` with the run-level summary: `task_id`, `trial_number`, `horizon_level`, `total_steps_taken`, `final_status`, `duration_seconds`, `grader_notes`.
- This should be a wrapper around your existing run loop — you're not changing how the agent behaves yet, only recording what it does.

**Step 1.2 — Build a small labeled benchmark of real failures**

Before you can auto-tag anything, you need examples of what each error category actually looks like. Manually review a batch of failed runs (100–200 is a reasonable starting target, per the paper) and hand-label:
- which step was the true root cause,
- which of the five categories it belongs to.

This labeled set does two jobs: it's what you'll use to sanity-check the automated tagger in Step 1.3, and it's your first real training data for Layer 5 later on.

**Step 1.3 — Build the automated tagger (an LLM-as-judge)**

- Write a prompt that takes a full trajectory and asks an LLM to go step by step and assign one of the five taxonomy labels (or none) to each step, with a short justification.
- Feed it the same runs you hand-labeled in Step 1.2, and compare its output to your human labels. Adjust the prompt until it agrees with your hand-labels closely and consistently — this is the same idea as the paper's "check the judge against human reviewers" step, and it's the single most important quality gate before you trust this tagger on the rest of your data.

**Step 1.4 — Build the root-cause finder**

- Among all the steps tagged with an error in a given trajectory, identify the **earliest** one as the root cause — not the last one, not the "most severe" one. The paper's key insight is that the earliest error is what actually started the cascade; fixing later symptoms without fixing this one rarely helps.
- Write this back into `metadata.json` as `root_cause_step_index` and `root_cause_error_type`.

**Step 1.5 — Build the corrective re-run loop**

- Once a root cause is identified, re-run the agent starting from that step, injecting a short, specific piece of corrective feedback (e.g., "You misjudged progress here — you had not actually verified the file existed before reporting success").
- If it fails again, tag the new failure the same way and refine the feedback; cap this at a small number of retries (2–3) so it doesn't loop forever.
- Log every retry attempt into the trajectory so you can see how many corrective passes it took, and whether it eventually succeeded.

**Step 1.6 — Schema additions (do this before starting Layer 2)**

Once the above is working, extend the schema with a few extra fields so Layers 2–4 have somewhere to write their own results later. This is a small, low-risk change to make now while you're already in this code:

```json
// metadata.json — add these fields
{
  "task_id": "chain_l2_a",
  "trial_number": 1,
  "horizon_level": 2,
  "total_steps_taken": 5,
  "final_status": "success",

  // NEW fields for this project:
  "root_cause_step_index": null,     // filled in by Layer 1's debugger (AgentDebug step 2)
  "root_cause_error_type": null,     // one of: memory | reflection | planning | action | system
  "preflect_flags": [],              // filled in by Layer 2 (list of flagged risky steps, if any)
  "drift_flags": [],                 // filled in by Layer 3 (periodic checks that found drift)
  "subgoal_plan": [],                // filled in by Layer 4 (list of subgoal strings)
  "subgoal_completion": []           // filled in by Layer 4 (which subgoals were hit, in order)
}
```

```json
// trajectory.json — add this per-step field
{
  "step_index": 3,
  "action": "...",
  "tool_call": "...",
  "result": "...",

  // NEW:
  "error_tag": null,        // memory | reflection | planning | action | system | null
  "is_root_cause": false,   // true for the single step AgentDebug identifies as the origin
  "subgoal_id": null        // which subgoal (from Layer 4) this step belongs to
}
```

This is the "data contract" the rest of the team should agree on before writing any new code — everyone reads/writes the same file shape.

---

## 2. Layer 2 — The Pre-Flight Plan Checker (PreFlect, Paper 3)

### 2.1 Plain-English explanation

Right now, when your agent fails, you find out *after* the fact — the run finishes, you look at the trajectory, you see it went wrong. That's useful for learning, but it doesn't stop the *next* bad run from happening.

PreFlect flips this around: **before the agent takes its first action, we show its plan to a separate "reflector" and ask: "Have we seen a plan like this go wrong before?"** If yes, we send the plan back for a rewrite *before anything irreversible happens* (an email gets sent, a file gets deleted, an order gets placed).

Think of it as a second pair of eyes that reviews the plan before it's executed — like a code review that happens before you merge, not after something breaks in production.

### 2.2 Two moving parts

You need to build **two things**, and they connect to each other:

**A. An "Error Pattern Library" (a small memory/database)**
This is built *offline*, from the data you already logged in Layer 1. It's simply: "here is a list of past plans/steps that were tagged as `planning` errors (root_cause_error_type == 'planning'), summarized into reusable patterns."

**B. A "Reflector" step in your agent's loop**
Before your agent executes a plan, you make one extra LLM call: "here is the plan the agent is about to run, here are known risky patterns — does this plan resemble any of them?" If yes, the agent revises its plan before acting.

### 2.3 Step-by-step build plan

**Step 2.1 — Build the Error Pattern Library**

- Pull every trajectory where `root_cause_error_type == "planning"` (or `"reflection"`, which is closely related) from your Layer 1 data.
- For each one, write a short 1–2 sentence summary of *what the plan did wrong* — e.g., "Agent chose to overwrite the file directly instead of checking if a backup existed first" or "Agent skipped verifying a unit conversion before using it in later steps."
  - This summarization can be done by an LLM call (cheap model is fine) — feed it the trajectory around the root-cause step and ask it to produce one plain-English "planning error pattern" sentence.
- Store these pattern summaries in a lightweight vector database (see box below) so they can be searched by similarity later.

> **For non-technical teammates: what's a "vector database"?**
> It's a searchable memory. Instead of searching by exact keyword match, it searches by *meaning* — so if a new plan says "delete the old log file" and a stored pattern says "removed a file without checking for a backup first," the database will surface that pattern even though the wording is different. Tools like Chroma, FAISS, or Pinecone all do this; any one is fine to start with, Chroma is the easiest to run locally with no extra infrastructure.

**Step 2.2 — Build the Reflector call**

Add a new step to your agent loop, *before* execution of each new plan:

```
1. Agent proposes a plan (as it does today).
2. NEW: Reflector step —
     a. Take the proposed plan.
     b. Query the Error Pattern Library for the top 3–5 most similar past planning-error patterns.
     c. Send the LLM a prompt like:
          "Here is a proposed plan: {plan}
           Here are patterns that caused failures in similar past tasks: {patterns}
           Does this plan risk repeating any of these mistakes? If yes, say which one and
           suggest a safer revision. If no, say 'plan looks safe.'"
     d. If flagged: log it to `preflect_flags` in metadata.json, and send the plan back to
        the agent for a revision (one retry loop, not infinite).
     e. If not flagged: proceed to execution as normal.
3. Agent executes the (possibly revised) plan.
```

**Step 2.3 — Log everything, even the "plan looks safe" cases**

This is important and easy to skip: log every reflector check, not just the ones that catch something. You need the "no problem found" cases too, later, to train the small model in Step 5 — otherwise it will only ever learn what "risky" looks like, never what "safe" looks like.

**When to run this:** only before actions that are costly or irreversible is the minimum viable version (sending a message, deleting/overwriting data, making a purchase, calling an external API with side effects). You can expand it to run before every plan once you've measured the added latency/cost is acceptable.

---

## 3. Layer 3 — The In-Flight Drift Monitor (also PreFlect-style, run periodically)

### 3.1 Plain-English explanation

Layer 2 checks the plan *once*, at the start. But long tasks change as they go — the agent might start out fine and then drift off course by step 12. Layer 3 is the same idea as Layer 2 (compare against known error patterns), but run **repeatedly, at intervals, throughout the task**, instead of just once at the beginning.

Think of Layer 2 as a pre-flight checklist, and Layer 3 as air traffic control periodically checking "are you still on course?" throughout the flight.

### 3.2 How it's different from Layer 2 (important distinction)

| | Layer 2 (Pre-Flight) | Layer 3 (Drift Monitor) |
|---|---|---|
| When it runs | Once, before the plan is executed | Every N steps, *during* execution |
| What it looks at | The proposed plan | The plan **vs. what has actually happened so far** |
| What it's catching | A bad plan before it starts | The agent quietly wandering from the plan, or repeating the early stages of a known failure pattern |
| Failure it prevents | "This plan was doomed from the start" | "This plan was fine, but execution has gone sideways" |

### 3.3 Step-by-step build plan

**Step 3.1 — Decide the check interval**

Simplest approach: every N steps (e.g., every 3–5 steps for a 20-step task), or every time a subgoal completes (once Layer 4 is built, this becomes the natural checkpoint — see Section 4). Start with a fixed step interval; switch to subgoal-based checkpoints once Layer 4 exists.

**Step 3.2 — Build the drift-check prompt**

At each checkpoint, send the LLM:

```
- The original plan
- What has actually happened so far (the trajectory up to this point)
- The same Error Pattern Library used in Layer 2 (reuse it — don't build a second one)

Ask: "Comparing what has actually happened to the original plan and to known failure
patterns — is this run drifting off track, repeating an early stage of a known failure,
or showing signs of the same root-cause categories we've tagged before (memory,
reflection, planning, action, system)? If yes, describe the drift and suggest a
correction. If no, say 'on track.'"
```

**Step 3.3 — Decide what happens on a flagged drift**

Three options, in increasing order of intervention (pick one to start, based on how costly a wrong pause is):
1. **Log only** — record the drift in `drift_flags`, don't interrupt the run. Good for the first 1–2 weeks so you can see how often it fires without false-alarming on live runs.
2. **Soft correction** — inject the drift warning into the agent's next reasoning step as extra context, let it self-correct.
3. **Hard stop / replan** — pause the run and trigger a full replan (this overlaps with what Layer 4 does at the subgoal level — prefer Layer 4's local replanning once it exists, since a full replan is expensive).

**Step 3.4 — Log to the same schema**

Same rule as Layer 2: log every check, flagged or not, into `drift_flags` in metadata.json, timestamped by step index. This is what eventually lets you plot "at what point in long tasks does drift usually start?" — genuinely useful on its own, even before Step 5.

---

## 4. Layer 4 — Subgoal Decomposition (Paper 4)

### 4.1 Plain-English explanation

Right now the agent plans the *whole* task in one go. The problem: if step 3 of a 20-step task goes wrong, the agent has no clean way to "reset" — it either barrels forward with a broken plan, or has to re-plan the entire 20 steps from scratch (expensive and error-prone).

The fix: break the big goal into a short checklist of **subgoals** up front (e.g., "open the map" → "search location" → "filter results" → "report answer"). If something goes wrong, **only the current subgoal gets re-planned**, not the whole task. This keeps errors contained instead of letting them spread across the whole run.

### 4.2 Step-by-step build plan

**Step 4.1 — Add a subgoal-planning step before the main plan**

Before the agent starts executing, add one LLM call: "Break this task into 3–8 subgoals, each a short checkpoint." Store this list as `subgoal_plan` in metadata.json.

**Step 4.2 — Track progress against subgoals**

At every step, instead of just "what do I do next," the agent (or a lightweight wrapper around it) asks itself three questions:
1. Which subgoals have I already completed?
2. Have I finished the current one?
3. What's next?

Log completed subgoals into `subgoal_completion`, and tag each trajectory step with which `subgoal_id` it belongs to (this is the field we added to the schema in Section 1.1).

**Step 4.3 — Local replanning on failure**

If a step fails or a drift is flagged (Layer 3) within a subgoal, **only replan that subgoal** — not the entire task. This is the core benefit of this layer: it turns "start over" into "redo this one small piece."

**Step 4.4 — Use subgoal boundaries as your Layer 3 checkpoints**

Once this exists, switch Layer 3's "check every N steps" rule to "check at every subgoal boundary" — it's a more meaningful checkpoint than an arbitrary step count, and it's free (you're already tracking these transitions).

**Optional, later:** the paper also describes training a version of the agent (they call it MiRA) that gets rewarded for hitting each subgoal, not just for finishing the whole task. This requires reinforcement-learning-style training and is a bigger lift — treat it as a stretch goal after the prompting-level version (4.1–4.4) is working and you've measured it helps.

---

## 5. Layer 5 — Distilling Everything Into a Small Planner/Checker Model

### 5.1 Plain-English explanation

By the time Layers 1–4 are running in production, you'll have a large, growing dataset of:
- Trajectories tagged with root-cause errors (Layer 1)
- Plans that were flagged as risky vs. safe, and why (Layer 2)
- Drift checks, flagged and unflagged, with what "drift" looked like (Layer 3)
- Subgoal plans and how well they were followed (Layer 4)

Right now, all of Layers 2 and 3 rely on calling a general-purpose LLM every time they need to check a plan. That's slow and adds cost on every single run. The final step is to **train a small, cheap model whose only job is this one narrow task**: "look at a plan (and optionally what's happened so far) and say whether it resembles a known failure pattern." This small model then *replaces* the general-purpose LLM calls in Layers 2 and 3 — same job, much cheaper and faster, because it's specialized instead of general.

This is exactly what the methodology document's Paper 5 (Horizon Reduction) points at from a different angle — fewer, cheaper decision points — and it's the natural payoff of everything you logged in Layers 1–4.

### 5.2 Step-by-step build plan

**Step 5.1 — Assemble the training dataset**

Pull from your combined `metadata.json` / `trajectory.json` store:
- **Inputs:** the plan (or partial trajectory, for drift cases) that was checked.
- **Labels:** whether it was flagged (Layer 2/3's verdict) and, if flagged, which error category it matched (from the Layer 1 taxonomy: memory / reflection / planning / action / system).
- Make sure you have a healthy mix of flagged **and** unflagged examples (this is why Step 2.3 and Step 3.4 told you to log the "looks safe" / "on track" cases too — without them the model only ever learns to say "risky").

**Step 5.2 — Choose the model size and training approach**

- Start small: a model in the 1–3B parameter range is plenty for a narrow classification-style task like this (it doesn't need broad world knowledge, just pattern-matching against your specific error taxonomy).
- Two realistic options, easiest first:
  1. **Fine-tune** a small open model (e.g., via LoRA/QLoRA) on (plan → verdict + category) pairs. Cheapest to run, fastest to iterate.
  2. **Distillation**: use your larger LLM's Layer 2/3 outputs as "teacher" labels to train the small "student" model to imitate them. This is really the same data as option 1, described from the "why it works" angle — you're compressing what the big model learned into a small one.
- Recommend starting with option 1; it's simpler to explain and debug, and gets you 90% of the benefit.

**Step 5.3 — Validate before replacing anything**

Before swapping the small model into production:
- Hold out a slice of your logged data (don't train on it) and compare the small model's flag/no-flag decisions against what the big LLM would have said.
- Track false positives (flagging a fine plan — annoying, causes unnecessary rewrites) and false negatives (missing a real risk — the costly kind) separately. Decide an acceptable threshold for each before going live.

**Step 5.4 — Deploy as a drop-in replacement**

Once validated, point Layer 2's Reflector call and Layer 3's Drift Monitor call at the small model instead of the general-purpose LLM. Keep the general-purpose LLM as a fallback for low-confidence cases (the small model can output a confidence score; anything below a threshold gets escalated to the bigger model, same idea as a spam filter escalating uncertain emails to a human).

**Step 5.5 — Keep it fresh**

Re-run Step 5.1–5.3 periodically (e.g., monthly, or after every N thousand new logged trajectories) as your agent encounters new kinds of tasks and new failure patterns. This is a retraining loop, not a one-time project.

---

## 6. How the five layers connect, end to end

```
                     ┌─────────────────────────────────────────────┐
                     │   Layer 1: Logging & Root-Cause Tagging      │
                     │   (trajectory.json + metadata.json)          │
                     │   — build this first, feeds everything below │
                     └───────────────┬───────────────────────────────┘
                                     │  (offline: mine "planning" & "reflection"
                                     │   errors into an Error Pattern Library)
                                     ▼
┌───────────────────┐   check plan   ┌──────────────────────────┐
│  Agent proposes a  │──────────────▶│ Layer 2: Pre-Flight       │
│  plan              │◀── revise ────│ Reflector                 │
└───────────────────┘   if flagged   └──────────────────────────┘
          │  plan approved
          ▼
┌───────────────────┐   at each subgoal / interval   ┌──────────────────────────┐
│  Layer 4: Subgoal   │──────────────────────────────▶│ Layer 3: Drift Monitor   │
│  Execution          │◀── local replan if flagged ───│                          │
└───────────────────┘                                 └──────────────────────────┘
          │
          ▼
   Run completes, logged back into Layer 1's store
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Periodically retrain the small planner/checker      │
│  model on everything logged above; deploy it to replace the   │
│  general-purpose LLM calls inside Layers 2 & 3.                │
└─────────────────────────────────────────────────────────────┘
```

The key idea to repeat to the team: **Layers 2, 3, and 4 don't just consume Layer 1's data — they also produce more of it.** Every plan check, drift check, and subgoal outcome gets written back into the same store. That growing, unified dataset is what makes Layer 5 possible. If any layer logs to a separate, inconsistent format, Layer 5 becomes much harder — so the schema in Section 1.1 is worth getting agreement on before writing any new code.

---

## 7. Suggested repo structure

```
error-prop/
├── eval/
│   ├── output/                       # Layer 1 trajectory + metadata logs (write target)
│   └── tasks/                        # existing task definitions
├── taxonomy/                         # NEW — Layer 1
│   ├── error_taxonomy.py             # the 5 shared error-category labels, defined once
│   ├── tagger.py                     # LLM-as-judge that tags each step
│   ├── root_cause_finder.py          # picks the earliest tagged step as root cause
│   └── corrective_rerun.py           # re-runs from the root cause with targeted feedback
├── pattern_library/                  # NEW — Layer 2/3
│   ├── build_library.py              # mines Layer 1 logs for planning/reflection errors
│   ├── pattern_store/                # vector DB files (e.g., Chroma persisted dir)
│   └── query.py                      # similarity search helper used by reflector + drift monitor
├── reflector/                        # NEW — Layer 2
│   └── preflect_check.py             # pre-execution plan check
├── drift_monitor/                    # NEW — Layer 3
│   └── drift_check.py                # periodic / subgoal-boundary check (reuses pattern_library)
├── subgoals/                         # NEW — Layer 4
│   ├── subgoal_planner.py            # breaks task into subgoals up front
│   └── subgoal_tracker.py            # tracks completion, triggers local replans
├── distillation/                     # NEW — Layer 5
│   ├── build_dataset.py              # assembles training data from all logs above
│   ├── train_small_model.py
│   └── eval_small_model.py           # validation against held-out data before go-live
├── harness/                          # existing agent harness
└── run_experiment.py                 # existing entry point — will call into reflector/
                                       # subgoals/drift_monitor as the loop is extended
```

---

## 8. Suggested rollout order (so the team isn't blocked waiting on each other)

| Phase | What ships | Who's unblocked to start next |
|---|---|---|
| 0 | Layer 1, Steps 1.1–1.2: trajectory/metadata logging + a hand-labeled failure benchmark | Nothing downstream yet — this is the foundation everyone waits on |
| 1 | Layer 1, Steps 1.3–1.6: automated tagger, root-cause finder, corrective re-run loop, schema additions | Layers 2, 3, 4 can now be built in parallel against real tagged data |
| 2 | Layer 2 (Pre-Flight Reflector) — log-only mode first, no plan rewrites yet | Confirms the pattern library works before it starts changing agent behavior |
| 3 | Layer 4 (Subgoal Decomposition), prompting-level only | Gives Layer 3 a natural checkpoint to use |
| 4 | Layer 3 (Drift Monitor), using subgoal boundaries from Phase 3 | Full Layers 1–4 loop is now live |
| 5 | Turn on Layer 2's plan-rewrite behavior (not just logging) | Measure impact on success rate before/after |
| 6 | Layer 5: dataset assembly + first small-model training run | Ongoing retraining cadence begins |

Each phase produces something demoable on its own — useful for keeping non-technical stakeholders updated without waiting for the entire system to be finished.

---

## 9. Glossary (for non-technical teammates)

- **Trajectory** — the full step-by-step recording of one agent run, like a transcript.
- **Root cause** — the single earliest step that started the chain of failure, as opposed to every small mistake along the way.
- **Reflector** — a separate LLM call whose only job is to critique a plan before or during execution, not to do the task itself.
- **Vector database / pattern library** — a searchable memory that finds *similar meaning*, not just exact text matches.
- **Drift** — when a run starts going off track partway through, even if it started with a good plan.
- **Subgoal** — a small checkpoint inside a larger task (e.g., "open the map" is a subgoal of "find the nearest cafe").
- **Distillation** — training a small, cheap model to imitate the judgments of a larger, more expensive one, using logged examples.
- **False positive / false negative** — a false positive is flagging a plan that was actually fine (annoying); a false negative is missing a plan that was actually risky (costly). Both matter, but false negatives are usually worse.

---

## PART III — Packaging as a Standalone Library (multi-harness, multi-model reuse)

**Purpose:** once Layers 1–4 work inside your own harness (the fine-tuning of Layer 5 excluded, per your note — that stays as-is for now), this is how to repackage everything as a library other harnesses and models can plug into, without depending on this specific repo.

### The core decision: library first, service optional

**Build a library, not a webhook, as the foundation.** A hosted service is a thin optional wrapper *around* the library later — not a replacement for it.

**Why not a service by default:**
- The actual work (LLM-as-judge tagging, vector DB pattern lookups, flag/no-flag decisions) needs low latency and tight access to trajectory data as it's generated.
- If every step round-trips over HTTP to an external service, you add network latency to *every agent step*.
- You'd force every adopting team to also depend on your service's uptime — a much bigger ask than `pip install`.

**When a service becomes worth building (later, not now):**
1. Non-Python teams need to call in.
2. You want a **centrally-updated, shared pattern library** so every team benefits from every other team's logged failures, instead of each team maintaining an isolated copy.

Build the library first; only add the service once the library's interface is stable and there's real cross-team demand for a shared pattern store.

---

### What you're actually building — three separable pieces

#### 1. The hook interface (the contract) — most important design decision

A small set of functions any harness calls into at specific points in its own loop:

```python
class LongHorizonGuard:
    def on_plan_proposed(self, plan: Plan, context: RunContext) -> PlanVerdict:
        """Layer 2. Called before execution. Returns approve/revise + reasoning."""

    def on_step(self, step: Step, context: RunContext) -> StepVerdict:
        """Layer 1 (logging) + Layer 3 (drift, fired on interval/subgoal boundary)."""

    def on_subgoal_boundary(self, context: RunContext) -> SubgoalVerdict:
        """Layer 4. Called when a subgoal completes or is requested."""

    def on_run_end(self, trajectory: Trajectory, context: RunContext) -> RunTags:
        """Layer 1. Root-cause tagging after the run finishes."""
```

This is the entire product, conceptually. Everything else (pattern library, tagger, storage) sits **behind** this interface and is swappable. A harness developer only needs to know these four signatures — never how tagging or pattern-matching works internally.

> **Stability rule:** pin this file as the stability contract. Breaking changes here need a major version bump and a migration note, since other teams' code depends on it directly. Everything behind it (tagger prompts, pattern library backend, checker model) can change freely in minor versions.

#### 2. The implementations behind the interface

The Layers 1–4 code already built — pattern library (vector DB), tagger (LLM-as-judge), subgoal tracker, storage layer. The refactor: stop calling these directly from the harness; make them the internals of `LongHorizonGuard`.

#### 3. A storage/backend abstraction

Abstract trajectory/metadata storage behind an interface (`TrajectoryStore`), with a local-JSON implementation as the default and a pluggable Postgres/S3 implementation for teams that need it. This is what makes the future hosted-service version possible without rewriting core logic — the service becomes `LongHorizonGuard` running behind FastAPI with a shared `TrajectoryStore`.

---

### Repo structure

```
longhorizon-guard/
├── longhorizon_guard/                 # the pip-installable package
│   ├── __init__.py                    # exports LongHorizonGuard, hook types
│   ├── interface.py                   # the 4-method contract above (this is the "spec")
│   ├── taxonomy.py                    # shared error taxonomy enum/constants
│   ├── layers/
│   │   ├── logging.py                 # Layer 1: writes trajectory/metadata
│   │   ├── tagger.py                  # Layer 1: LLM-as-judge root-cause tagging
│   │   ├── preflight.py               # Layer 2: reflector
│   │   ├── drift.py                   # Layer 3: drift monitor
│   │   └── subgoals.py                # Layer 4: subgoal tracker
│   ├── pattern_library/
│   │   ├── store.py                   # vector DB wrapper (pluggable: Chroma/FAISS/Pinecone)
│   │   └── builder.py                 # offline mining of past errors into patterns
│   ├── storage/
│   │   ├── base.py                    # TrajectoryStore abstract interface
│   │   ├── local_json.py              # default implementation
│   │   └── postgres.py                # optional, for teams that need it
│   ├── config.py                      # thresholds, model choice, check intervals — all configurable
│   └── adapters/                      # OPTIONAL, built later, one per popular harness
│       ├── langgraph.py
│       └── autogen.py
├── service/                           # OPTIONAL — the webhook/hosted wrapper
│   ├── main.py                        # FastAPI exposing the same 4 operations over REST
│   ├── Dockerfile
│   └── auth.py                        # API keys, per-team pattern libraries
├── examples/
│   └── minimal_harness_integration.py # copy-pasteable "here's how you wire this in"
├── tests/
├── pyproject.toml
└── README.md
```

**Note on `adapters/`:** rather than making every harness team hand-write glue code, write thin adapters for popular frameworks (LangGraph, AutoGen, CrewAI, whatever's in use internally) that translate *their* loop hooks into calls to the 4-method interface. This is the difference between "a library people have to study" and "a library people add in one line."

---

### Usage — library mode (default)

```python
from longhorizon_guard import LongHorizonGuard

guard = LongHorizonGuard(
    pattern_library_path="s3://our-bucket/patterns",
    checker_model="gpt-4o-mini",   # swappable — this is where the distilled small model plugs in later
    drift_check_every_n_steps=4,
)

# inside their existing harness loop:
plan = agent.propose_plan(task)
verdict = guard.on_plan_proposed(plan, context)
if verdict.flagged:
    plan = agent.revise_plan(plan, verdict.reasoning)

for step in agent.execute(plan):
    guard.on_step(step, context)
    if guard.should_check_drift(context):
        drift = guard.on_step(step, context)
        if drift.flagged:
            plan = agent.replan_subgoal(drift.reasoning)

tags = guard.on_run_end(trajectory, context)
```

They `pip install longhorizon-guard`, instantiate one object, call four methods at the right points in their existing loop. No service dependency, no network calls unless `checker_model` points at a remote API (which it typically will anyway, since it's an LLM call).

---

### Usage — service/webhook mode (optional, later)

Same interface, exposed over HTTP:

```
POST /v1/plan-check      { plan, context } -> { flagged, reasoning, revised_plan? }
POST /v1/drift-check     { trajectory_so_far, context } -> { flagged, reasoning }
POST /v1/subgoal-check   { context } -> { current_subgoal, completed, replan_needed }
POST /v1/run-complete    { trajectory } -> { root_cause_step, error_type, tags }
```

This is FastAPI wrapping the same `LongHorizonGuard` object, with `TrajectoryStore` pointed at shared infrastructure instead of local files. Stand this up once other teams want a shared, centrally-updated pattern library — not just because some teams use JS instead of Python. For non-Python teams, a REST call in the loop is fine, since LLM-call latency at those points already dominates any network hop.

---

### Deployment

- **Library** — publish to internal PyPI (or public PyPI for external adoption), with semantic versioning. `interface.py` is the stability contract — breaking changes there = major version bump + migration note.
- **Service** (if/when built) — standard Docker image, deployed like any internal microservice, behind the standard auth/gateway. Vector DB and trajectory store are their own managed dependencies, not embedded in the container.
- **Pattern library distribution** — two options:
  - **Centralized** (service mode): one shared store, more powerful, but couples every team's uptime to your infra.
  - **Snapshot-based** (library mode): each team pulls the latest patterns periodically, more resilient, but patterns go stale between pulls.
  - **Recommendation:** start with snapshot distribution; only move to centralized once there's real cross-team demand.

---

### The one thing to get right before writing any code

Spend real design time on `interface.py` specifically, before touching implementation. Once other teams start calling those four methods, changing their signatures becomes a cross-team coordination problem. Everything else behind the interface is much cheaper to change later.

---

## PART IV — Operational Build Playbook (Data Cleanup → Tagger → Scaffold)

**Purpose:** the exact prompt to hand a coding agent right now, given where the project actually stands (Layer 1 logging built, tagging not built, existing data contaminated with rate-limit noise). This is phase-gated — the agent must stop and confirm with you at each checkpoint rather than running straight through.

**How to use:** paste the fenced prompt block below into your coding agent (Claude Code, Antigravity, etc.), inside this repo. Point it at your extra folder of past runs when it reaches Phase 0.

```
You are helping me build out the next phase of a long-horizon agent error-mitigation
project. Work through the phases below IN ORDER. Do not skip ahead. After finishing
each phase, STOP, show me a summary of what you did/found, and wait for me to confirm
before starting the next phase. Where a phase says "STOP AND ASK ME", you must pause
and wait for my answer before proceeding — do not guess or assume a default.

I will also be pasting in an additional folder of past run logs partway through
Phase 0 — treat that as another data source alongside whatever already exists in
this repo (eval/output/, output/, findings/, and any other run-log directories
you find).

═══════════════════════════════════════════════════════════════════
PHASE 0 — CONSOLIDATE AND CLEAN THE EXISTING RUN DATA
═══════════════════════════════════════════════════════════════════

Goal: build one clean, deduplicated set of GENUINE agent-executed runs (successes
AND real failures), with all infrastructure noise (rate limits, 429s, timeouts,
connection errors, empty trajectories) removed.

Steps:

1. Scan every directory in this repo that contains run logs (metadata.json /
   trajectory.json pairs, or equivalent) — including eval/output/, output/,
   findings/, and [I will paste an additional folder of past runs — incorporate
   it into this scan too].

2. For every run found, classify it into exactly one of these buckets:
   a) GENUINE SUCCESS — agent executed, produced an answer, grader marked pass.
   b) GENUINE FAILURE — agent executed (steps > 0, no infra error in the trajectory),
      produced an answer or attempted to, but the grader marked it fail OR it
      errored out mid-reasoning for a real reason (e.g. malformed tool call it
      generated itself, bad logic, wrong final answer).
   c) INFRA NOISE — any run where the failure is clearly NOT the agent's fault:
      HTTP 429 rate limit, HTTP 5xx, connection/timeout errors on the FIRST step
      before any real reasoning happened, empty trajectory ({"steps": []}),
      or total_steps_taken == 1 where that single step is an API/provider error
      rather than an agent action.

3. Deduplicate across directories. Some of these folders are known duplicates of
   each other (e.g. output/ mirrors eval/output/) — use run_id/task_id/trial_number
   to detect and collapse duplicates, keeping only one copy of each genuine run.

4. Produce a report with:
   - Total runs scanned, broken down by source directory.
   - Count in each bucket (genuine success / genuine failure / infra noise).
   - The final count of GENUINE runs (success + failure combined) after dedup.
   - A breakdown by horizon level (L1/L2/L3/L4) of genuine runs — I need to know
     if any horizon level ends up with very few or zero genuine runs, since that's
     a coverage gap I need to know about before moving on.

5. Write the cleaned, deduplicated set of genuine runs into a new consolidated
   directory (e.g. eval/clean_runs/), preserving the original metadata.json/
   trajectory.json structure so nothing downstream needs to change format.
   Leave the original source directories untouched (do not delete/modify them —
   this is a read-and-copy operation, not a move).

🛑 STOP AND ASK ME: if the final genuine-run count is below 50, tell me clearly
and ask whether I want to (a) proceed anyway with fewer runs, (b) go collect more
data first, or (c) proceed with what we have but flag it as a limited-calibration
set. Do not just proceed silently with a small number.

✅ MANUAL TEST (I will do this before approving Phase 1):
- Open 3 random runs from eval/clean_runs/ marked GENUINE FAILURE and confirm by
  reading the trajectory that they really are genuine agent mistakes, not
  misclassified infra errors.
- Open 3 random runs marked INFRA NOISE (from the report, not the clean set) and
  confirm they really are rate-limit/infra issues, not real agent failures being
  wrongly discarded.
- Confirm the total genuine count roughly matches what the report claims.

═══════════════════════════════════════════════════════════════════
PHASE 1 — SCHEMA ADDITIONS
═══════════════════════════════════════════════════════════════════

Goal: add the fields needed to store tagging output, without touching any
existing behavior.

Steps:

1. Add these fields to the RunMetadata schema (default null/empty, so existing
   code that constructs RunMetadata without them still works):
   - root_cause_step_index (int, nullable)
   - root_cause_error_type (string, nullable)
   - tag_confidence (float, nullable)  — how sure the tagger was, 0-1
   - tag_source (string, nullable) — "human" or "llm_judge", so we always know
     which tags came from calibration vs automated tagging later

2. Add this field to the StepRecord (or per-step trajectory entry) schema:
   - error_tag (string, nullable)

3. Confirm these additions don't break any existing code that reads/writes
   metadata.json or trajectory.json — run the existing test suite / a sample
   experiment run if one exists, and confirm output still parses correctly.

🛑 STOP AND ASK ME about the taxonomy to use before writing any tagging logic.
There are two label sets already floating around in this repo:
   - The 5-category AgentDebug taxonomy from the methodology doc: memory,
     reflection, planning, action, system.
   - The 6-category scheme already sitting unused in tagger.py: planning_error,
     memory_error, tool_use_error, external_error, grader_error, other.
Show me both side by side and ask me which one to standardize on (or how to
map one onto the other) before Phase 2 starts tagging anything. Also flag
explicitly: the audit found at least one case where the agent reasoned
CORRECTLY but failed the grader on output formatting (verbose paragraph
instead of a bare answer) — ask me whether that deserves its own category
(e.g. "grader_error"/"format_error") separate from genuine reasoning mistakes,
since lumping it in with real errors would pollute the pattern library later
with "errors" that aren't actually planning mistakes.

✅ MANUAL TEST (I will do this before approving Phase 2):
- Open the updated schema file and confirm the new fields are present with
  sensible defaults.
- Run one existing experiment end-to-end and confirm metadata.json /
  trajectory.json are still written correctly with the new fields present
  but null.

═══════════════════════════════════════════════════════════════════
PHASE 2 — HUMAN CALIBRATION LABELS (ground truth for the tagger)
═══════════════════════════════════════════════════════════════════

Goal: get real human-assigned labels on a subset of eval/clean_runs/ so we have
ground truth to check the automated tagger against.

Steps:

1. Take the existing manual tagger.py CLI (currently unused) and update it to:
   - Read from eval/clean_runs/ instead of wherever it currently points.
   - Use the taxonomy we agreed on in Phase 1.
   - Write output into the root_cause_error_type / root_cause_step_index /
     tag_source="human" fields directly in each run's metadata.json (in
     eval/clean_runs/), instead of a separate tags.csv — so everything lives
     in one place going forward.
   - Only present GENUINE FAILURE runs for labeling (skip successes — nothing
     to tag there).

2. Tell me how many genuine failure runs are available to label, and roughly
   how long this will take me (a rough estimate based on run count).

🛑 STOP AND ASK ME: I will personally run this CLI and label the runs myself —
this step requires my judgment, not yours. Do not attempt to auto-generate
these labels. Confirm the CLI works on 1-2 runs with me before I sit down and
label the full set.

✅ MANUAL TEST (I will do this — this IS the phase, not a check afterward):
- I run the CLI against every genuine-failure run in eval/clean_runs/ and
  hand-assign a category + root cause step to each one.
- Once done, I'll ask you to report back: how many were labeled, and the
  distribution across categories (so I can sanity check nothing is wildly
  imbalanced before we calibrate against it).

═══════════════════════════════════════════════════════════════════
PHASE 3 — AUTOMATED LLM-AS-JUDGE TAGGER
═══════════════════════════════════════════════════════════════════

Goal: build the automated tagger, and prove it agrees with the human labels
from Phase 2 before trusting it on anything else.

Steps:

1. Build a new module (e.g. taxonomy/llm_tagger.py) that:
   - Takes a full trajectory as input.
   - Prompts an LLM to go step-by-step, assign one of our agreed taxonomy
     labels to each step (or null if the step looks fine), with a short
     justification per tag.
   - From the tagged steps, picks the EARLIEST tagged step as the root cause
     (not the last, not the most severe) and writes root_cause_step_index +
     root_cause_error_type + tag_source="llm_judge" back to metadata.json,
     and error_tag into each relevant step of trajectory.json.
   - Only run this on GENUINE FAILURE runs (skip successes, same as Phase 2).

2. 🛑 STOP AND ASK ME about which model/provider to use for the judge calls
   BEFORE running this on anything. This is a real bottleneck: running an LLM
   judge over 50-100 trajectories is itself a batch of LLM calls, and I am
   still on free-tier rate limits. Give me options (e.g. a smaller/cheaper
   model, a different free-tier provider, batching with delays, or running it
   in smaller chunks over multiple days) and let me choose before you start
   burning quota on this.

3. Once I've chosen a provider/model and you've run the tagger on the
   Phase-2-labeled subset, produce an agreement report:
   - % of runs where the LLM judge's root_cause_error_type matches my human
     label exactly.
   - % where it matches the root_cause_step_index (or is within 1 step of it).
   - A breakdown of disagreements — show me the specific runs where it
     disagreed with me, with both labels and the trajectory, so I can judge
     whether the LLM judge or I was more likely correct.

🛑 STOP AND ASK ME: if agreement is below roughly 70-75%, do not proceed to
Phase 4. Tell me clearly, show me the disagreement examples, and ask whether
I want to (a) revise the tagging prompt and re-test, (b) add more human labels
to test against, or (c) proceed anyway with the caveat that the tagger is
noisy. Do not silently accept a low-agreement tagger and move on.

✅ MANUAL TEST (I will do this before approving Phase 4):
- Review the disagreement examples you show me and independently judge a
  handful of them myself.
- Spot-check 3-5 runs where the tagger agreed with me, to make sure that
  agreement isn't a coincidence (e.g. it isn't just defaulting to the most
  common category every time).

═══════════════════════════════════════════════════════════════════
PHASE 4 — RUN THE CALIBRATED TAGGER ON THE FULL CLEAN SET
═══════════════════════════════════════════════════════════════════

Goal: tag every genuine failure in eval/clean_runs/ (not just the Phase-2
labeled subset), now that the tagger is calibrated.

Steps:

1. Run the tagger from Phase 3 across all remaining genuine-failure runs in
   eval/clean_runs/ that weren't part of the Phase 2 human-labeled subset.

2. Produce a final summary:
   - Total tagged runs.
   - Distribution across error categories.
   - Distribution across horizon levels — flag if any horizon level still has
     very few tagged "planning"/"reflection" (or whatever we're calling that
     category) entries, since that's specifically what Layer 2 (PreFlect) will
     need next.

🛑 STOP AND ASK ME: if the count of planning/reflection-category tags (the ones
PreFlect's pattern library will be built from) is very small (e.g. under 15-20),
tell me — I may need to collect more data before PreFlect will have anything
meaningful to work with.

✅ MANUAL TEST (I will do this before approving Phase 5):
- Spot check 5 newly-tagged runs across different horizon levels for
  plausibility.
- Confirm the category distribution roughly makes sense given what I know
  about how these tasks tend to fail.

═══════════════════════════════════════════════════════════════════
PHASE 5 — LIBRARY SCAFFOLD (can be built in parallel with Phases 0-4 if you
have capacity, but confirm with me before doing so — otherwise do it here)
═══════════════════════════════════════════════════════════════════

Goal: set up the reusable library structure, without wiring in real logic yet
beyond what already exists.

Steps:

1. Create this directory structure (see repo notes I have for the full
   picture — build the skeleton now, fill in Phase 6+ later):

   longhorizon_guard/
   ├── __init__.py
   ├── interface.py       # the 4-method hook contract (stub bodies for now)
   ├── taxonomy.py         # the taxonomy we agreed on in Phase 1, as constants/enum
   ├── layers/
   │   ├── logging.py      # move existing trajectory-writing logic here
   │   └── tagger.py       # move the Phase 3 automated tagger here
   ├── storage/
   │   ├── base.py         # TrajectoryStore abstract interface
   │   └── local_json.py   # wraps existing read/write logic behind that interface
   └── config.py           # model choice, thresholds — pull hardcoded values here

2. Write interface.py with these four method signatures (bodies can raise
   NotImplementedError for now except on_run_end, which should call the
   Phase 3/4 tagger):

   class LongHorizonGuard:
       def on_plan_proposed(self, plan, context): ...   # not implemented yet
       def on_step(self, step, context): ...              # not implemented yet
       def on_subgoal_boundary(self, context): ...         # not implemented yet
       def on_run_end(self, trajectory, context): ...      # WIRE THIS to the tagger

3. Refactor the existing experiment runner to call storage/local_json.py and
   layers/logging.py instead of writing files directly, so the new structure
   is actually in use, not just sitting alongside the old code unused.

🛑 STOP AND ASK ME: show me the new file layout and the diff to the runner
before merging anything — this touches the code path everything else depends
on, so I want to review it, not just run it.

✅ MANUAL TEST (I will do this before calling this done):
- Run a full experiment end-to-end through the refactored runner and confirm
  output is byte-for-byte equivalent (or equivalent in the fields that matter)
  to what it produced before the refactor.
- Call guard.on_run_end() manually on a known genuine-failure trajectory and
  confirm it produces the same tag the Phase 3/4 tagger would have.

═══════════════════════════════════════════════════════════════════
AFTER PHASE 5 — DO NOT PROCEED FURTHER WITHOUT ME
═══════════════════════════════════════════════════════════════════

Stop here. The next phase (building the PreFlect pattern library behind
on_plan_proposed) depends on how many planning/reflection-tagged runs came
out of Phase 4 — I'll decide with you whether we have enough to proceed once
we see that number.

═══════════════════════════════════════════════════════════════════
GENERAL RULES FOR THE WHOLE SESSION
═══════════════════════════════════════════════════════════════════

- Never delete or overwrite original run-log directories — only read from them
  and write into new locations (eval/clean_runs/, the new longhorizon_guard/
  package).
- Never make a batch of real LLM API calls without telling me the estimated
  count first and letting me confirm — I'm on free-tier limits and need to
  manage quota deliberately across phases.
- If you hit an ambiguity not covered by a "STOP AND ASK ME" above, stop and
  ask rather than guessing — I'd rather answer a question than redo a phase.
- After each phase, give me a short written summary (a few sentences, not a
  wall of text) of what changed and what I need to go test manually before
  you continue.
```
### Notes on using this playbook

- **Phase 0 is where you paste in your extra folder of past runs.** Point the agent at wherever you drop it and tell it to include that directory in the scan.
- **Phase 2 is the one phase that's fundamentally yours, not the agent's.** No amount of prompting can substitute your judgment for ground-truth labels — the agent's job there is just to hand you a working CLI, not to label anything itself.
- **The biggest real bottleneck flagged above is Phase 3's judge-model quota.** Tagging 50-100 trajectories is itself a batch of LLM calls. Worth deciding in advance whether you want to use a different (cheaper/more generous) provider just for judge calls versus whatever you use for the agent itself — they don't have to be the same model.
- **Don't let it merge Phase 5 silently.** That phase touches your run-writing code path; review the diff before accepting it, since a bug there would corrupt data going forward, not just this session.

---

## PART V — Open Decisions & Known Bottlenecks (read before building anything further)

These are the unresolved questions flagged so far. Resolve them explicitly rather than letting them get decided implicitly by whatever the coding agent happens to do first.

1. **Taxonomy standardization.** Two label sets currently exist: the 5-category AgentDebug set (`memory`, `reflection`, `planning`, `action`, `system`) used throughout Part II, and the 6-category set already sitting in `tagger.py` (`planning_error`, `memory_error`, `tool_use_error`, `external_error`, `grader_error`, `other`). Pick one, or define an explicit mapping between them, before Layer 1's automated tagger is built — every layer downstream (pattern library, drift monitor, distillation dataset) inherits whichever categories you lock in here.

2. **Does "correct reasoning, wrong output format" deserve its own category?** The audit surfaced a real case: the agent computed the right answer, then failed the grader by returning a verbose paragraph instead of a bare value. That's not a `planning`/`reflection`/`action` mistake in the usual sense — it's a formatting/instruction-following miss. Lumping it into an existing category would quietly pollute the pattern library with "errors" that aren't really planning failures. Decide whether this needs a `grader_error`/`format_error` category of its own.

3. **Judge-model quota management.** The automated tagger (Layer 1, Step 1.3) itself makes an LLM call per trajectory being tagged — tagging 50–100 runs is itself a batch of API calls, on top of whatever the agent-under-test uses. Decide whether the judge model should be a different (cheaper/more available) provider than the agent being evaluated, before running Phase 3 of Part IV.

4. **Data coverage gaps by horizon level.** As of the last audit, there are effectively zero confirmed genuine reasoning failures at any horizon level (L1–L4), because the small number of "errors" that existed were all rate-limit noise. The Part IV playbook's Phase 0/4 checkpoints are specifically designed to surface this — do not proceed to building the Layer 2 pattern library until Phase 4 of Part IV confirms a meaningful number of genuine `planning`/`reflection`-tagged runs exist (rough target: 15–20+, ideally spread across horizon levels, not clustered in one).

5. **Library vs. service timing.** Part III recommends building the library first and only adding the hosted-service wrapper once there's real cross-team demand for a shared pattern library, or a non-Python team needs to call in. Don't build the service speculatively — it adds an uptime dependency for every consumer that a plain library doesn't have.

6. **Layer 5 stays out of scope for now.** Per current instructions, the fine-tuning/distillation implementation itself (Layer 5, Section 5.2 in Part II) is intentionally not being built yet — everything up through Layer 4 plus the library packaging (Part III) comes first. Layer 5's dataset-assembly logic (Step 5.1) is worth keeping schema-compatible with as you build Layers 1–4, even though training won't start until later.
