# Build Prompt: Data Cleanup → Automated Tagger → Library Scaffold

**How to use this file:** paste this whole thing to your coding agent (Claude Code, Antigravity, etc.) in your repo. It's written as a single prompt with explicit phases. The agent should stop and ask you at every point marked `🛑 STOP AND ASK ME`, and must not skip ahead to a later phase until the current phase's manual test (marked `✅ MANUAL TEST`) has passed. Do not let it batch multiple phases into one shot — confirm each phase before moving to the next.

---

## PROMPT STARTS HERE — paste everything below into your coding agent

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

## PROMPT ENDS HERE

---

## Notes for you (not part of the prompt — for your own tracking)

- **Phase 0 is where you paste in your extra folder of past runs.** Point the agent at wherever you drop it and tell it to include that directory in the scan.
- **Phase 2 is the one phase that's fundamentally yours, not the agent's.** No amount of prompting can substitute your judgment for ground-truth labels — the agent's job there is just to hand you a working CLI, not to label anything itself.
- **The biggest real bottleneck flagged above is Phase 3's judge-model quota.** Tagging 50-100 trajectories is itself a batch of LLM calls. Worth deciding in advance whether you want to use a different (cheaper/more generous) provider just for judge calls versus whatever you use for the agent itself — they don't have to be the same model.
- **Don't let it merge Phase 5 silently.** That phase touches your run-writing code path; review the diff before accepting it, since a bug there would corrupt data going forward, not just this session.
