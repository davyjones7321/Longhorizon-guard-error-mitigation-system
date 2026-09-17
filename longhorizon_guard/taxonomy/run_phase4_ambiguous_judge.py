import sys
import os
import re
import json
import asyncio
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq

from longhorizon_guard.taxonomy.judge import JUDGE_SYSTEM_PROMPT, _load_dotenv_if_needed
from longhorizon_guard.taxonomy.extract_phase4_structural_gt import extract_dataset

_load_dotenv_if_needed()

OUTPUT_PATH = Path("new_dataset/ambiguous_judgments.jsonl")
MODEL_NAME = "gemini-3.1-flash-lite"
MAX_REQUESTS_TODAY = 500
RATE_LIMIT_DELAY = 4.2  # 4.2 seconds = ~14.3 RPM (strictly under 15 RPM limit)

SINGLE_TRAJECTORY_JUDGE_PROMPT = """You are an expert AI agent failure auditor specializing in root-cause error analysis for multi-step agent trajectories.

Your task is to analyze the tail of an agent's trajectory on a coding/system task where structural signals were ambiguous, and decide:
1. Did the agent ultimately SUCCEED or FAIL at the requested task?
2. If it failed, what was the primary root-cause category according to the 7-category taxonomy below?
3. Provide a concise one-line reason and a brief chain-of-thought justifying your decision.

### Standardized Error Taxonomy (7 Categories)
1. planning_error: Inappropriate overall strategy, wrong direction, or omission of critical requirements.
2. reflection_error: Misjudged environment state, overlooked tool output errors, or claimed victory when goal was not met.
3. tool_use_error: Invalid tool, wrong arguments, syntax/format errors in command.
4. memory_error: Repetitive loops, forgetting previous discoveries, loss of long-horizon context.
5. external_error: Environment timeouts, infrastructure drops, system limits outside agent control.
6. grader_error: Task was actually completed correctly, but evaluation tests/harness were flawed.
7. other: Edge case or unclassified failure.

### Output JSON Format:
Respond with ONLY valid JSON:
{
  "verdict": "success" | "failure",
  "root_cause_error_type": "planning_error" | "reflection_error" | "tool_use_error" | "memory_error" | "external_error" | "grader_error" | "other" | null,
  "confidence": 0.0 to 1.0,
  "one_line_reason": "Single sentence summarizing why this trajectory succeeded or failed.",
  "chain_of_thought": "Brief step-by-step reasoning."
}
"""

async def call_gemini_judge(prompt: str, api_key: str, model_name: str = MODEL_NAME) -> Optional[Dict[str, Any]]:
    """Calls Gemini API directly with retry logic and JSON response formatting."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    req_data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=30)
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_content = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
                # Extract first valid JSON object
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f" [HTTP 429 Rate Limit - backing off {8 * (attempt + 1)}s] ", end="", flush=True)
                await asyncio.sleep(8.0 * (attempt + 1))
            else:
                print(f" [HTTP {e.code}: {e.reason}] ", end="", flush=True)
                await asyncio.sleep(3.0 * (attempt + 1))
        except Exception as exc:
            print(f" [Err: {exc}] ", end="", flush=True)
            await asyncio.sleep(3.0 * (attempt + 1))

    return None


def extract_trajectory_tail(turns: List[Dict[str, Any]], max_turns: int = 10) -> List[Dict[str, Any]]:
    """Extract the last `max_turns` from the conversation, keeping content readable and bounded."""
    tail = turns[-max_turns:] if len(turns) > max_turns else turns
    cleaned_tail = []
    for t in tail:
        role = t.get("from") or t.get("role")
        val = str(t.get("value") or t.get("content") or "").strip()
        if len(val) > 2500:
            val = val[:1250] + "\n... [TRUNCATED VERBOSE OUTPUT] ...\n" + val[-1250:]
        cleaned_tail.append({"role": role, "content": val})
    return cleaned_tail


def get_all_ambiguous_trajectories() -> List[Dict[str, Any]]:
    """Collect all 554 ambiguous trajectories across the 4 benchmarks."""
    datasets = [
        ("terminal-bench-2", "new_dataset/agent_launch_pad/terminal-bench-2.parquet", True),
        ("scienceagentbench", "new_dataset/agent_launch_pad/scienceagentbench.parquet", True),
        ("swe_bench_test", "new_dataset/jetbrains_swe/swe_bench_test_trajectories.parquet", False),
        ("swe_smith", "new_dataset/jetbrains_swe/swe_smith_trajectories.parquet", False),
    ]

    records = []
    for name, path, is_alp in datasets:
        recs = extract_dataset(path, name, is_alp=is_alp, use_calibrated_failure=True)
        t = pq.read_table(path)
        
        for idx, r in enumerate(recs):
            if r["structural_verdict"] == "ambiguous":
                if is_alp:
                    conv = t.column("conversations")[idx].as_py()
                    instruction = t.column("instruction")[idx].as_py() if "instruction" in t.column_names else ""
                else:
                    conv = t.column("messages")[idx].as_py()
                    instruction = conv[1].get("content", "") if len(conv) > 1 else ""

                records.append({
                    "dataset": name,
                    "task_id": r["task_id"],
                    "benchmark_label": r["benchmark_label"],
                    "structural_reason": r["structural_reason"],
                    "instruction": str(instruction)[:600],
                    "tail_turns": extract_trajectory_tail(conv, max_turns=10),
                    "total_turns": len(conv),
                })

    return records


async def run_ambiguous_evaluation():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment or .env file!")
        sys.exit(1)

    print("================================================================================")
    print("PHASE 4 PART 2: AMBIGUOUS TRAJECTORY REFERENCE-MODEL EVALUATION (GEMINI)")
    print(f"Model: {MODEL_NAME} | Rate Limit Pacing: {RATE_LIMIT_DELAY}s (~14.3 RPM)")
    print("================================================================================")

    all_ambiguous = get_all_ambiguous_trajectories()
    total_pool = len(all_ambiguous)
    print(f"Total ambiguous trajectories identified across all 4 datasets: {total_pool}")
    ds_counts = Counter(r["dataset"] for r in all_ambiguous)
    for ds, cnt in ds_counts.items():
        print(f"  - {ds:<20}: {cnt}")

    # Idempotent resume check
    existing_judgments = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        key = (record.get("dataset"), record.get("task_id"))
                        existing_judgments[key] = record
                    except Exception:
                        pass
        print(f"\nLoaded {len(existing_judgments)} existing judgments from {OUTPUT_PATH}")

    # Determine pending items
    pending = [r for r in all_ambiguous if (r["dataset"], r["task_id"]) not in existing_judgments]
    print(f"Pending to evaluate: {len(pending)}/{total_pool}")

    if not pending:
        print("All trajectories have already been evaluated!")
        return

    # Enforce today's cap of 500 requests
    limit_for_run = min(len(pending), MAX_REQUESTS_TODAY)
    to_run = pending[:limit_for_run]
    print(f"Executing batch of {len(to_run)} today (stopping cleanly at quota cap, remaining will resume tomorrow).\n")

    completed_this_run = 0
    start_time = time.time()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(to_run, 1):
            t0 = time.time()
            task_id = item["task_id"]
            dataset = item["dataset"]

            payload = {
                "dataset": dataset,
                "task_id": task_id,
                "instruction": item["instruction"],
                "total_turns": item["total_turns"],
                "tail_turns": item["tail_turns"],
            }

            prompt = (
                f"{SINGLE_TRAJECTORY_JUDGE_PROMPT}\n\n"
                f"Analyze this trajectory tail:\n"
                f"{json.dumps(payload, indent=2)}\n\n"
                f"Output JSON:"
            )

            print(f"[{idx}/{len(to_run)}] ({dataset}) {task_id[:25]:<25} ... ", end="", flush=True)

            res = await call_gemini_judge(prompt, api_key=api_key, model_name=MODEL_NAME)

            if res and isinstance(res, dict):
                record = {
                    "task_id": task_id,
                    "dataset": dataset,
                    "benchmark_label": item["benchmark_label"],
                    "structural_reason": item["structural_reason"],
                    "verdict": res.get("verdict"),
                    "root_cause_error_type": res.get("root_cause_error_type"),
                    "confidence": res.get("confidence"),
                    "one_line_reason": res.get("one_line_reason"),
                    "chain_of_thought": res.get("chain_of_thought"),
                    "total_turns": item["total_turns"],
                    "tail_turns": item["tail_turns"],
                    "model_used": MODEL_NAME,
                    "verification_status": "Judge single-trajectory output, reference-model evaluated",
                    "timestamp": time.time(),
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                completed_this_run += 1
                v = record.get("verdict", "?")
                cat = record.get("root_cause_error_type") or "none"
                print(f"DONE -> verdict={v} ({cat})")
            else:
                print("FAILED / NULL RESPONSE")

            # Pacing to stay strictly under 15 RPM
            elapsed = time.time() - t0
            sleep_needed = max(0.0, RATE_LIMIT_DELAY - elapsed)
            if sleep_needed > 0:
                await asyncio.sleep(sleep_needed)

    total_time = time.time() - start_time
    print(f"\n================================================================================")
    print(f"COMPLETED TODAY'S BATCH: {completed_this_run} evaluations in {total_time/60:.1f} minutes.")
    print(f"Total judgments now stored in {OUTPUT_PATH}: {len(existing_judgments) + completed_this_run}/{total_pool}")
    if len(existing_judgments) + completed_this_run < total_pool:
        remaining = total_pool - (len(existing_judgments) + completed_this_run)
        print(f"Remaining for tomorrow: {remaining} trajectories (run same script to resume seamlessly).")
    print(f"================================================================================")

if __name__ == "__main__":
    asyncio.run(run_ambiguous_evaluation())
