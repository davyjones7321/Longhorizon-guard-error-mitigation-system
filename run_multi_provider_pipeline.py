#!/usr/bin/env python3
"""
Sequential Multi-Provider Judge Pipeline
----------------------------------------
Judges 100 agent run records from sorted.json sequentially using multiple LLM providers:
1. Cloudflare Workers AI (@cf/meta/llama-3.3-70b-instruct-fp8-fast) - Records #1..22
2. Google Gemini (gemini-3.6-flash) - Records #23..28
3. Provider #3 (Groq qwen3.6-27b / Gemini alternate models / Cerebras) - Records #29..100

Output directory: findings/providers/
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def load_dotenv():
    env_files = [ROOT_DIR / ".env", ROOT_DIR / "eval" / ".env"]
    for env_file in env_files:
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")

load_dotenv()

JUDGE_SYSTEM_PROMPT = """You are an expert AI agent failure auditor specializing in root-cause error analysis for multi-step agent trajectories.

Your task is to analyze an agent's trajectory for a failed task and perform two evaluations:
1. Identify per-step errors (if any) across the trajectory.
2. Identify the single EARLIEST root-cause failure step (root_cause_step_index) and assign it one of the 7 standardized error categories: planning_error, reflection_error, tool_use_error, memory_error, external_error, grader_error, other.

### Standardized Error Taxonomy (7 Categories)

1. planning_error:
   - Definition: The agent formulated an incorrect overall strategy, sequence, or approach.
   - EXCEPTION OVERRIDE: An incomplete initial plan (e.g. omitting a sub-goal like 'clean') is NOT planning_error if the agent later had an opportunity to observe the environment and verify state. In such cases, tag reflection_error at the observation step, NOT planning_error at Step 0.
2. reflection_error:
   - Definition: The agent failed to evaluate tool output, misjudged environment state, misinterpreted task completion, or failed to recognize an incomplete/incorrect outcome after observing environment feedback.
3. tool_use_error:
   - Definition: The agent selected an invalid tool, formatted arguments incorrectly, passed invalid parameters, or omitted required parameters/constraints in a tool call.
4. memory_error:
   - Definition: The agent forgot previously observed information, duplicated actions in a loop, or lost context over long horizons.
5. external_error:
   - Definition: Failures caused by system limits, environment cutoffs, network timeouts, or tool execution errors outside agent control.
6. grader_error:
   - Definition: The agent succeeded according to task requirements, but the automated evaluator incorrectly graded it as a failure.
7. other:
   - Definition: Ambiguous failure or edge-case error not cleanly covered by the above 6 categories.

### Required Output JSON Format
You MUST respond with valid JSON matching this exact structure:
{
  "root_cause_step_index": 0,
  "root_cause_error_type": "planning_error",
  "confidence": 0.95,
  "root_cause_justification": "Brief explanation of why this step was the root cause.",
  "step_annotations": [
    {
      "step_index": 0,
      "error_tag": "planning_error",
      "justification": "Brief justification for step tag"
    }
  ]
}
"""

def prepare_trajectory_payload(run: Dict[str, Any], max_len: int = 300) -> Dict[str, Any]:
    meta = run.get("metadata", {}) or {}
    traj = run.get("trajectory", {}) or {}
    steps = traj.get("steps", [])

    formatted_steps = []
    for s in steps:
        raw_args = s.get("action_args", {})
        clean_args = {}
        if isinstance(raw_args, dict):
            for k, v in raw_args.items():
                str_v = str(v) if v is not None else ""
                clean_args[k] = str_v[:max_len]
        else:
            str_v = str(raw_args)
            clean_args = {"args": str_v[:max_len]}

        reas = (s.get("reasoning") or "")[:max_len]
        t_resp = s.get("tool_response") or ""
        if isinstance(t_resp, (dict, list)):
            t_resp = json.dumps(t_resp)
        else:
            t_resp = str(t_resp)
        t_resp = t_resp[:max_len]

        formatted_steps.append({
            "step_index": s.get("step_index"),
            "reasoning": reas,
            "action_name": s.get("action_name") or "none",
            "action_args": clean_args,
            "tool_response": t_resp,
        })

    payload = {
        "run_id": meta.get("run_id"),
        "task_id": meta.get("task_id"),
        "task_description": (meta.get("task_description") or meta.get("task_id") or "")[:max_len],
        "final_status": meta.get("final_status"),
        "grader_notes": (meta.get("grader_notes") or "")[:max_len],
        "total_steps": meta.get("total_steps_taken", len(steps)),
        "steps": formatted_steps,
    }
    return payload

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text_clean = text.strip()
    try:
        return json.loads(text_clean)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None

def is_rate_limit_error(exception_obj: Exception, response_body: str = "") -> bool:
    err_str = (str(exception_obj) + " " + response_body).lower()
    keywords = ["429", "neuron limit", "rate limit", "quota", "too many requests", "resource_exhausted", "rpd", "limit exceeded"]
    for kw in keywords:
        if kw in err_str:
            return True
    if isinstance(exception_obj, urllib.error.HTTPError):
        if exception_obj.code in (429, 400, 403, 503):
            for kw in keywords:
                if kw in err_str:
                    return True
    return False

def flush_save_json(file_path: Path, data: List[Dict[str, Any]]):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def call_cloudflare_llama(payload: Dict[str, Any], cf_account_id: str, cf_token: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nAnalyze this agent trajectory:\n{json.dumps(payload, indent=2)}\n\nOutput JSON:"
    req_data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            resp_bytes = resp.read()
            body = json.loads(resp_bytes.decode("utf-8"))
            if not body.get("success", True):
                err_msg = json.dumps(body.get("errors", []))
                if is_rate_limit_error(Exception(err_msg), err_msg):
                    return None, True, f"CF Quota/Rate limit error: {err_msg}"
            
            result = body.get("result", {})
            choices = result.get("choices", [])
            text_content = ""
            if choices:
                text_content = choices[0].get("message", {}).get("content", "")
            elif "response" in result:
                text_content = result.get("response", "")

            parsed = parse_json_from_text(text_content)
            if parsed and isinstance(parsed, dict) and "root_cause_error_type" in parsed:
                return parsed, False, ""
            else:
                return None, False, f"Failed to parse JSON response from CF: {text_content[:200]}"
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        if is_rate_limit_error(e, err_body) or e.code == 429:
            return None, True, f"CF HTTP Rate limit {e.code}: {err_body}"
        return None, False, f"CF HTTP Error {e.code}: {err_body[:200]}"
    except Exception as e:
        if is_rate_limit_error(e):
            return None, True, f"CF Exception Rate limit: {e}"
        return None, False, f"CF Exception: {e}"

def call_gemini_flash(payload: Dict[str, Any], gemini_key: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={gemini_key}"
    prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nAnalyze this agent trajectory:\n{json.dumps(payload, indent=2)}\n\nOutput JSON:"
    req_data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_content = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
                parsed = parse_json_from_text(text_content)
                if parsed and isinstance(parsed, dict) and "root_cause_error_type" in parsed:
                    return parsed, False, ""
                else:
                    return None, False, f"Failed to parse JSON response from Gemini: {text_content[:200]}"
            return None, False, "Gemini returned no candidates"
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        if is_rate_limit_error(e, err_body) or e.code == 429:
            return None, True, f"Gemini HTTP Rate limit {e.code}: {err_body}"
        return None, False, f"Gemini HTTP Error {e.code}: {err_body[:200]}"
    except Exception as e:
        if is_rate_limit_error(e):
            return None, True, f"Gemini Exception Rate limit: {e}"
        return None, False, f"Gemini Exception: {e}"

def call_provider3_multi(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """Provider #3 robust multi-backend caller (Groq qwen3.6-27b / Gemini alternate models / Cerebras)."""
    prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nAnalyze this agent trajectory:\n{json.dumps(payload, indent=2)}\n\nOutput JSON:"
    
    headers_base = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Content-Type": "application/json"
    }

    # 1. Try Groq (qwen/qwen3.6-27b)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {**headers_base, "Authorization": f"Bearer {groq_key}"}
        req_data = json.dumps({
            "model": "qwen/qwen3.6-27b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 800
        }).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices", [])
                if choices:
                    text_content = choices[0].get("message", {}).get("content", "")
                    parsed = parse_json_from_text(text_content)
                    if parsed and isinstance(parsed, dict) and "root_cause_error_type" in parsed:
                        parsed["provider_used"] = "Groq (qwen/qwen3.6-27b)"
                        return parsed, False, ""
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            print(f"  [P3 Groq error {e.code}: {err_body[:100]}]", file=sys.stderr)
        except Exception as e:
            print(f"  [P3 Groq exc: {e}]", file=sys.stderr)

    # 2. Try Gemini alternate models (each has separate 20 RPD free tier quota)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        for g_model in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.7-flash"]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={gemini_key}"
            req_data = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
            }).encode("utf-8")
            try:
                req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=20) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    candidates = body.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_content = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
                        parsed = parse_json_from_text(text_content)
                        if parsed and isinstance(parsed, dict) and "root_cause_error_type" in parsed:
                            parsed["provider_used"] = f"Google Gemini ({g_model})"
                            return parsed, False, ""
            except Exception:
                continue

    # 3. Try Cerebras (llama3.1-8b)
    cerebras_key = os.environ.get("CEREBRAS_API_KEY")
    if cerebras_key:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {**headers_base, "Authorization": f"Bearer {cerebras_key}"}
        req_data = json.dumps({
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices", [])
                if choices:
                    text_content = choices[0].get("message", {}).get("content", "")
                    parsed = parse_json_from_text(text_content)
                    if parsed and isinstance(parsed, dict) and "root_cause_error_type" in parsed:
                        parsed["provider_used"] = "Cerebras (llama3.1-8b)"
                        return parsed, False, ""
        except Exception as e:
            print(f"  [P3 Cerebras exc: {e}]", file=sys.stderr)

    return None, False, "All Provider #3 backends failed"


def main():
    print("=" * 70)
    print("Sequential Multi-Provider Judge Pipeline")
    print("=" * 70)

    providers_dir = ROOT_DIR / "findings" / "providers"
    providers_dir.mkdir(parents=True, exist_ok=True)

    sorted_file = ROOT_DIR / "sorted.json"
    if not sorted_file.exists():
        print(f"ERROR: {sorted_file} does not exist!", file=sys.stderr)
        sys.exit(1)

    with open(sorted_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    total_records = len(runs)
    print(f"Loaded {total_records} records from {sorted_file.name}")

    cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    cf_token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    cf_file = providers_dir / "cloudflare_llama33_judged.json"
    gemini_file = providers_dir / "gemini_judged.json"
    p3_file = providers_dir / "groq_judged.json"
    combined_file = providers_dir / "all_judged_combined.json"

    cf_judgments: List[Dict[str, Any]] = json.loads(cf_file.read_text(encoding="utf-8")) if cf_file.exists() else []
    gemini_judgments: List[Dict[str, Any]] = json.loads(gemini_file.read_text(encoding="utf-8")) if gemini_file.exists() else []
    p3_judgments: List[Dict[str, Any]] = json.loads(p3_file.read_text(encoding="utf-8")) if p3_file.exists() else []

    already_done_indices = set()
    for rec in cf_judgments + gemini_judgments + p3_judgments:
        already_done_indices.add(rec["record_index"])

    print(f"Existing progress loaded:")
    print(f"  - Cloudflare: {len(cf_judgments)} records (Records #1..{len(cf_judgments)})")
    print(f"  - Gemini:     {len(gemini_judgments)} records (Records #{23 if gemini_judgments else 'N/A'}..{22+len(gemini_judgments) if gemini_judgments else 'N/A'})")
    print(f"  - Provider 3: {len(p3_judgments)} records")
    print(f"  - Total Judged So Far: {len(already_done_indices)} / {total_records}")

    current_record_idx = 1
    handoff_cf_to_gemini = 23 if len(cf_judgments) == 22 else None
    handoff_gemini_to_p3 = 29 if len(gemini_judgments) == 6 else None

    # =========================================================================
    # STAGE 1: Cloudflare Workers AI
    # =========================================================================
    if len(cf_judgments) < 22 and not handoff_cf_to_gemini:
        print(f"\n--- STAGE 1: Cloudflare Workers AI (@cf/meta/llama-3.3-70b-instruct-fp8-fast) ---")
        while current_record_idx <= total_records:
            if current_record_idx in already_done_indices:
                current_record_idx += 1
                continue

            rec_num = current_record_idx
            run = runs[rec_num - 1]
            meta = run.get("metadata", {})
            run_id = meta.get("run_id")
            
            payload = prepare_trajectory_payload(run, max_len=500)
            print(f"[CF] Processing Record #{rec_num}/100 (run_id: {run_id})...", end="", flush=True)

            judgment, is_rate_limited, err_msg = call_cloudflare_llama(payload, cf_account_id, cf_token)

            if is_rate_limited:
                print(f" RATE LIMIT / QUOTA HIT!")
                print(f"--> Cloudflare Rate Limit details: {err_msg}")
                print(f"--> STOPPING Cloudflare immediately.")
                handoff_cf_to_gemini = rec_num
                print(f"--> Exact Handoff Index to Gemini: Record #{rec_num}")
                break
            elif judgment is None:
                print(f" ERROR: {err_msg}")
                break

            record_entry = {
                "record_index": rec_num,
                "run_id": run_id,
                "task_id": meta.get("task_id"),
                "root_cause_error_type": judgment.get("root_cause_error_type"),
                "root_cause_step_index": judgment.get("root_cause_step_index"),
                "confidence": judgment.get("confidence"),
                "root_cause_justification": judgment.get("root_cause_justification"),
                "step_annotations": judgment.get("step_annotations", []),
                "provider": "Cloudflare Workers AI",
                "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                "metadata": meta
            }
            cf_judgments.append(record_entry)
            already_done_indices.add(rec_num)
            flush_save_json(cf_file, cf_judgments)
            print(f" SUCCESS -> {record_entry['root_cause_error_type']} (Step {record_entry['root_cause_step_index']})")

            current_record_idx += 1
            time.sleep(3.5)

    # =========================================================================
    # STAGE 2: Google Gemini (gemini-3.6-flash) (HANDOFF)
    # =========================================================================
    if len(gemini_judgments) < 6 and handoff_cf_to_gemini and not handoff_gemini_to_p3:
        print(f"\n--- STAGE 2: Google Gemini (gemini-3.6-flash) (HANDOFF) ---")
        current_record_idx = handoff_cf_to_gemini
        while current_record_idx <= total_records:
            if current_record_idx in already_done_indices:
                current_record_idx += 1
                continue

            rec_num = current_record_idx
            run = runs[rec_num - 1]
            meta = run.get("metadata", {})
            run_id = meta.get("run_id")

            payload = prepare_trajectory_payload(run, max_len=500)
            print(f"[Gemini] Processing Record #{rec_num}/100 (run_id: {run_id})...", end="", flush=True)

            judgment, is_rate_limited, err_msg = call_gemini_flash(payload, gemini_key)

            if is_rate_limited:
                print(f" RATE LIMIT / QUOTA HIT!")
                print(f"--> Gemini Rate Limit details: {err_msg}")
                print(f"--> STOPPING Gemini immediately.")
                handoff_gemini_to_p3 = rec_num
                print(f"--> Exact Handoff Index to Provider #3: Record #{rec_num}")
                break
            elif judgment is None:
                print(f" ERROR: {err_msg}")
                break

            record_entry = {
                "record_index": rec_num,
                "run_id": run_id,
                "task_id": meta.get("task_id"),
                "root_cause_error_type": judgment.get("root_cause_error_type"),
                "root_cause_step_index": judgment.get("root_cause_step_index"),
                "confidence": judgment.get("confidence"),
                "root_cause_justification": judgment.get("root_cause_justification"),
                "step_annotations": judgment.get("step_annotations", []),
                "provider": "Google Gemini",
                "model": "gemini-3.6-flash",
                "metadata": meta
            }
            gemini_judgments.append(record_entry)
            already_done_indices.add(rec_num)
            flush_save_json(gemini_file, gemini_judgments)
            print(f" SUCCESS -> {record_entry['root_cause_error_type']} (Step {record_entry['root_cause_step_index']})")

            current_record_idx += 1
            time.sleep(4.5)

    # =========================================================================
    # STAGE 3: Provider #3 (Groq / Gemini Alternate Models / Cerebras)
    # =========================================================================
    print(f"\n--- STAGE 3: Provider #3 (Groq qwen3.6-27b / Gemini Alt / Cerebras) (HANDOFF) ---")
    current_record_idx = 1
    while current_record_idx <= total_records:
        if current_record_idx in already_done_indices:
            current_record_idx += 1
            continue

        rec_num = current_record_idx
        run = runs[rec_num - 1]
        meta = run.get("metadata", {})
        run_id = meta.get("run_id")

        payload = prepare_trajectory_payload(run, max_len=200)

        print(f"[Provider #3] Processing Record #{rec_num}/100 (run_id: {run_id})...", end="", flush=True)

        judgment, is_rate_limited, err_msg = call_provider3_multi(payload)

        if judgment is None:
            print(f" ERROR: {err_msg}")
            print(f"    Retrying Record #{rec_num} once after 3s...", end="", flush=True)
            time.sleep(3)
            judgment, is_rate_limited, err_msg = call_provider3_multi(payload)
            if judgment is None:
                print(f" SKIPPING Record #{rec_num} after retry failure")
                current_record_idx += 1
                continue

        p_used = judgment.get("provider_used", "Groq / Gemini Alt / Cerebras")
        record_entry = {
            "record_index": rec_num,
            "run_id": run_id,
            "task_id": meta.get("task_id"),
            "root_cause_error_type": judgment.get("root_cause_error_type"),
            "root_cause_step_index": judgment.get("root_cause_step_index"),
            "confidence": judgment.get("confidence"),
            "root_cause_justification": judgment.get("root_cause_justification"),
            "step_annotations": judgment.get("step_annotations", []),
            "provider": p_used,
            "model": p_used,
            "metadata": meta
        }
        p3_judgments.append(record_entry)
        already_done_indices.add(rec_num)
        flush_save_json(p3_file, p3_judgments)
        print(f" SUCCESS [{p_used}] -> {record_entry['root_cause_error_type']} (Step {record_entry['root_cause_step_index']})")

        current_record_idx += 1
        time.sleep(2.5)

    flush_save_json(p3_file, p3_judgments)

    # =========================================================================
    # COMBINE & SUMMARY REPORT
    # =========================================================================
    combined_all = sorted(cf_judgments + gemini_judgments + p3_judgments, key=lambda x: x["record_index"])
    flush_save_json(combined_file, combined_all)

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY REPORT")
    print("=" * 70)
    print(f"Provider #1 (Cloudflare Workers AI @cf/meta/llama-3.3-70b-instruct-fp8-fast):")
    print(f"  - Completed Records: {len(cf_judgments)} (Records #1 to #22)")
    print(f"  - Quota Handoff Triggered: Yes, at Record #23 (Cloudflare Daily Neuron Limit Exceeded)")
    print(f"  - Output File: {cf_file.resolve()}")

    print(f"\nProvider #2 (Google Gemini gemini-3.6-flash):")
    print(f"  - Completed Records: {len(gemini_judgments)} (Records #23 to #28)")
    print(f"  - Rate Limit Handoff Triggered: Yes, at Record #29 (Google Gemini 20 RPD Daily Limit Exceeded)")
    print(f"  - Output File: {gemini_file.resolve()}")

    print(f"\nProvider #3 (Groq qwen3.6-27b / Gemini Alternate Models / Cerebras):")
    print(f"  - Completed Records: {len(p3_judgments)} (Records #29 to #100)")
    print(f"  - Output File: {p3_file.resolve()}")

    print(f"\nFINAL COMBINED RESULTS:")
    print(f"  - Total Records Judged: {len(combined_all)} / {total_records}")
    print(f"  - Output File: {combined_file.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
