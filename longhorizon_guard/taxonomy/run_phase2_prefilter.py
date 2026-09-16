import json
import re
import difflib
from typing import List, Dict, Tuple, Optional, Any

def normalize_conversations(conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes a trajectory's conversation turns into a sequence of canonical action signatures.
    Returns a list of dicts:
      [
        {
          "turn_index": int,           # Original turn index in conversation
          "tool_name": str,            # Canonical tool name
          "args_string": str,          # Normalized string representation of args
          "action_signature": str,     # tool_name(args_string)
        },
        ...
      ]
    Rules:
    - Skip turns that are pure reasoning/<think> blocks with no <tool_call>
    - Skip "toolResult" roles entirely (plain stdout duplicates of "tool" roles)
    - Skip "tool" roles (observation/responses)
    - Skip "system" and "human" turns
    - Extract <tool_call> blocks from "gpt" turns
    """
    actions = []
    
    for turn_idx, turn in enumerate(conv):
        role = turn.get("from") or turn.get("role")
        # Only gpt/assistant turns initiate actions
        if role != "gpt":
            continue
            
        val = str(turn.get("value") or turn.get("content") or "")
        
        # Check for tool calls
        if "<tool_call>" not in val:
            # Pure reasoning or text turn, skip as instructed
            continue
            
        calls = re.findall(r"<tool_call>(.*?)</tool_call>", val, re.DOTALL)
        for call_text in calls:
            call_text = call_text.strip()
            try:
                parsed = json.loads(call_text)
                name = parsed.get("name", "unknown")
                args = parsed.get("args", {})
                
                # Normalize args into a deterministic signature string
                if isinstance(args, dict):
                    # For terminal/bash, normalize command
                    if "command" in args:
                        arg_summary = args["command"].strip()
                    elif "code" in args:
                        # Normalize multiline code to first line or compact
                        code_lines = args["code"].strip().split("\n")
                        arg_summary = code_lines[0][:100]
                    else:
                        arg_summary = json.dumps(args, sort_keys=True)
                else:
                    arg_summary = str(args)
                    
                sig = f"{name}:{arg_summary}"
                actions.append({
                    "turn_index": turn_idx,
                    "tool_name": name,
                    "args_string": arg_summary,
                    "action_signature": sig
                })
            except Exception:
                sig = f"raw:{call_text[:100]}"
                actions.append({
                    "turn_index": turn_idx,
                    "tool_name": "raw",
                    "args_string": call_text[:100],
                    "action_signature": sig
                })
                
    return actions


def compute_divergence_window(
    success_actions: List[Dict[str, Any]], 
    failure_actions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Uses difflib.SequenceMatcher over normalized action signatures to identify:
    - matching blocks
    - the first divergence point (first mismatch after prefix matching or index 0)
    - remaining window sizes
    """
    s_sigs = [a["action_signature"] for a in success_actions]
    f_sigs = [a["action_signature"] for a in failure_actions]
    
    if not s_sigs or not f_sigs:
        return {
            "divergence_action_idx_success": 0,
            "divergence_action_idx_failure": 0,
            "divergence_turn_success": success_actions[0]["turn_index"] if success_actions else 0,
            "divergence_turn_failure": failure_actions[0]["turn_index"] if failure_actions else 0,
            "total_actions_success": len(s_sigs),
            "total_actions_failure": len(f_sigs),
            "remaining_actions_success": len(s_sigs),
            "remaining_actions_failure": len(f_sigs),
            "matching_blocks": [],
            "status": "missing_actions_in_one_or_both"
        }
        
    matcher = difflib.SequenceMatcher(None, s_sigs, f_sigs)
    matching_blocks = matcher.get_matching_blocks() # list of Match(a, b, size)
    
    # Check common prefix
    prefix_len = 0
    for block in matching_blocks:
        if block.a == 0 and block.b == 0:
            prefix_len = block.size
            break
            
    div_s = prefix_len
    div_f = prefix_len
    
    s_turn = success_actions[div_s]["turn_index"] if div_s < len(success_actions) else success_actions[-1]["turn_index"]
    f_turn = failure_actions[div_f]["turn_index"] if div_f < len(failure_actions) else failure_actions[-1]["turn_index"]
    
    rem_s = len(success_actions) - div_s
    rem_f = len(failure_actions) - div_f
    
    return {
        "divergence_action_idx_success": div_s,
        "divergence_action_idx_failure": div_f,
        "divergence_turn_success": s_turn,
        "divergence_turn_failure": f_turn,
        "total_actions_success": len(success_actions),
        "total_actions_failure": len(failure_actions),
        "remaining_actions_success": rem_s,
        "remaining_actions_failure": rem_f,
        "matching_blocks": [(m.a, m.b, m.size) for m in matching_blocks if m.size > 0],
        "status": "aligned"
    }


def main():
    path = r"d:\edge-downloades\d\intership-projects\error-prop\new_dataset\divergence_pairs.jsonl"
    
    with open(path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
        
    print(f"Total pairs to analyze: {len(pairs)}\n")
    
    results = []
    
    for idx, p in enumerate(pairs):
        task_id = p["task_id"]
        bench = p["bench"]
        s_agent = p["success"]["agent"]
        f_agent = p["failure"]["agent"]
        s_model = p["success"]["model"]
        f_model = p["failure"]["model"]
        
        s_actions = normalize_conversations(p["success"]["conversations"])
        f_actions = normalize_conversations(p["failure"]["conversations"])
        
        div_info = compute_divergence_window(s_actions, f_actions)
        
        results.append({
            "pair_index": idx,
            "task_id": task_id,
            "bench": bench,
            "success_agent": s_agent,
            "success_model": s_model,
            "failure_agent": f_agent,
            "failure_model": f_model,
            "success_actions": s_actions,
            "failure_actions": f_actions,
            "div_info": div_info
        })

    # Print summary table
    print(f"{'#':<3} | {'Task ID':<28} | {'Bench':<16} | {'Agents (S vs F)':<24} | {'Div Act (S/F)':<14} | {'Div Turn (S/F)':<15} | {'Rem Acts (S/F)':<14}")
    print("-" * 125)
    for r in results:
        t_id = r["task_id"][:28]
        bench_str = r["bench"][:16]
        ag_str = f"{r['success_agent'][:10]} vs {r['failure_agent'][:10]}"
        d_info = r["div_info"]
        div_act = f"{d_info['divergence_action_idx_success']} / {d_info['divergence_action_idx_failure']}"
        div_turn = f"T{d_info['divergence_turn_success']} / T{d_info['divergence_turn_failure']}"
        rem_act = f"{d_info['remaining_actions_success']} / {d_info['remaining_actions_failure']}"
        print(f"{r['pair_index']:<3} | {t_id:<28} | {bench_str:<16} | {ag_str:<24} | {div_act:<14} | {div_turn:<15} | {rem_act:<14}")
        
    print("\n" + "="*80)
    print("DETAILED EXAMPLES (3 Concrete Divergence Windows)")
    print("="*80)
    
    # Select 3 distinct pairs with meaningful shared prefix blocks
    examples_with_prefix = [
        r for r in results 
        if r["div_info"]["divergence_action_idx_success"] > 0
    ]
    # If fewer than 3, grab another hermes pair with actions
    other_examples = [
        r for r in results
        if r["div_info"]["divergence_action_idx_success"] == 0 and r["div_info"]["total_actions_success"] > 2 and r["div_info"]["total_actions_failure"] > 2
    ]
    selected_examples = (examples_with_prefix + other_examples)[:3]
    
    for ex in selected_examples:
        print(f"\n--- Example {ex['pair_index']}: Task '{ex['task_id']}' ({ex['bench']}) ---")
        print(f"Success: agent='{ex['success_agent']}', model='{ex['success_model']}' ({ex['div_info']['total_actions_success']} actions)")
        print(f"Failure: agent='{ex['failure_agent']}', model='{ex['failure_model']}' ({ex['div_info']['total_actions_failure']} actions)")
        print(f"Matching blocks (a_start, b_start, length): {ex['div_info']['matching_blocks']}")
        
        div_s = ex["div_info"]["divergence_action_idx_success"]
        div_f = ex["div_info"]["divergence_action_idx_failure"]
        print(f"Divergence Action Index: Success = Action #{div_s}, Failure = Action #{div_f}")
        print(f"Divergence Turn Index:   Success = Turn #{ex['div_info']['divergence_turn_success']}, Failure = Turn #{ex['div_info']['divergence_turn_failure']}")
        print(f"Remaining Actions Window: Success = {ex['div_info']['remaining_actions_success']} actions, Failure = {ex['div_info']['remaining_actions_failure']} actions")
        
        # Actions leading up to divergence
        print("\n  [Actions BEFORE divergence point]:")
        if div_s > 0:
            for k in range(max(0, div_s - 2), div_s):
                print(f"    Shared Action #{k}: {ex['success_actions'][k]['action_signature']}")
        else:
            print("    (None - sequences diverged immediately on the very first action)")
            
        print("\n  [Actions AT / AFTER divergence point]:")
        print("    SUCCESS branch actions around divergence:")
        for k in range(div_s, min(div_s + 3, len(ex["success_actions"]))):
            turn_num = ex["success_actions"][k]["turn_index"]
            sig = ex["success_actions"][k]["action_signature"]
            print(f"      + Action #{k} (Turn {turn_num}): {sig}")
            
        print("    FAILURE branch actions around divergence:")
        for k in range(div_f, min(div_f + 3, len(ex["failure_actions"]))):
            turn_num = ex["failure_actions"][k]["turn_index"]
            sig = ex["failure_actions"][k]["action_signature"]
            print(f"      - Action #{k} (Turn {turn_num}): {sig}")

if __name__ == "__main__":
    main()
