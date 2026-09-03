"""CLI Entry point for longhorizon-guard single-command execution."""
import argparse
import json
import sys
from pathlib import Path
from longhorizon_guard import __version__
from longhorizon_guard.interface import GuardInterface

def main():
    parser = argparse.ArgumentParser(
        prog="longhorizon-guard",
        description="Long-Horizon Agent Error Mitigation & Trajectory Guard"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a JSON trajectory file for error propagation")
    eval_parser.add_argument("--trajectory", "-t", required=True, help="Path to trajectory JSON file")
    eval_parser.add_argument("--pattern-library", "-p", default=None, help="Path to pattern_library.json")

    # Command: info
    info_parser = subparsers.add_parser("info", help="Show Guard library status and loaded pattern counts")

    args = parser.parse_args()

    if args.command == "evaluate":
        tpath = Path(args.trajectory)
        if not tpath.exists():
            print(f"Error: File not found: {tpath}", file=sys.stderr)
            sys.exit(1)

        with open(tpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        steps = data.get("steps", []) if isinstance(data, dict) else data
        task_desc = data.get("task_description", "") if isinstance(data, dict) else ""
        plan = data.get("proposed_plan", "") if isinstance(data, dict) else ""

        guard = GuardInterface(pattern_library_path=args.pattern_library)
        guard.on_plan_proposed(task_desc, plan)

        history = []
        for s in steps:
            res = guard.on_step(s, history)
            history.append(s)

        summary = guard.on_run_end(metadata={"task": task_desc}, trajectory={"steps": steps})
        print("\n=== LONGHORIZON GUARD EVALUATION SUMMARY ===")
        print(json.dumps(summary, indent=2))

    elif args.command == "info" or not args.command:
        guard = GuardInterface()
        print(f"LongHorizon Guard v{__version__} Ready")
        print(f"  Loaded Patterns: {len(guard._matcher._patterns)}")
        print(f"  Broad-Corpus IDF Terms: {len(guard._matcher._idf)}")
        timeout_val = getattr(guard, "timeout", getattr(guard, "_timeout", 2.0))
        print(f"  Fail-Open Timeout: {timeout_val}s")

if __name__ == "__main__":
    main()
