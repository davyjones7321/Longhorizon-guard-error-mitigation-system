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

    # Command: proxy
    proxy_parser = subparsers.add_parser("proxy", help="Run HTTP API proxy to monitor coding assistants in real time")
    proxy_parser.add_argument("--host", "-H", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    proxy_parser.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    proxy_parser.add_argument(
        "--upstream",
        "-u",
        default="https://api.openai.com/v1",
        help="Target upstream LLM provider URL (default: https://api.openai.com/v1)",
    )
    proxy_parser.add_argument(
        "--log-dir",
        "-l",
        default="findings/proxy_sessions",
        help="Directory to save session JSONL logs (default: findings/proxy_sessions)",
    )
    proxy_parser.add_argument(
        "--no-fail-open",
        action="store_true",
        help="Raise guard errors instead of failing open",
    )

    # Command: setup-codex
    codex_parser = subparsers.add_parser("setup-codex", help="Configure LongHorizon Guard hooks automatically in OpenAI Codex (~/.codex/config.toml)")
    codex_parser.add_argument("--python", default=None, help="Custom Python path to use for hook execution")

    # Command: setup-claude
    claude_parser = subparsers.add_parser("setup-claude", help="Configure LongHorizon Guard hooks automatically in Claude Code (.claude/settings.json)")
    claude_parser.add_argument("--global", dest="is_global", action="store_true", help="Configure globally in ~/.claude/settings.json")
    claude_parser.add_argument("--dir", default=None, help="Project directory to configure (default: current directory)")
    claude_parser.add_argument("--python", default=None, help="Custom Python path to use for hook execution")

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

    elif args.command == "proxy":
        from longhorizon_guard.proxy import run_proxy

        print("\n" + "=" * 65)
        print(f"🛡️  LongHorizon Guard Real-Time API Proxy Running")
        print(f"   Listening on: http://{args.host}:{args.port}")
        print(f"   Upstream LLM: {args.upstream}")
        print(f"   Log Directory: {args.log_dir}")
        print(f"   Fail-Open:    {not args.no_fail_open}")
        print("=" * 65)
        print(f"\nTo monitor OpenCode, Cursor, Aider, or Claude Code, configure:")
        print(f"   export OPENAI_BASE_URL=\"http://{args.host}:{args.port}/v1\"")
        print(f"\nSession transcripts will be saved automatically to:\n   {args.log_dir}")
        print("\nWaiting for agent requests... (Press Ctrl+C to stop)\n")

        server = run_proxy(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            fail_open=not args.no_fail_open,
            log_dir=args.log_dir,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping LongHorizon Guard proxy...")
            server.shutdown()

    elif args.command == "setup-codex":
        py_exec = args.python or sys.executable
        codex_dir = Path.home() / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        config_file = codex_dir / "config.toml"

        # Remove conflicting legacy hooks.json
        legacy_json = codex_dir / "hooks.json"
        if legacy_json.exists():
            try:
                legacy_json.unlink()
                print(f"  Cleaned up legacy {legacy_json}")
            except Exception:
                pass

        hook_cmd = f"{py_exec} -m longhorizon_guard.hook".replace("\\", "\\\\")
        toml_snippet = f"""
[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = '{hook_cmd}'
timeout = 30
statusMessage = "LongHorizon Guard is checking the task"

[[hooks.PreToolUse]]
matcher = ".*"
[[hooks.PreToolUse.hooks]]
type = "command"
command = '{hook_cmd}'
timeout = 30
statusMessage = "LongHorizon Guard is checking the action"

[[hooks.PostToolUse]]
matcher = ".*"
[[hooks.PostToolUse.hooks]]
type = "command"
command = '{hook_cmd}'
timeout = 30
statusMessage = "LongHorizon Guard is reviewing the result"

[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = '{hook_cmd}'
timeout = 30
statusMessage = "LongHorizon Guard is finalizing the run"
"""
        content = config_file.read_text(encoding="utf-8") if config_file.exists() else ""
        if "longhorizon_guard.hook" in content:
            print(f"🛡️  LongHorizon Guard hooks are already registered in: {config_file}")
        else:
            with open(config_file, "a", encoding="utf-8") as f:
                f.write("\n# LongHorizon Guard Lifecycle Hooks\n" + toml_snippet.strip() + "\n")
            print(f"🛡️  Successfully configured LongHorizon Guard hooks in: {config_file}")
        print(f"   Python executable: {py_exec}")
        print("   OpenAI Codex is ready to run with live LongHorizon Guard protection!\n")

    elif args.command == "setup-claude":
        py_exec = args.python or sys.executable
        if args.is_global:
            claude_dir = Path.home() / ".claude"
        else:
            base = Path(args.dir) if args.dir else Path.cwd()
            claude_dir = base / ".claude"

        claude_dir.mkdir(parents=True, exist_ok=True)
        settings_file = claude_dir / "settings.json"

        settings_data = {}
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)
            except Exception:
                settings_data = {}

        hook_cmd = f"{py_exec} -m longhorizon_guard.hook"
        hooks_config = settings_data.setdefault("hooks", {})
        hooks_config["UserPromptSubmit"] = [{"type": "command", "command": hook_cmd, "timeout": 30}]
        hooks_config["PreToolUse"] = [{"matcher": ".*", "hooks": [{"type": "command", "command": hook_cmd, "timeout": 30}]}]
        hooks_config["PostToolUse"] = [{"matcher": ".*", "hooks": [{"type": "command", "command": hook_cmd, "timeout": 30}]}]
        hooks_config["Stop"] = [{"hooks": [{"type": "command", "command": hook_cmd, "timeout": 30}]}]

        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=2)

        print(f"🛡️  Successfully configured LongHorizon Guard hooks in: {settings_file}")
        print(f"   Python executable: {py_exec}")
        print("   Claude Code is ready to run with live LongHorizon Guard protection!\n")

    elif args.command == "info" or not args.command:
        guard = GuardInterface()
        print(f"LongHorizon Guard v{__version__} Ready")
        print(f"  Loaded Patterns: {len(guard._matcher._patterns)}")
        print(f"  Broad-Corpus IDF Terms: {len(guard._matcher._idf)}")
        timeout_val = getattr(guard, "timeout", getattr(guard, "_timeout", 2.0))
        print(f"  Fail-Open Timeout: {timeout_val}s")

if __name__ == "__main__":
    main()
