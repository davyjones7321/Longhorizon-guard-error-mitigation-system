"""AgentDebug 2-stage Failure Detector wrapper for longhorizon_guard."""

import os
import sys
from typing import Any, Dict, Optional

try:
    from research.agentdebug_detector.fine_grained_analysis import ErrorTypeDetector
    from research.agentdebug_detector.critical_error_detection import CriticalErrorAnalyzer
except ImportError:
    from longhorizon_guard.taxonomy.agentdebug_detector.fine_grained_analysis import ErrorTypeDetector
    from longhorizon_guard.taxonomy.agentdebug_detector.critical_error_detection import CriticalErrorAnalyzer
from longhorizon_guard.taxonomy.agentdebug_adapter import convert_to_agentdebug_format


MODULE_TO_CATEGORY: Dict[str, str] = {
    "memory": "memory_error",
    "reflection": "reflection_error",
    "planning": "planning_error",
    "action": "tool_use_error",
    "system": "external_error",
    "others": "other",
}


def _load_dotenv_if_needed() -> None:
    """Load API keys from .env file into os.environ if present."""
    from pathlib import Path
    for env_file in [Path(".env"), Path("eval") / ".env"]:
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


_load_dotenv_if_needed()


async def run_two_stage(run: Dict[str, Any], provider: str = "gemini") -> Optional[Dict[str, Any]]:
    """Run AgentDebug 2-stage failure detection pipeline on one trajectory record.

    Stage 1: Per-step module error detector (fine_grained_analysis.py)
    Stage 2: Critical error localization & cascade tracer (critical_error_detection.py)
    """
    if provider == "cloudflare":
        cf_key = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_KEY")
        cf_acc = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not cf_key or not cf_acc:
            print("ERROR: CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID environment variables not set.", file=sys.stderr)
            return None
        api_config = {
            "base_url": f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/run",
            "api_key": cf_key,
            "account_id": cf_acc,
            "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "temperature": 0.3,
            "max_retries": 3,
            "timeout": 60,
        }
        provider_label = "AgentDebug 2-stage (Cloudflare Llama 3.3 70B)"
    elif provider == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
            return None
        api_config = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "api_key": gemini_key,
            "model": "gemini-2.5-flash",
            "temperature": 0.0,
            "max_retries": 3,
            "timeout": 60,
        }
        provider_label = "AgentDebug 2-stage (Gemini gemini-2.5-flash)"
    else:
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            print("ERROR: GROQ_API_KEY environment variable not set.", file=sys.stderr)
            return None
        api_config = {
            "base_url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key": groq_key,
            "model": "openai/gpt-oss-120b",
            "temperature": 0.0,
            "max_retries": 2,
            "timeout": 60,
        }
        provider_label = "AgentDebug 2-stage (Groq openai/gpt-oss-120b)"

    try:
        # Convert raw record to AgentDebug dictionary format
        converted = convert_to_agentdebug_format(run)

        # Stage 1: Step-level per-module error detection
        detector = ErrorTypeDetector(api_config)
        phase1_results = await detector.analyze_trajectory(converted)

        # Skip successful tasks (Stage 2 handles this as well)
        if converted.get("success") or phase1_results.get("task_success"):
            return None

        # Stage 2: Critical error localization & cascading effect identification
        analyzer = CriticalErrorAnalyzer(api_config)
        critical_error = await analyzer.identify_critical_error(phase1_results, converted)

        if not critical_error:
            return None

        raw_module = (critical_error.critical_module or "others").lower()
        mapped_category = MODULE_TO_CATEGORY.get(raw_module, "other")

        # Convert 1-indexed step back to 0-indexed
        step_index = critical_error.critical_step - 1 if critical_error.critical_step is not None else 0
        if step_index < 0:
            step_index = 0

        return {
            "run_id": run.get("metadata", {}).get("run_id", "unknown"),
            "root_cause_error_type": mapped_category,
            "root_cause_step_index": step_index,
            "provider_used": provider_label,
            "raw_critical_module": critical_error.critical_module,
            "raw_error_type": critical_error.error_type,
            "cascading_effects": critical_error.cascading_effects,
            "correction_guidance": critical_error.correction_guidance,
            "confidence": critical_error.confidence,
        }

    except Exception as exc:
        print(f"ERROR in run_two_stage for {run.get('metadata', {}).get('run_id')}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None
