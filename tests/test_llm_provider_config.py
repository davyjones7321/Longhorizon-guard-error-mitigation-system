"""Unit tests for pluggable LLM provider configuration and model overrides."""

import asyncio
import json
from pathlib import Path
import pytest

from longhorizon_guard.engines.llm_provider import (
    load_provider_config,
    get_available_providers,
    _get_provider_credentials,
    call_llm,
)


def test_load_provider_config_json(tmp_path):
    """Verify loading custom providers from a JSON file."""
    config_file = tmp_path / "providers.json"
    config_data = {
        "providers": {
            "custom_vllm": {
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "api_key": "vllm-secret-token",
            }
        }
    }
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    loaded = load_provider_config(str(config_file))
    assert "custom_vllm" in loaded
    assert loaded["custom_vllm"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert loaded["custom_vllm"]["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_get_available_providers_merging(tmp_path):
    """Verify get_available_providers merges defaults with custom configs."""
    config_file = tmp_path / "providers.json"
    config_data = {
        "providers": {
            "my_test_llm": {
                "type": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "model": "my-test-model",
            },
            "gemini": {
                "model": "gemini-2.5-pro",  # Override built-in model
            }
        }
    }
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    available = get_available_providers(str(config_file))
    # Built-in providers exist
    assert "groq" in available
    assert "cloudflare" in available
    # Custom provider merged
    assert "my_test_llm" in available
    assert available["my_test_llm"]["model"] == "my-test-model"
    # Model override honored
    assert available["gemini"]["model"] == "gemini-2.5-pro"


def test_provider_credentials_resolution(monkeypatch):
    """Verify API keys can be resolved from explicit key, config, or env var."""
    # 1. Explicit key
    creds = _get_provider_credentials("any", explicit_key="sk-explicit-123")
    assert creds["key"] == "sk-explicit-123"

    # 2. Key defined in custom config
    cfg = {"api_key": "sk-config-456"}
    creds = _get_provider_credentials("custom", custom_cfg=cfg)
    assert creds["key"] == "sk-config-456"

    # 3. Key from custom env var
    monkeypatch.setenv("MY_CUSTOM_KEY", "sk-env-789")
    cfg_env = {"api_key_env": "MY_CUSTOM_KEY"}
    creds = _get_provider_credentials("custom_env", custom_cfg=cfg_env)
    assert creds["key"] == "sk-env-789"


@pytest.mark.asyncio
async def test_call_llm_with_openai_compatible_mock(monkeypatch):
    """Verify call_llm dispatches to openai-compatible driver with correct overrides."""
    captured_args = {}

    async def mock_call_openai_compatible(prompt, base_url, model, api_key=None, **kwargs):
        captured_args["prompt"] = prompt
        captured_args["base_url"] = base_url
        captured_args["model"] = model
        captured_args["api_key"] = api_key
        return {
            "root_cause_error_type": "planning_error",
            "root_cause_step_index": 0,
            "root_cause_justification": "Mocked test response",
        }

    from longhorizon_guard.engines import llm_provider
    monkeypatch.setattr(llm_provider, "_call_openai_compatible", mock_call_openai_compatible)

    sem = asyncio.Semaphore(1)
    res = await call_llm(
        prompt="Analyze this trajectory",
        provider="openai-compatible",
        semaphore=sem,
        model="qwen2.5:14b",
        base_url="http://localhost:11434/v1",
        api_key="sk-ollama",
    )

    assert res is not None
    assert res["root_cause_error_type"] == "planning_error"
    assert captured_args["model"] == "qwen2.5:14b"
    assert captured_args["base_url"] == "http://localhost:11434/v1"
    assert captured_args["api_key"] == "sk-ollama"


def test_judge_cli_args_parsing():
    """Verify judge.py argument parser accepts custom provider, model, and base-url flags."""
    import argparse
    from longhorizon_guard.taxonomy import judge

    # Create parser mimicking judge.py main
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args([
        "--provider", "local_vllm",
        "--model", "deepseek-coder",
        "--base-url", "http://192.168.1.100:8000/v1",
        "--api-key", "my-key",
    ])

    assert args.provider == "local_vllm"
    assert args.model == "deepseek-coder"
    assert args.base_url == "http://192.168.1.100:8000/v1"
    assert args.api_key == "my-key"
