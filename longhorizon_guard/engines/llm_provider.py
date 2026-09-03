"""LLM Provider Engine for dispatching judge API calls across providers.

Supports:
1. Native direct providers: Cloudflare, NVIDIA, Groq, Gemini, OpenRouter, TokenRouter.
2. Generic OpenAI-compatible endpoints: Ollama, vLLM, DeepSeek, OpenAI, LocalAI, LMStudio.
3. Declarative configuration files: providers.yaml or providers.json.
4. CLI overrides for --model, --base-url, and --api-key without modifying any code.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rate_limits import _groq_rate_limit
except ImportError:
    try:
        from longhorizon_guard.taxonomy.rate_limits import _groq_rate_limit
    except ImportError:
        async def _groq_rate_limit():
            pass


PROVIDER_DETAILS: Dict[str, Dict[str, Any]] = {
    "cloudflare": {
        "label": "Cloudflare Workers AI",
        "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "type": "native",
    },
    "nvidia": {
        "label": "NVIDIA NIM",
        "model": "moonshotai/kimi-k3",
        "type": "native",
    },
    "groq": {
        "label": "Groq",
        "model": "openai/gpt-oss-120b",
        "type": "native",
    },
    "gemini": {
        "label": "Google Direct API",
        "model": "gemini-2.5-flash",
        "type": "native",
    },
    "openrouter": {
        "label": "OpenRouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "type": "native",
    },
    "tokenrouter": {
        "label": "TokenRouter",
        "model": "z-ai/glm-5.3-free",
        "type": "native",
    },
    "openai": {
        "label": "OpenAI Official",
        "model": "gpt-4o-mini",
        "type": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "label": "DeepSeek API",
        "model": "deepseek-chat",
        "type": "openai-compatible",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "local_ollama": {
        "label": "Local Ollama",
        "model": "qwen2.5:7b",
        "type": "openai-compatible",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
}


def load_provider_config(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load custom provider definitions from providers.yaml or providers.json.

    Returns a dict mapping provider name -> provider configuration dict.
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    else:
        candidates.extend([
            Path("providers.yaml"),
            Path("providers.yml"),
            Path("providers.json"),
            Path(__file__).resolve().parents[2] / "providers.yaml",
            Path(__file__).resolve().parents[2] / "providers.json",
        ])

    loaded_cfg: Dict[str, Any] = {}
    for p in candidates:
        if p.exists():
            try:
                if p.suffix in (".yaml", ".yml"):
                    try:
                        import yaml
                        with open(p, "r", encoding="utf-8") as f:
                            raw = yaml.safe_load(f)
                            if isinstance(raw, dict):
                                loaded_cfg = raw.get("providers", raw)
                                break
                    except ImportError:
                        pass
                else:
                    with open(p, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        if isinstance(raw, dict):
                            loaded_cfg = raw.get("providers", raw)
                            break
            except Exception as exc:
                print(f"Warning: Failed loading provider config from {p}: {exc}", file=sys.stderr)

    return loaded_cfg if isinstance(loaded_cfg, dict) else {}


def get_available_providers(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return all available providers by merging built-in defaults with custom config."""
    merged = dict(PROVIDER_DETAILS)
    custom = load_provider_config(config_path)
    for name, cfg in custom.items():
        if isinstance(cfg, dict):
            if name in merged:
                merged[name] = {**merged[name], **cfg}
            else:
                merged[name] = {
                    "label": cfg.get("label", name),
                    "model": cfg.get("model", "default"),
                    "type": cfg.get("type", "openai-compatible"),
                    **cfg,
                }
    return merged


def _get_provider_credentials(
    provider: str,
    custom_cfg: Optional[Dict[str, Any]] = None,
    explicit_key: Optional[str] = None,
) -> Dict[str, str]:
    """Return validated credentials for one selected provider."""
    if explicit_key:
        return {"key": explicit_key}

    cfg = custom_cfg or {}
    if "api_key" in cfg and cfg["api_key"]:
        return {"key": str(cfg["api_key"])}

    if "api_key_env" in cfg and cfg["api_key_env"]:
        key_val = os.environ.get(cfg["api_key_env"])
        if key_val:
            return {"key": key_val}
        raise RuntimeError(
            f"Provider '{provider}' requires environment variable {cfg['api_key_env']}."
        )

    if provider == "cloudflare":
        cf_key = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_KEY")
        cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or cfg.get("account_id")
        missing = []
        if not cf_key:
            missing.append("CLOUDFLARE_API_TOKEN or CLOUDFLARE_API_KEY")
        if not cf_account_id:
            missing.append("CLOUDFLARE_ACCOUNT_ID")
        if missing:
            raise RuntimeError(
                "Selected provider 'cloudflare' requires environment variable(s): "
                + ", ".join(missing)
            )
        return {"key": cf_key, "account_id": cf_account_id}

    env_map = {
        "nvidia": "NVIDIA_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "tokenrouter": "TOKENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var:
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(f"Selected provider '{provider}' requires environment variable {env_var}.")
        return {"key": api_key}

    # For local/custom providers that don't require an API key
    return {"key": "local-dummy-key"}


# ---------------------------------------------------------------------------
# Universal OpenAI-Compatible HTTP Driver
# ---------------------------------------------------------------------------

async def _call_openai_compatible(
    prompt: str,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """Invoke any OpenAI-compatible /v1/chat/completions endpoint."""
    import urllib.request
    import urllib.error

    clean_url = base_url.rstrip("/")
    if not clean_url.endswith("/chat/completions"):
        if clean_url.endswith("/v1"):
            clean_url = f"{clean_url}/chat/completions"
        else:
            clean_url = f"{clean_url}/v1/chat/completions"

    req_data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(clean_url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=timeout)
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                text_content = choices[0].get("message", {}).get("content", "")
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            if attempt == 2:
                print(f"OpenAI-compatible request to {clean_url} failed: {exc}", file=sys.stderr)
            await asyncio.sleep(2.0 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Native Provider Drivers
# ---------------------------------------------------------------------------

async def _call_direct_gemini(
    prompt: str,
    gemini_key: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    model_name = model or "gemini-2.5-flash"
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={gemini_key}"
    )
    req_data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(5):
        try:
            req = urllib.request.Request(endpoint, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=30)
            body = json.loads(resp.read().decode("utf-8"))
            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_content = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            await asyncio.sleep(4.0 * (attempt + 1))
    return None


async def _call_openrouter(
    prompt: str,
    openrouter_key: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    model_name = model or "meta-llama/llama-3.3-70b-instruct"
    url = "https://openrouter.ai/api/v1/chat/completions"
    req_data = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": 800,
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=30)
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                text_content = choices[0]["message"]["content"]
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception:
            await asyncio.sleep(2.0)
    return None


async def _call_cloudflare_ai(
    prompt: str,
    cf_key: str,
    cf_account_id: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    model_name = model or "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/{model_name}"
    req_data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {cf_key}", "Content-Type": "application/json"}

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=45)
            body = json.loads(resp.read().decode("utf-8"))
            result = body.get("result", {})
            choices = result.get("choices", [])
            text_content = ""
            if choices:
                text_content = choices[0]["message"]["content"]
            elif "response" in result:
                text_content = result["response"]

            if text_content:
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            print(f"Cloudflare call failed: {exc}", file=sys.stderr)
            await asyncio.sleep(2.0)
    return None


async def _call_groq(
    prompt: str,
    groq_key: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    await _groq_rate_limit()
    import urllib.request
    import urllib.error

    model_name = model or "openai/gpt-oss-120b"
    url = "https://api.groq.com/openai/v1/chat/completions"
    req_data = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0.1",
    }

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=30)
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                text_content = choices[0].get("message", {}).get("content", "")
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            print(f"Groq API call attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            await asyncio.sleep(2.0)
    return None


async def _call_nvidia(
    prompt: str,
    nvidia_key: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    model_name = model or "moonshotai/kimi-k3"
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    req_data = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {nvidia_key}", "Content-Type": "application/json"}

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=120)
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                text_content = choices[0].get("message", {}).get("content", "")
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            print(f"NVIDIA API call attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            await asyncio.sleep(2.0)
    return None


async def _call_tokenrouter(
    prompt: str,
    tokenrouter_key: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    model_name = model or "z-ai/glm-5.3-free"
    url = "https://api.tokenrouter.com/v1/chat/completions"
    req_data = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": 32000,
        "reasoning_effort": "low",
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {tokenrouter_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0.1",
    }

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=300)
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                text_content = choices[0].get("message", {}).get("content") or ""
                match = re.search(r"\{.*\}", text_content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            print(f"TokenRouter API call attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            await asyncio.sleep(2.0)
    return None


# ---------------------------------------------------------------------------
# Central Dispatcher
# ---------------------------------------------------------------------------

async def call_llm(
    prompt: str,
    provider: str,
    semaphore: asyncio.Semaphore,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Invoke a provider's judge helper with optional model and URL overrides."""
    available = get_available_providers(config_path)
    provider_cfg = available.get(provider, {})

    target_model = model or provider_cfg.get("model") or "default"
    credentials = _get_provider_credentials(provider, custom_cfg=provider_cfg, explicit_key=api_key)

    is_openai_compat = (
        base_url is not None
        or provider_cfg.get("type") == "openai-compatible"
        or provider == "openai-compatible"
        or "base_url" in provider_cfg
    )

    async with semaphore:
        if is_openai_compat:
            target_url = base_url or provider_cfg.get("base_url")
            if not target_url:
                raise ValueError(f"Provider '{provider}' requires a 'base_url'.")
            res = await _call_openai_compatible(
                prompt=prompt,
                base_url=target_url,
                model=target_model,
                api_key=credentials.get("key"),
                temperature=provider_cfg.get("temperature", 0.0),
                max_tokens=provider_cfg.get("max_tokens", 2048),
                timeout=provider_cfg.get("timeout", 60),
            )
        elif provider == "cloudflare":
            res = await _call_cloudflare_ai(prompt, credentials["key"], credentials["account_id"], model=target_model)
        elif provider == "nvidia":
            res = await _call_nvidia(prompt, credentials["key"], model=target_model)
        elif provider == "groq":
            res = await _call_groq(prompt, credentials["key"], model=target_model)
        elif provider == "gemini":
            res = await _call_direct_gemini(prompt, credentials["key"], model=target_model)
        elif provider == "openrouter":
            res = await _call_openrouter(prompt, credentials["key"], model=target_model)
        elif provider == "tokenrouter":
            res = await _call_tokenrouter(prompt, credentials["key"], model=target_model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    if res and isinstance(res, dict) and "root_cause_error_type" in res:
        label = provider_cfg.get("label", provider)
        res["provider_used"] = f"{label} ({target_model})"
        return res

    print(
        f"ERROR: Selected provider '{provider}' did not return a valid judgment.",
        file=sys.stderr,
    )
    return None
