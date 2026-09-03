"""LLM Provider Engine for dispatching judge API calls across providers."""

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from rate_limits import _groq_rate_limit
except ImportError:
    from longhorizon_guard.taxonomy.rate_limits import _groq_rate_limit


PROVIDER_DETAILS = {
    "cloudflare": {
        "label": "Cloudflare Workers AI",
        "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    },
    "nvidia": {
        "label": "NVIDIA NIM",
        "model": "moonshotai/kimi-k3",
    },
    "groq": {
        "label": "Groq",
        "model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "label": "Google Direct API",
        "model": "gemini-3.5-flash",
    },
    "openrouter": {
        "label": "OpenRouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
    "tokenrouter": {
        "label": "TokenRouter",
        "model": "z-ai/glm-5.3-free",
    },
}


def _get_provider_credentials(provider: str) -> Dict[str, str]:
    """Return validated credentials for one explicitly selected provider."""
    if provider == "cloudflare":
        cf_key = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_KEY")
        cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
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

    env_var = {
        "nvidia": "NVIDIA_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "tokenrouter": "TOKENROUTER_API_KEY",
    }[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"Selected provider '{provider}' requires environment variable {env_var}.")
    return {"key": api_key}


async def _call_direct_gemini(prompt: str, gemini_key: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={gemini_key}"
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
            await asyncio.sleep(6.0 * (attempt + 1))
    return None


async def _call_openrouter(prompt: str, openrouter_key: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    url = "https://openrouter.ai/api/v1/chat/completions"
    req_data = json.dumps({
        "model": "meta-llama/llama-3.3-70b-instruct",
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


async def _call_cloudflare_ai(prompt: str, cf_key: str, cf_account_id: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
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


async def _call_groq(prompt: str, groq_key: str) -> Optional[Dict[str, Any]]:
    await _groq_rate_limit()
    import urllib.request
    import urllib.error

    url = "https://api.groq.com/openai/v1/chat/completions"
    req_data = json.dumps({
        "model": "openai/gpt-oss-120b",
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


async def _call_nvidia(prompt: str, nvidia_key: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    req_data = json.dumps({
        "model": "moonshotai/kimi-k3",
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


async def _call_tokenrouter(prompt: str, tokenrouter_key: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    url = "https://api.tokenrouter.com/v1/chat/completions"
    req_data = json.dumps({
        "model": "z-ai/glm-5.3-free",
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


async def call_llm(
    prompt: str,
    provider: str,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """Invoke exactly one explicitly selected provider's existing judge helper."""
    credentials = _get_provider_credentials(provider)

    async with semaphore:
        if provider == "cloudflare":
            res = await _call_cloudflare_ai(prompt, credentials["key"], credentials["account_id"])
        elif provider == "nvidia":
            res = await _call_nvidia(prompt, credentials["key"])
        elif provider == "groq":
            res = await _call_groq(prompt, credentials["key"])
        elif provider == "gemini":
            res = await _call_direct_gemini(prompt, credentials["key"])
        elif provider == "openrouter":
            res = await _call_openrouter(prompt, credentials["key"])
        elif provider == "tokenrouter":
            res = await _call_tokenrouter(prompt, credentials["key"])
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    if res and isinstance(res, dict) and "root_cause_error_type" in res:
        details = PROVIDER_DETAILS[provider]
        res["provider_used"] = f"{details['label']} ({details['model']})"
        return res

    print(
        f"ERROR: Selected provider '{provider}' did not return a valid judgment.",
        file=sys.stderr,
    )
    return None
