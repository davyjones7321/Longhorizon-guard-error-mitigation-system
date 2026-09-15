"""Centralized rate limiting for API providers."""

import asyncio
import time
from typing import List

_groq_call_times: List[float] = []
_groq_rate_lock = asyncio.Lock()


async def _groq_rate_limit(estimated_tokens: int = 0) -> None:
    """Enforce rate limits for Groq: max ~6 calls per 60 seconds (10s spacing) to stay well under 8000 TPM."""
    async with _groq_rate_lock:
        now = time.time()
        # Ensure at least 8.0 seconds between consecutive Groq calls
        if _groq_call_times:
            elapsed = now - _groq_call_times[-1]
            if elapsed < 8.0:
                await asyncio.sleep(8.0 - elapsed)
                now = time.time()

        while _groq_call_times and now - _groq_call_times[0] > 60.0:
            _groq_call_times.pop(0)

        if len(_groq_call_times) >= 6:
            sleep_duration = 60.0 - (now - _groq_call_times[0]) + 0.5
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            _groq_call_times.pop(0)

        _groq_call_times.append(time.time())
