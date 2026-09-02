"""Centralized rate limiting for API providers."""

import asyncio
import time
from typing import List

_groq_call_times: List[float] = []
_groq_rate_lock = asyncio.Lock()


async def _groq_rate_limit(estimated_tokens: int = 0) -> None:
    """Enforce max 25 calls per rolling 60-second window for Groq."""
    async with _groq_rate_lock:
        now = time.time()
        while _groq_call_times and now - _groq_call_times[0] > 60.0:
            _groq_call_times.pop(0)

        if len(_groq_call_times) >= 25:
            sleep_duration = 60.0 - (now - _groq_call_times[0]) + 0.1
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            _groq_call_times.pop(0)

        _groq_call_times.append(time.time())
