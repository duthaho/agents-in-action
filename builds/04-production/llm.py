"""
Async LLM wrapper.

This file grows over three steps:

    Step 1 (now):  llm_call(ctx, ...) — one async call, publishes
                   LLMCallStarted before and LLMCallCompleted after.
                   No retries, no streaming, no cost emission.

    Step 3:        Add a price table and publish CostRecorded from
                   usage on every successful call.

    Step 4:        Wrap llm_call in exponential-backoff retry for an
                   enumerated set of transient errors
                   (APITimeoutError, APIConnectionError,
                    RateLimitError, InternalServerError). Publishes
                   LLMCallFailed on each failed attempt and raises
                   MaxRetriesExceeded when exhausted.

    Step 7:        Add llm_stream(ctx, ...) — async generator that
                   yields text deltas and publishes TokenChunk events.
                   Retries are intentionally not wrapped around the
                   stream path.

The client is lazily constructed so importing this module doesn't
require OPENAI_API_KEY to be set (important for tests and for
running --help).
"""
from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessage

from context import RunContext
from errors import MaxRetriesExceeded
from events import (
    CostRecorded,
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallStarted,
    now,
)

# Retry policy for transient LLM errors. Permanent errors
# (BadRequestError, AuthenticationError, etc.) propagate as-is —
# retrying them is wasted money.
MAX_ATTEMPTS = 4
_RETRYABLE = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)
_BACKOFF_BASE = 0.5       # seconds, doubled each retry
_BACKOFF_JITTER = 0.25    # +U(0, JITTER) on each sleep

# USD per 1,000 tokens. Hardcoded — a real system reads this from
# config or a pricing API. Unknown models fall back to 0.0.
MODEL_PRICES_USD_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini":   {"in": 0.00015, "out": 0.0006},
    "gpt-4o":        {"in": 0.0025,  "out": 0.01},
    "gpt-4.1-mini":  {"in": 0.0004,  "out": 0.0016},
    "gpt-4.1":       {"in": 0.002,   "out": 0.008},
}


def _price_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = MODEL_PRICES_USD_PER_1K.get(model)
    if not p:
        return 0.0
    return (prompt_tokens / 1000) * p["in"] + (completion_tokens / 1000) * p["out"]

_client: AsyncOpenAI | None = None

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


async def llm_call(
    ctx: RunContext,
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    agent_name: str = "",
    temperature: float = 0.7,
) -> ChatCompletionMessage:
    """
    Call the LLM with exponential-backoff retry on transient errors.

    On each attempt we publish LLMCallStarted. On success we publish
    LLMCallCompleted (+ CostRecorded if usage is available) and
    return. On a retryable failure we publish LLMCallFailed with the
    attempt number, sleep with jitter, and try again. After
    MAX_ATTEMPTS we raise MaxRetriesExceeded.

    Non-retryable errors (BadRequestError, AuthenticationError, ...)
    propagate as-is — they're permanent.
    """
    model = model or DEFAULT_MODEL

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    delay = _BACKOFF_BASE

    for attempt in range(1, MAX_ATTEMPTS + 1):
        await ctx.bus.publish(LLMCallStarted(
            run_id=ctx.run_id, timestamp=now(),
            agent=agent_name, model=model,
        ))

        try:
            response = await _get_client().chat.completions.create(**kwargs)
            message = response.choices[0].message
            usage = response.usage.model_dump() if response.usage else {}

            await ctx.bus.publish(LLMCallCompleted(
                run_id=ctx.run_id, timestamp=now(),
                agent=agent_name, model=model, usage=usage,
            ))

            if usage:
                await ctx.bus.publish(CostRecorded(
                    run_id=ctx.run_id, timestamp=now(),
                    agent=agent_name, model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    usd=_price_usd(
                        model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    ),
                ))

            return message

        except _RETRYABLE as e:
            await ctx.bus.publish(LLMCallFailed(
                run_id=ctx.run_id, timestamp=now(),
                agent=agent_name, model=model,
                attempt=attempt, error=repr(e),
            ))
            if attempt == MAX_ATTEMPTS:
                raise MaxRetriesExceeded(f"{type(e).__name__}: {e}") from e
            await asyncio.sleep(delay + random.uniform(0, _BACKOFF_JITTER))
            delay *= 2

    # Unreachable — either we return from the try or raise from the except.
    raise MaxRetriesExceeded("retry loop exited without return")
