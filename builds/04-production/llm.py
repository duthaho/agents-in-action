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

import os
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from context import RunContext
from events import LLMCallStarted, LLMCallCompleted, now

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
    model = model or DEFAULT_MODEL

    await ctx.bus.publish(LLMCallStarted(
        run_id=ctx.run_id, timestamp=now(),
        agent=agent_name, model=model,
    ))

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    response = await _get_client().chat.completions.create(**kwargs)
    message = response.choices[0].message
    usage = response.usage.model_dump() if response.usage else {}

    await ctx.bus.publish(LLMCallCompleted(
        run_id=ctx.run_id, timestamp=now(),
        agent=agent_name, model=model, usage=usage,
    ))

    return message
