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

from dataclasses import dataclass, field
from typing import AsyncIterator

from context import RunContext
from errors import MaxRetriesExceeded
from events import (
    CostRecorded,
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallStarted,
    TokenChunk,
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


# ─── Streaming (Step 7) ────────────────────────────────────────────

@dataclass
class StreamChunk:
    """What `llm_stream` yields to its caller.

    kind="text"   → a text delta arrived. `text` holds the delta.
    kind="final"  → the stream has ended. `content` holds the full
                    concatenated text; `tool_calls` holds the
                    reassembled tool-call list.
    """
    kind: str
    text: str = ""
    content: str = ""
    tool_calls: list = field(default_factory=list)


def _merge_tool_call_deltas(acc: dict[int, dict], deltas) -> None:
    """
    OpenAI streams tool calls in fragments. Each delta carries an
    index (which tool call it belongs to), optionally an id, and
    optionally name + argument fragments that must be concatenated.

    We keep an accumulator keyed by index and merge fragments in
    place. After the stream closes, acc.values() is the completed
    tool-call list.
    """
    for d in deltas or []:
        idx = d.index
        slot = acc.setdefault(idx, {
            "id": None,
            "type": "function",
            "function": {"name": "", "arguments": ""},
        })
        if getattr(d, "id", None):
            slot["id"] = d.id
        fn = getattr(d, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["function"]["name"] = fn.name
            if getattr(fn, "arguments", None):
                slot["function"]["arguments"] += fn.arguments


async def llm_stream(
    ctx: RunContext,
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    agent_name: str = "",
    temperature: float = 0.7,
) -> AsyncIterator[StreamChunk]:
    """
    Streaming LLM call. Yields StreamChunk(kind="text") for each
    content delta, then a single StreamChunk(kind="final") when the
    stream closes.

    No retries. Retry-after-partial-stream is a rabbit hole (would
    require replaying deltas). If the connection drops, we surface
    the error; the caller can fall back to llm_call for that step.

    Tool calls execute AFTER the stream closes — fragment-by-fragment
    execution on half-parsed args is unsafe.

    `stream_options={"include_usage": True}` tells OpenAI to send a
    final chunk carrying the usage totals, so the cost event still
    fires with real numbers.
    """
    model = model or DEFAULT_MODEL
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = tools

    await ctx.bus.publish(LLMCallStarted(
        run_id=ctx.run_id, timestamp=now(),
        agent=agent_name, model=model,
    ))

    stream = await _get_client().chat.completions.create(**kwargs)
    text_parts: list[str] = []
    tool_accumulator: dict[int, dict] = {}
    usage_dict: dict = {}

    async for chunk in stream:
        # The terminal usage chunk has no choices — it's sent at the
        # very end when stream_options.include_usage is True.
        if getattr(chunk, "usage", None):
            usage_dict = (
                chunk.usage.model_dump()
                if hasattr(chunk.usage, "model_dump")
                else dict(chunk.usage)
            )
            continue
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if getattr(delta, "content", None):
            text_parts.append(delta.content)
            await ctx.bus.publish(TokenChunk(
                run_id=ctx.run_id, timestamp=now(),
                agent=agent_name, text=delta.content,
            ))
            yield StreamChunk(kind="text", text=delta.content)

        if getattr(delta, "tool_calls", None):
            _merge_tool_call_deltas(tool_accumulator, delta.tool_calls)

    await ctx.bus.publish(LLMCallCompleted(
        run_id=ctx.run_id, timestamp=now(),
        agent=agent_name, model=model, usage=usage_dict,
    ))
    if usage_dict:
        await ctx.bus.publish(CostRecorded(
            run_id=ctx.run_id, timestamp=now(),
            agent=agent_name, model=model,
            prompt_tokens=usage_dict.get("prompt_tokens", 0),
            completion_tokens=usage_dict.get("completion_tokens", 0),
            usd=_price_usd(
                model,
                usage_dict.get("prompt_tokens", 0),
                usage_dict.get("completion_tokens", 0),
            ),
        ))

    yield StreamChunk(
        kind="final",
        content="".join(text_parts),
        tool_calls=list(tool_accumulator.values()),
    )
