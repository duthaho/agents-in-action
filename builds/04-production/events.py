"""
Event bus and event types — the central plumbing.

Every significant action in the system publishes an Event to the bus.
Production features are implemented as subscribers (not instrumentation
sprinkled through agent code):

    - JsonlLogger       writes events as JSON lines
    - CostTracker       aggregates CostRecorded events
    - SqliteCheckpointer persists ResearchState on StateTransition
    - token printer     renders TokenChunk to stdout (streaming mode)

Design: sequential fan-out (`for h in subs: await h(e)`), not
asyncio.gather. Sequential preserves event ordering — the logger and
checkpointer must see events in the order they happened, which matters
for replay and for crash-safety.

Trade-off: a slow subscriber back-pressures the agent loop. Accepted
for learning. A production system would hand events to a background
queue.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class Event:
    run_id: str
    timestamp: float
    kind: str = "event"


# ─── LLM lifecycle ─────────────────────────────────────────────────

@dataclass
class LLMCallStarted(Event):
    agent: str = ""
    model: str = ""
    kind: str = "llm.call.started"


@dataclass
class LLMCallCompleted(Event):
    agent: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    kind: str = "llm.call.completed"


@dataclass
class LLMCallFailed(Event):
    agent: str = ""
    model: str = ""
    attempt: int = 0
    error: str = ""
    kind: str = "llm.call.failed"


# ─── Tool lifecycle ────────────────────────────────────────────────

@dataclass
class ToolCallStarted(Event):
    agent: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    kind: str = "tool.call.started"


@dataclass
class ToolCallCompleted(Event):
    agent: str = ""
    tool: str = ""
    result_preview: str = ""
    kind: str = "tool.call.completed"


@dataclass
class ToolCallFailed(Event):
    agent: str = ""
    tool: str = ""
    error: str = ""
    kind: str = "tool.call.failed"


# ─── Orchestration ─────────────────────────────────────────────────

@dataclass
class StateTransition(Event):
    from_status: str = ""
    to_status: str = ""
    snapshot: dict = field(default_factory=dict)
    kind: str = "state.transition"


# ─── HITL ──────────────────────────────────────────────────────────

@dataclass
class ApprovalRequested(Event):
    agent: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    kind: str = "approval.requested"


@dataclass
class ApprovalResolved(Event):
    tool: str = ""
    granted: bool = False
    kind: str = "approval.resolved"


# ─── Streaming ─────────────────────────────────────────────────────

@dataclass
class TokenChunk(Event):
    agent: str = ""
    text: str = ""
    kind: str = "token.chunk"


# ─── Cost ──────────────────────────────────────────────────────────

@dataclass
class CostRecorded(Event):
    agent: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    kind: str = "cost.recorded"


# ─── The bus ───────────────────────────────────────────────────────

Handler = Callable[[Event], Awaitable[None]]


class AsyncEventBus:
    def __init__(self) -> None:
        self._subs: list[Handler] = []

    def subscribe(self, handler: Handler) -> None:
        self._subs.append(handler)

    async def publish(self, event: Event) -> None:
        for handler in self._subs:
            await handler(event)


def now() -> float:
    return time.time()
