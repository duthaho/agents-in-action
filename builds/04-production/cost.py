"""
CostTracker — a subscriber that aggregates CostRecorded events.

Two rollups:
    .total      running total across the whole run
    .by_agent   per-agent breakdown (router/researcher/writer)

Not a singleton, not coupled to any specific bus or run — just a
plain object that has a `handle` coroutine you hand to
`bus.subscribe(tracker.handle)`. Create one per run and dispose.

Why a separate file instead of mixing into llm.py: the *emission* of
cost events belongs in llm.py (it's where we see the usage). The
*aggregation* is a policy decision, swap-in-able. A production system
might replace CostTracker with a Prometheus exporter and not touch
llm.py at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from events import CostRecorded, Event


@dataclass
class CostSummary:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0


class CostTracker:
    def __init__(self) -> None:
        self.total = CostSummary()
        self.by_agent: dict[str, CostSummary] = {}

    async def handle(self, event: Event) -> None:
        if not isinstance(event, CostRecorded):
            return
        self._add(self.total, event)
        agent_summary = self.by_agent.setdefault(event.agent, CostSummary())
        self._add(agent_summary, event)

    @staticmethod
    def _add(summary: CostSummary, event: CostRecorded) -> None:
        summary.prompt_tokens += event.prompt_tokens
        summary.completion_tokens += event.completion_tokens
        summary.usd += event.usd

    def report(self) -> str:
        lines = [
            f"Total: {self.total.prompt_tokens} in + "
            f"{self.total.completion_tokens} out = ${self.total.usd:.6f}"
        ]
        for agent, s in self.by_agent.items():
            lines.append(
                f"  {agent}: {s.prompt_tokens} in + "
                f"{s.completion_tokens} out = ${s.usd:.6f}"
            )
        return "\n".join(lines)
