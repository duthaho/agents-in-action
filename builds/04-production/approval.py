"""
AsyncApprovalGate — the human-in-the-loop gate for sensitive tools.

When a tool marked `sensitive=True` is about to execute, the Tool
wrapper calls `gate.request(agent, tool, args)` and the gate blocks
on user input before letting the call proceed.

Event flow:

    1. Agent tries to execute a sensitive tool.
    2. Tool.execute sees sensitive=True, calls gate.request(...).
    3. Gate publishes ApprovalRequested.
    4. Gate prints a CLI prompt (input() in a threadpool executor).
    5. User answers y/N.
    6. Gate publishes ApprovalResolved(granted=bool).
    7. If granted → tool executes. If denied → ApprovalDenied raised,
       caught by the agent loop, and "DENIED: ..." is fed back to the
       LLM, which replans.

Why run input() in a threadpool executor:
    The event loop must stay responsive. Without the executor, blocking
    input() would freeze every other coroutine — which matters once
    streaming is active (Step 7), since token events from other agents
    would stall while a human is thinking.

CLI today, HTTP tomorrow:
    Only `_prompt()` is transport-specific. A web UI version replaces
    it with an HTTP endpoint that awaits a Future. Nothing else needs
    to change.
"""
from __future__ import annotations

import asyncio

from context import RunContext
from events import ApprovalRequested, ApprovalResolved, now


class AsyncApprovalGate:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    async def request(self, *, agent: str, tool: str, args: dict) -> bool:
        await self.ctx.bus.publish(ApprovalRequested(
            run_id=self.ctx.run_id, timestamp=now(),
            agent=agent, tool=tool, args=args,
        ))
        granted = await self._prompt(tool, args)
        await self.ctx.bus.publish(ApprovalResolved(
            run_id=self.ctx.run_id, timestamp=now(),
            tool=tool, granted=granted,
        ))
        return granted

    async def _prompt(self, tool: str, args: dict) -> bool:
        msg = f"\n[APPROVAL] {tool}({args}) — approve? [y/N]: "
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, input, msg)
        return answer.strip().lower() == "y"
