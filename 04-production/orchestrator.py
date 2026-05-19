"""
Orchestrators — two patterns for multi-agent coordination.

SequentialPipeline (CrewAI-style):
    Fixed order: plan → research → write.
    Predictable, one pass of each agent, no loops.

GraphOrchestrator (LangGraph-style):
    State-machine loop with conditional routing. After each research
    pass the router evaluates findings and either loops back
    (researching) or advances (writing). Capped by max_iterations.

Both patterns:
    - Publish a StateTransition event before each agent runs. The
      checkpointer (Step 5) subscribes to these and persists the
      full ResearchState snapshot, enabling --resume.
    - Expose run(ctx, query) to start fresh and resume(ctx, state)
      to continue from a loaded checkpoint. resume() dispatches
      based on state.status, so crash-then-resume is the same code
      path as "keep going" — the orchestrator doesn't care.

Checkpoint semantics:
    The StateTransition is published BEFORE the agent runs. If the
    agent crashes mid-execution, resume re-runs the entire agent
    step — no inconsistent partial state. "Checkpoint forward,
    execute after."
"""
from __future__ import annotations

from agents import researcher_gather, router_evaluate, router_plan, writer_produce
from context import RunContext
from events import StateTransition, now
from state import ResearchState


async def _transition(ctx: RunContext, state: ResearchState, to_status: str) -> None:
    """Publish a StateTransition and update state.status."""
    old = state.status
    state.status = to_status
    await ctx.bus.publish(StateTransition(
        run_id=ctx.run_id, timestamp=now(),
        from_status=old, to_status=to_status,
        snapshot=state.to_dict(),
    ))


class SequentialPipeline:
    """Fixed pipeline: plan → research → write → done."""

    async def run(self, ctx: RunContext, query: str) -> ResearchState:
        state = ResearchState(query=query)
        await self._execute_from(ctx, state)
        return state

    async def resume(self, ctx: RunContext, state: ResearchState) -> ResearchState:
        await self._execute_from(ctx, state)
        return state

    async def _execute_from(self, ctx: RunContext, state: ResearchState) -> None:
        # Each block is a no-op if state.status has already moved past it,
        # so resume() can enter partway through and do only what's left.
        if state.status == "planning":
            await _transition(ctx, state, "planning")
            state = await router_plan(ctx, state)

        if state.status == "researching":
            await _transition(ctx, state, "researching")
            state = await researcher_gather(ctx, state)

        if state.status in ("researching", "writing"):
            await _transition(ctx, state, "writing")
            state = await writer_produce(ctx, state)


class GraphOrchestrator:
    """State-machine loop with conditional routing from the router."""

    async def run(self, ctx: RunContext, query: str) -> ResearchState:
        state = ResearchState(query=query)
        await self._loop(ctx, state)
        return state

    async def resume(self, ctx: RunContext, state: ResearchState) -> ResearchState:
        await self._loop(ctx, state)
        return state

    async def _loop(self, ctx: RunContext, state: ResearchState) -> None:
        while state.status != "done":
            if state.status == "planning":
                await _transition(ctx, state, "planning")
                state = await router_plan(ctx, state)

            elif state.status == "researching":
                await _transition(ctx, state, "researching")
                state = await researcher_gather(ctx, state)

                # Conditional routing: router_evaluate inspects findings
                # and sets state.status to "writing" or "researching".
                # Hard cap on iterations as a safety net.
                if state.iteration >= state.max_iterations:
                    state.status = "writing"
                else:
                    state = await router_evaluate(ctx, state)

            elif state.status == "writing":
                await _transition(ctx, state, "writing")
                state = await writer_produce(ctx, state)

            else:
                raise RuntimeError(f"Unknown status: {state.status}")
