"""
Writer agent — turns the accumulated findings into a polished
markdown report. Text-only, no tools.

Always sets state.status to "done" on exit.
"""
from __future__ import annotations

from agents.base import BaseAgent
from context import RunContext
from state import ResearchState

WRITER_PROMPT = """\
You are the writer. Given a query and a list of findings, produce a
clear, well-structured markdown answer. Cite source URLs inline where
provided. Do not invent facts that aren't in the findings. Keep it
focused — if the findings are short, the answer can be short.
"""


class WriterAgent(BaseAgent):
    name = "writer"

    def __init__(self) -> None:
        super().__init__(system_prompt=WRITER_PROMPT, tools=[], max_iterations=2)


async def writer_produce(ctx: RunContext, state: ResearchState) -> ResearchState:
    agent = WriterAgent()
    prompt = (
        f"Query: {state.query}\n\n"
        f"Findings:\n" + "\n".join(f"- {f}" for f in state.findings)
    )
    state.report = await agent.run(ctx, prompt)
    state.status = "done"
    return state
