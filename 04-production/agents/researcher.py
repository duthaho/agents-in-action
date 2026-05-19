"""
Researcher agent — uses web_search, calculator, and (after Step 6)
write_file to gather facts against the research plan.

Each call to researcher_gather:
    1. Reads state.research_plan and state.findings (prior passes).
    2. Runs one ReAct loop via BaseAgent.
    3. Appends the LLM's summary to state.findings (one entry per pass).
    4. Increments state.iteration.

In the GraphOrchestrator this function may be called multiple times;
the router_evaluate step decides whether to loop again.
"""
from __future__ import annotations

from agents.base import BaseAgent
from context import RunContext
from state import ResearchState
from tools import RESEARCH_TOOLS

RESEARCHER_PROMPT = """\
You are the researcher. Use the available tools to gather facts that
address the research plan. When you have enough, summarize your
findings as a compact bulleted list — one fact per bullet, each with
its source URL if available. Do not produce prose commentary or
rewrite the plan. Focus on facts.
"""


class ResearcherAgent(BaseAgent):
    name = "researcher"

    def __init__(self) -> None:
        super().__init__(
            system_prompt=RESEARCHER_PROMPT,
            tools=list(RESEARCH_TOOLS),
            max_iterations=8,
        )


async def researcher_gather(ctx: RunContext, state: ResearchState) -> ResearchState:
    agent = ResearcherAgent()
    if state.findings:
        task = (
            f"Research plan:\n{state.research_plan}\n\n"
            f"Prior findings (don't repeat these; fill gaps):\n"
            + "\n".join(f"- {f}" for f in state.findings)
        )
    else:
        task = f"Research plan:\n{state.research_plan}"

    result = await agent.run(ctx, task)
    state.findings.append(result)
    state.iteration += 1
    return state
