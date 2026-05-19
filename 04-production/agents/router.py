"""
Router agent — two roles:

    router_plan      At the start of a run, turn the user query into
                     a concrete research plan (2-4 bullets). Advances
                     state from "planning" to "researching".

    router_evaluate  Only used by the GraphOrchestrator. After each
                     research pass, decide whether findings are
                     sufficient to write the report, or whether the
                     researcher should loop again.

Both are text-only — the router never calls tools. The LLM reads the
plan + findings and returns either READY or CONTINUE.
"""
from __future__ import annotations

from agents.base import BaseAgent
from context import RunContext
from state import ResearchState

PLAN_PROMPT = """\
You are the research router. Given a user query, produce a short,
concrete research plan: 2-4 bullet points describing what to look up
or compute. Do not attempt to answer the question — only plan the
research. Be specific about what facts to gather.
"""

EVAL_PROMPT = """\
You are the research router. You have the original query, the plan,
and a list of findings gathered so far. Decide if the findings are
enough to write a solid answer.

Respond with exactly one of:
    CONTINUE: <one sentence on what's still missing>
    READY: <one sentence confirming coverage>

Do not write the answer itself.
"""


class RouterPlanAgent(BaseAgent):
    name = "router"

    def __init__(self) -> None:
        super().__init__(system_prompt=PLAN_PROMPT, tools=[], max_iterations=2)


class RouterEvalAgent(BaseAgent):
    name = "router"

    def __init__(self) -> None:
        super().__init__(system_prompt=EVAL_PROMPT, tools=[], max_iterations=2)


async def router_plan(ctx: RunContext, state: ResearchState) -> ResearchState:
    agent = RouterPlanAgent()
    plan = await agent.run(ctx, state.query)
    state.research_plan = plan
    state.status = "researching"
    return state


async def router_evaluate(ctx: RunContext, state: ResearchState) -> ResearchState:
    agent = RouterEvalAgent()
    prompt = (
        f"Query: {state.query}\n\n"
        f"Plan:\n{state.research_plan}\n\n"
        f"Findings so far:\n" + "\n".join(f"- {f}" for f in state.findings)
    )
    verdict = await agent.run(ctx, prompt)
    if verdict.strip().upper().startswith("READY"):
        state.status = "writing"
    else:
        state.status = "researching"
    return state
