"""
Researcher agent — gathers facts using web search and tools.

This is the only agent with tools. Router and Writer are text-only.
The researcher takes the research plan and executes it, using web search
to find real data.

Compare to CrewAI:
  CrewAI agents can have per-agent tool sets.
  The researcher is equivalent to an Agent(role="Researcher", tools=[SearchTool]).
"""

from state import ResearchState
from tools import RESEARCH_TOOLS
from .base import BaseAgent

RESEARCHER_PROMPT = """You are a thorough research assistant.

Given a research plan or set of questions, use web search to find accurate,
current information. For each question:
  1. Search for relevant information
  2. Extract key facts, statistics, and expert opinions
  3. Note the sources

Be specific and factual. Include numbers, dates, and quotes where available.
Do NOT make up information — only report what you find via search.
If a search returns no useful results, say so clearly.

Compile all findings into a structured summary."""


def researcher_agent(state: ResearchState) -> ResearchState:
    """
    Researcher agent function: (state) → state.

    Takes the research plan, uses web search to gather facts,
    and adds findings to the shared state.
    """
    agent = BaseAgent(
        system_prompt=RESEARCHER_PROMPT,
        tools=RESEARCH_TOOLS,
        max_iterations=15,  # More iterations — researcher may need many searches
    )

    print(f"\n  [Researcher] Executing research plan (iteration {state.iteration + 1})...")

    task = (
        f"Research the following:\n\n{state.research_plan}\n\n"
        f"Use web search to find accurate, current information."
    )
    if state.findings:
        task += f"\n\nPrevious findings (avoid repeating):\n" + "\n---\n".join(state.findings[-2:])

    result = agent.run(task)
    state.findings.append(result)
    state.iteration += 1
    print(f"  [Researcher] Added findings (total: {len(state.findings)})")

    return state
