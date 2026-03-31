"""
Writer agent — produces a polished report from research findings.

No tools needed — the writer only works with text.
This is a pure LLM task: summarize, organize, and format.

Compare to CrewAI:
  This is equivalent to the last task in a sequential crew,
  which receives context from all prior tasks.
"""

from state import ResearchState
from .base import BaseAgent

WRITER_PROMPT = """You are an expert technical writer.

Given research findings, produce a clear, well-structured report.

Report format:
1. **Executive Summary** — 2-3 sentence overview
2. **Key Findings** — organized by theme, with specifics
3. **Analysis** — your synthesis of what the findings mean
4. **Conclusion** — main takeaways and recommendations

Guidelines:
- Be concise but thorough
- Use bullet points for clarity
- Include specific data points and sources when available
- Write for a technical audience
- Use markdown formatting"""


def writer_agent(state: ResearchState) -> ResearchState:
    """
    Writer agent function: (state) → state.

    Takes all gathered findings and produces a formatted report.
    """
    agent = BaseAgent(system_prompt=WRITER_PROMPT)

    print(f"\n  [Writer] Producing report from {len(state.findings)} finding(s)...")

    task = (
        f"Write a comprehensive report on: {state.query}\n\n"
        f"Research plan that was followed:\n{state.research_plan}\n\n"
        f"Research findings:\n" + "\n\n---\n\n".join(state.findings)
    )

    state.report = agent.run(task)
    state.status = "done"
    print(f"  [Writer] Report complete ({len(state.report)} chars)")

    return state
