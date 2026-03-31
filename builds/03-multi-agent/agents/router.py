"""
Router agent — analyzes the request and creates a research plan.

In the graph pattern, it also evaluates whether findings are sufficient.

Compare to CrewAI's hierarchical process:
  A "manager agent" receives all tasks and delegates to appropriate agents.
  Our router is a simplified version of that concept.
"""

from state import ResearchState
from .base import BaseAgent

ROUTER_PROMPT = """You are a research planning specialist.

Your job depends on the current phase:

PLANNING PHASE (no findings yet):
  Analyze the user's query and create a clear, structured research plan.
  Break it into 2-4 specific research questions that can be answered via web search.
  Return the plan as a numbered list of research questions.

EVALUATION PHASE (findings exist):
  Review the gathered findings against the original query.
  Decide if we have enough information to write a comprehensive report.

  If SUFFICIENT: respond with exactly "SUFFICIENT" on the first line,
  followed by a brief summary of what we have.

  If INSUFFICIENT: respond with exactly "INSUFFICIENT" on the first line,
  followed by 1-2 additional research questions to fill the gaps."""


def router_agent(state: ResearchState) -> ResearchState:
    """
    Router agent function: (state) → state.

    This is the LangGraph pattern — each agent is a function that
    transforms shared state. CrewAI does this implicitly through
    task context; we make it explicit.
    """
    agent = BaseAgent(system_prompt=ROUTER_PROMPT)

    if not state.findings:
        # Planning phase — create research plan
        print(f"\n  [Router] Creating research plan for: {state.query}")
        task = f"Create a research plan for this query: {state.query}"
        state.research_plan = agent.run(task)
        state.status = "researching"
        print(f"  [Router] Plan created")
    else:
        # Evaluation phase — check if findings are sufficient
        print(f"\n  [Router] Evaluating {len(state.findings)} findings...")
        task = (
            f"Original query: {state.query}\n\n"
            f"Research plan: {state.research_plan}\n\n"
            f"Findings so far:\n" + "\n---\n".join(state.findings) +
            f"\n\nAre these findings sufficient to write a comprehensive report?"
        )
        evaluation = agent.run(task)

        if evaluation.strip().upper().startswith("SUFFICIENT"):
            state.status = "writing"
            print(f"  [Router] Findings are sufficient → moving to writer")
        else:
            state.status = "researching"
            state.research_plan = evaluation  # Updated plan with gap-filling questions
            print(f"  [Router] Need more research → back to researcher")

    return state
