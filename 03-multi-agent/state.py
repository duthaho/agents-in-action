"""
Shared state — the data structure that flows between agents.

This is the Phase 3 equivalent of LangGraph's State (a TypedDict).
In LangGraph: each node receives state, modifies it, returns it.
In CrewAI: task outputs flow via the `context` field.

We use a dataclass because it's simpler and type-checked.

Key insight: in multi-agent systems, agents don't talk to each other
directly. They communicate through shared state. This decouples agents
and makes the system easier to debug and test.
"""

from dataclasses import dataclass, field


@dataclass
class ResearchState:
    """
    State object shared between all agents in the pipeline.

    Compare to LangGraph's approach:
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            query: str

    Compare to CrewAI's approach:
        Task(description="...", context=[previous_task])
        # context is implicitly the output of prior tasks
    """

    # Input
    query: str                          # Original user request

    # Router output
    research_plan: str = ""             # What to research and how

    # Researcher output
    findings: list[str] = field(default_factory=list)  # Gathered facts

    # Writer output
    report: str = ""                    # Final polished report

    # Control flow
    status: str = "planning"            # planning | researching | writing | done
    iteration: int = 0                  # Loop counter (for graph pattern)
    max_iterations: int = 3             # Safety limit for research loops
