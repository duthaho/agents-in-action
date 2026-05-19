"""
Orchestrator — two patterns for multi-agent coordination.

Pattern 1: SequentialPipeline (CrewAI-style)
  Fixed order: router → researcher → writer
  Simple and predictable. Each agent runs once.

Pattern 2: GraphOrchestrator (LangGraph-style)
  State machine with conditional routing and loops.
  Router can send back to researcher if findings are insufficient.
  More adaptive but more complex.

Compare to CrewAI's _execute_tasks() in crew.py:
  CrewAI iterates through tasks in order, passing context forward.
  Our SequentialPipeline does the same with explicit function calls.

Compare to LangGraph's StateGraph:
  LangGraph defines nodes + edges + conditional edges, then compiles.
  Our GraphOrchestrator manually implements the same state machine logic.
  (We don't use LangGraph as a dependency — we build the pattern ourselves.)
"""

from state import ResearchState
from agents import router_agent, researcher_agent, writer_agent


class SequentialPipeline:
    """
    CrewAI-style: fixed pipeline, each agent runs exactly once.

    router → researcher → writer → done

    Pros: Simple, predictable, easy to debug
    Cons: Can't loop back if research is insufficient
    """

    def run(self, query: str) -> ResearchState:
        state = ResearchState(query=query)

        print("=" * 60)
        print("  SEQUENTIAL PIPELINE")
        print(f"  Query: {query}")
        print("=" * 60)

        # Step 1: Router creates research plan
        state = router_agent(state)

        # Step 2: Researcher gathers facts
        state = researcher_agent(state)

        # Step 3: Writer produces report
        state.status = "writing"
        state = writer_agent(state)

        return state


class GraphOrchestrator:
    """
    LangGraph-style: state machine with conditional routing.

    ┌──────────┐     ┌────────────┐     ┌──────────┐
    │  router  │────►│ researcher │────►│  router   │
    │ (plan)   │     │ (search)   │     │ (evaluate)│
    └──────────┘     └────────────┘     └─────┬─────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              INSUFFICIENT          SUFFICIENT
                                    │                   │
                                    ▼                   ▼
                              researcher            writer → done

    Compare to LangGraph:
        graph = StateGraph(State)
        graph.add_node("router", router_agent)
        graph.add_node("researcher", researcher_agent)
        graph.add_node("writer", writer_agent)
        graph.add_conditional_edges("router", route_fn, {...})
        graph.compile()

    We implement the same logic manually — the state machine pattern
    is the important concept, not the framework API.
    """

    def run(self, query: str) -> ResearchState:
        state = ResearchState(query=query)

        print("=" * 60)
        print("  GRAPH ORCHESTRATOR")
        print(f"  Query: {query}")
        print("=" * 60)

        # The state machine loop
        while state.status != "done":
            print(f"\n  --- State: {state.status} (iteration {state.iteration}) ---")

            if state.status == "planning":
                state = router_agent(state)

            elif state.status == "researching":
                state = researcher_agent(state)
                # After research, go back to router for evaluation
                # This is the "conditional edge" in LangGraph terms
                state = self._route_after_research(state)

            elif state.status == "writing":
                state = writer_agent(state)

            else:
                print(f"  ⚠ Unknown status: {state.status}")
                break

        return state

    def _route_after_research(self, state: ResearchState) -> ResearchState:
        """
        Conditional routing — the key concept from LangGraph.

        In LangGraph this would be:
            graph.add_conditional_edges(
                "researcher",
                lambda state: "writer" if enough_data(state) else "researcher",
                {"writer": "writer", "researcher": "researcher"}
            )

        We implement the same logic: after research, the router evaluates
        whether findings are sufficient. If not, we loop back.
        """
        if state.iteration >= state.max_iterations:
            print(f"  [Orchestrator] Max iterations reached → forcing write")
            state.status = "writing"
            return state

        # Router evaluates findings
        state = router_agent(state)
        # router_agent sets status to "writing" or "researching"
        return state
