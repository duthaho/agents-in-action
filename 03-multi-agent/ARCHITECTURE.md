# Phase 3 — Architecture: Multi-Agent Systems

## What changed from Phase 2

Phase 2 had **one agent with many tools**. It could search, read files,
and query a knowledge base — but it was a single brain doing everything.

Phase 3 introduces **multiple specialized agents** working together.
Each agent has its own role, tools, and prompt. An orchestrator
coordinates them.

## Two Orchestration Paradigms

### Paradigm A: Sequential Pipeline (CrewAI-style)

```
User request
    │
    ▼
┌──────────┐    ┌────────────┐    ┌──────────┐
│  ROUTER  │───►│ RESEARCHER │───►│  WRITER  │
│          │    │            │    │          │
│ Analyzes │    │ Searches   │    │ Formats  │
│ request, │    │ web, reads │    │ findings │
│ creates  │    │ docs,      │    │ into     │
│ research │    │ gathers    │    │ polished │
│ plan     │    │ facts      │    │ report   │
└──────────┘    └────────────┘    └──────────┘
    │                │                 │
    ▼                ▼                 ▼
 research         raw findings      final report
   plan                              (returned)
```

**How CrewAI does this (crew.py):**
- `Crew(agents=[...], tasks=[...], process=Process.sequential)`
- Tasks execute in order, each assigned to a specific agent
- Task outputs flow forward via `context` — each task sees prior results
- Simple, predictable, but can't loop back if results are insufficient

### Paradigm B: Graph-Based (LangGraph-style)

```
User request
    │
    ▼
┌──────────┐
│  ROUTER  │◄─────────────────────────┐
│          │                          │
│ Analyzes │    ┌────────────┐        │
│ & routes ├───►│ RESEARCHER │        │
│          │    │ gathers    │        │
└──────────┘    │ facts      │        │
                └─────┬──────┘        │
                      │               │
                      ▼               │
               ┌─────────────┐        │
               │   ROUTER    │        │
               │  evaluates: │        │
               │  enough     │────────┘
               │  data?      │   NO → loop back
               │             │
               └──────┬──────┘
                      │ YES
                      ▼
               ┌──────────┐
               │  WRITER  │
               │ produces  │
               │ report    │
               └──────┬───┘
                      │
                      ▼
                 final report
```

**How LangGraph does this (StateGraph):**
- Define nodes (agents) and edges (transitions)
- Conditional edges: routing function checks state, returns next node
- Shared state (TypedDict) flows through all nodes
- Supports cycles — router can send back to researcher if data is lacking

## Our Design: Both Patterns

We'll implement both so you can see the tradeoffs:

### Shared State

```python
@dataclass
class ResearchState:
    query: str               # Original user request
    research_plan: str       # Router's analysis
    findings: list[str]      # Researcher's gathered facts
    report: str              # Writer's final output
    status: str              # "planning" | "researching" | "writing" | "done"
    iteration: int           # Current loop count (for graph pattern)
```

### Agent Definitions

Each agent is a Phase 1-style ReAct agent with:
- A unique system prompt (its "role")
- A specific set of tools
- A function: `(state) → updated state`

| Agent | Role | Tools | Input | Output |
|-------|------|-------|-------|--------|
| Router | Analyze request, create research plan, evaluate completeness | none (reasoning only) | query | research_plan, or "done" signal |
| Researcher | Gather facts using web search and tools | web_search, calculator | research_plan | findings[] |
| Writer | Produce polished report from findings | none (writing only) | findings[] | report |

### Key Difference: Agent as Function vs Agent as Object

```
Phase 1-2 (single agent):
    agent = Agent(tools=[...])
    answer = agent.chat("question")   # stateless per-call

Phase 3 (multi-agent):
    def researcher(state: ResearchState) -> ResearchState:
        '''Agent as a function that transforms shared state.'''
        agent = Agent(tools=[web_search], system_prompt=RESEARCHER_PROMPT)
        result = agent.chat(state.research_plan)
        state.findings.append(result)
        return state
```

This is exactly what LangGraph does — each node is a function
`(state) → state`. CrewAI wraps this differently (Task + Agent objects),
but the core idea is the same.

## File Structure

```
builds/03-multi-agent/
├── ARCHITECTURE.md
├── state.py          # ResearchState dataclass — shared between agents
├── llm.py            # Same as Phase 1 (reused)
├── tools.py          # Web search + calculator (subset of Phase 2)
├── agents/
│   ├── __init__.py
│   ├── base.py       # BaseAgent — Phase 1 ReAct agent, reusable
│   ├── router.py     # Router agent — plans and evaluates
│   ├── researcher.py # Researcher agent — gathers facts
│   └── writer.py     # Writer agent — produces reports
├── orchestrator.py   # Both patterns: SequentialPipeline + GraphOrchestrator
└── main.py           # Entry point: pick pattern, run query
```

## Data Flow Comparison

```
SEQUENTIAL:
  router(state) → researcher(state) → writer(state) → done
  (3 steps, always)

GRAPH:
  router(state) → researcher(state) → router(state) → ...
                                          │
                                   enough data? ──YES──→ writer(state) → done
                                          │
                                          NO → researcher(state) → router...
  (variable steps, adapts to complexity)
```
