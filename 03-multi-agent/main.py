"""
Entry point — run the multi-agent research system.

Usage:
    export OPENAI_API_KEY="sk-..."

    # Sequential pipeline (CrewAI-style):
    python main.py sequential "What are the latest advances in quantum computing?"

    # Graph orchestrator (LangGraph-style):
    python main.py graph "Compare React vs Vue vs Svelte in 2025"

    # Default (graph):
    python main.py "What are the pros and cons of microservices?"

Try different queries to see how the agents collaborate:
    - Factual: "What is the current state of fusion energy research?"
    - Comparative: "Compare Rust vs Go for backend development"
    - Analytical: "What caused the 2024 CrowdStrike outage and what were the lessons?"
"""

import sys
from orchestrator import SequentialPipeline, GraphOrchestrator

DEFAULT_QUERY = "What are the latest trends in AI agents and agentic frameworks in 2025?"


def main():
    # Parse arguments
    args = sys.argv[1:]

    # Determine orchestration pattern
    if args and args[0] in ("sequential", "graph"):
        pattern = args[0]
        query = " ".join(args[1:]) or DEFAULT_QUERY
    else:
        pattern = "graph"  # default to the more interesting pattern
        query = " ".join(args) or DEFAULT_QUERY

    # Run
    if pattern == "sequential":
        orchestrator = SequentialPipeline()
    else:
        orchestrator = GraphOrchestrator()

    state = orchestrator.run(query)

    # Print the final report
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(state.report)
    print("\n" + "=" * 60)
    print(f"  Stats: {state.iteration} research iteration(s), "
          f"{len(state.findings)} finding(s), "
          f"{len(state.report)} chars in report")
    print("=" * 60)


if __name__ == "__main__":
    main()
