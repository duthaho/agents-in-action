"""
Entry point — set an objective and watch the agent work.

Usage:
    # Set your OpenAI API key first:
    export OPENAI_API_KEY="sk-..."

    # Run with default objective:
    python main.py

    # Run with custom objective:
    python main.py "Research the pros and cons of microservices vs monoliths"

    # Override model:
    LLM_MODEL=gpt-4o python main.py "Design a REST API for a todo app"

Compare to BabyAGI:
    BabyAGI read OBJECTIVE and INITIAL_TASK from .env (babyagi.py:45-46)
    We accept it as a command-line argument (more convenient for experimentation)
"""

import sys
from agent import Agent

# Default objective for demonstration
DEFAULT_OBJECTIVE = (
    "Research and create a brief report on how AI agents work, "
    "covering: what an agent loop is, how memory helps agents, "
    "and the difference between single-agent and multi-agent systems."
)


def main():
    # Get objective from command line or use default
    if len(sys.argv) > 1:
        objective = " ".join(sys.argv[1:])
    else:
        objective = DEFAULT_OBJECTIVE

    # Create and run the agent
    # max_iterations=10 is our safety net — BabyAGI had no limit!
    agent = Agent(objective=objective, max_iterations=10)
    agent.run()

    # Print the full trace for learning purposes
    print("\n" + "=" * 60)
    print("  FULL EXECUTION TRACE")
    print("=" * 60)
    for i, entry in enumerate(agent.memory.get_all(), 1):
        print(f"\n--- Step {i}: {entry['task']} ---")
        print(entry["result"][:500])
        if len(entry["result"]) > 500:
            print("... (truncated)")
    print()


if __name__ == "__main__":
    main()
