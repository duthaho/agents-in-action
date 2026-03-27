"""
Interactive REPL — chat with the ReAct agent.

Usage:
    export OPENAI_API_KEY="sk-..."
    python main.py

    # Override model:
    LLM_MODEL=gpt-4o python main.py

Try these to test tool use:
    > What is 47 * 89 + 12?              (uses calculator)
    > What time is it?                    (uses get_current_time)
    > Write a Python function that checks if a number is prime, then test it with 17
                                          (uses python_repl)
    > Calculate the compound interest on $10000 at 5% for 10 years
                                          (uses calculator, may chain calls)
"""

from agent import Agent
from tools import DEFAULT_TOOLS


def main():
    agent = Agent(tools=DEFAULT_TOOLS, max_iterations=10)

    print("=" * 60)
    print("  ReAct Agent with Tools")
    print("  Type 'quit' to exit, 'tools' to list available tools")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "tools":
            print("\nAvailable tools:")
            for t in agent.tools:
                print(f"  - {t.name}: {t.description}")
            print()
            continue

        print()  # blank line before response
        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
