"""
Interactive REPL with document ingestion support.

Usage:
    export OPENAI_API_KEY="sk-..."
    pip install duckduckgo-search chromadb
    python main.py

Commands:
    /ingest <file_path>  — Ingest a file into the knowledge base
    /stats               — Show knowledge base stats
    /tools               — List available tools
    quit                 — Exit

Try:
    > /ingest ../../PLAN.md
    > What phases are in the learning plan?
    > Search the web for latest news about AI agents
    > What is 2**10 + 3**7?
    > Read the file ./agent.py and explain the main loop
"""

import os
from agent import Agent
from rag import RAGEngine
from tools import build_all_tools


def main():
    # Initialize RAG engine
    rag = RAGEngine(persist_dir="./chroma_data")

    # Build all tools including RAG-powered ones
    tools = build_all_tools(rag_engine=rag)

    # Create agent with full tool set
    agent = Agent(tools=tools, max_iterations=15)

    print("=" * 60)
    print("  Tooled Agent with RAG")
    print("  Commands: /ingest <file>, /stats, /tools, quit")
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

        # ── Slash commands ──────────────────────────────
        if user_input.startswith("/ingest "):
            file_path = user_input[8:].strip()
            _ingest_file(rag, file_path)
            continue

        if user_input == "/stats":
            stats = rag.get_stats()
            print(f"\nKnowledge base: {stats['total_chunks']} chunks stored\n")
            continue

        if user_input == "/tools":
            print("\nAvailable tools:")
            for t in agent.tools:
                print(f"  - {t.name}: {t.description[:80]}...")
            print()
            continue

        # ── Normal chat ─────────────────────────────────
        print()
        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


def _ingest_file(rag: RAGEngine, file_path: str):
    """Ingest a file into the knowledge base."""
    if not os.path.exists(file_path):
        print(f"\n  Error: File not found: {file_path}\n")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"\n  Error reading file: {e}\n")
        return

    source = os.path.basename(file_path)
    count = rag.ingest(text, source=source)
    stats = rag.get_stats()
    print(f"\n  Ingested {count} chunks from '{source}'")
    print(f"  Knowledge base now has {stats['total_chunks']} total chunks\n")


if __name__ == "__main__":
    main()
