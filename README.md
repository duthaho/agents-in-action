# agents-in-action

> A build-along curriculum for understanding AI agents from first principles.

This repo is a sequence of small, runnable Python programs that each build a
slightly more capable agent. It's aimed at developers who already know Python
and want to understand what's actually happening inside frameworks like
LangGraph and CrewAI — by writing the loop, the tools, the orchestration, and
the production patterns themselves, in plain Python.

## The learning arc

| Phase | What you'll build | What you'll learn |
|-------|-------------------|-------------------|
| [00 — Understand the loop](./00-understand-loop/) | A minimal BabyAGI-style task-decomposition loop | The core insight: an agent is just a loop calling an LLM with structured prompts |
| [01 — Minimal agent](./01-minimal-agent/) | A ReAct agent with tools and memory | Tool calls, observation feedback, the think–act–observe cycle |
| [02 — Tooled agent](./02-tooled-agent/) | The same agent with a richer toolbox and RAG | Tool abstraction, retrieval-augmented generation, file and web tools |
| [03 — Multi-agent](./03-multi-agent/) | Router → researcher → writer pipeline sharing typed state | Orchestration patterns, shared state, agent-to-agent handoff |
| [04 — Production](./04-production/) | The phase-03 pipeline rebuilt with observability, retries, checkpoints, HITL approval, cost tracking, streaming | What separates a demo from a production agent: event bus, async-first, six concrete patterns |

Each phase has its own `ARCHITECTURE.md` that walks through the design.

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## How to run a phase

Each phase is self-contained. Pick one, create a virtualenv inside it, and
run it.

```bash
cd 00-understand-loop                                 # or any other phase
python -m venv .venv
source .venv/bin/activate                             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env                               # then fill in OPENAI_API_KEY
python main.py
```

Phase 02 also stores a small ChromaDB index under `chroma_data/` the first
time it runs (gitignored). Phase 04 writes JSONL logs and SQLite checkpoints
under `runs/` (also gitignored).

## What this is intentionally not

This is a **learning repo**, not a framework. The code is written for
readability, not for reuse across projects. For real applications, reach for
[LangGraph](https://github.com/langchain-ai/langgraph),
[CrewAI](https://github.com/crewAIInc/crewAI), or another production-grade
framework. The point of this repo is to understand what those frameworks are
doing under the hood — so that when you pick one, you know what trade-offs
you're accepting.

## Acknowledgements

The progression here is heavily influenced by:

- [BabyAGI](https://github.com/yoheinakajima/babyagi) — the original
  task-decomposition agent loop that phase 00 distills.
- [LangGraph](https://github.com/langchain-ai/langgraph) — for the
  state-graph framing that shapes phases 03 and 04.
- [CrewAI](https://github.com/crewAIInc/crewAI) — for the
  multi-agent role/task framing that shapes phase 03.

## License

[MIT](./LICENSE) © 2026 duthaho
