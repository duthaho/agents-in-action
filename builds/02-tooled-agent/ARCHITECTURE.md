# Phase 2 — Architecture: Tool Use and RAG

## What changed from Phase 1

Phase 1 had 3 hardcoded tools (calculator, time, python_repl).
Phase 2 adds:
- **File tools** — read/write/list files on disk
- **Web search** — search the internet via DuckDuckGo
- **RAG tool** — ingest documents, embed them, retrieve relevant chunks
- **Semantic memory** — remember and recall past interactions

## How CrewAI does Knowledge (what we learned)

CrewAI's knowledge system:
```
Document → chunk (4000 chars, 200 overlap) → embed (OpenAI) → store (ChromaDB)
Query → embed → cosine similarity search → top-k results
```

CrewAI's memory system is much more complex:
- LLM-driven consolidation, adaptive recall depth, importance scoring
- Uses LanceDB, background writes, scope hierarchies

**We simplify dramatically:** same chunk→embed→store→retrieve pipeline,
but without LLM analysis, consolidation, or fancy scoring.

## Architecture

```
builds/02-tooled-agent/
├── tools/
│   ├── __init__.py       # Exports all tools + registry
│   ├── base.py           # Tool class + @tool decorator (from Phase 1)
│   ├── core_tools.py     # calculator, time, python_repl (from Phase 1)
│   ├── file_tools.py     # read_file, write_file, list_directory
│   ├── web_search.py     # DuckDuckGo search
│   └── rag_tool.py       # ingest docs + search knowledge base
├── rag.py                # RAG engine: chunk → embed → store → retrieve
├── memory.py             # Conversation memory + semantic memory layer
├── llm.py                # Same as Phase 1
├── agent.py              # ReAct agent (mostly same as Phase 1)
└── main.py               # REPL with /ingest command
```

## RAG Pipeline

```
                    INGESTION (one-time per document)
                    ─────────────────────────────────

    Document (text/file)
         │
         ▼
    ┌──────────┐
    │  CHUNK   │  Split into overlapping segments
    │          │  chunk_size=1000, overlap=200
    └────┬─────┘
         │  ["chunk 1...", "chunk 2...", ...]
         ▼
    ┌──────────┐
    │  EMBED   │  Convert text → vectors via OpenAI API
    │          │  model: text-embedding-3-small (1536 dims)
    └────┬─────┘
         │  [[0.02, -0.15, ...], [0.08, 0.31, ...], ...]
         ▼
    ┌──────────┐
    │  STORE   │  Save in ChromaDB collection
    │          │  vectors + original text + metadata
    └──────────┘


                    RETRIEVAL (every query)
                    ──────────────────────

    User query
         │
         ▼
    ┌──────────┐
    │  EMBED   │  Same embedding model as ingestion
    └────┬─────┘
         │  [0.05, -0.22, ...]
         ▼
    ┌──────────┐
    │  SEARCH  │  Cosine similarity in ChromaDB
    │          │  Return top-k most similar chunks
    └────┬─────┘
         │  ["relevant chunk 1", "relevant chunk 2", ...]
         ▼
    Inject as context into LLM prompt
```

Compare to CrewAI:
- CrewAI uses 4000 char chunks. We use 1000 (better for focused retrieval).
- CrewAI supports PDF, CSV, Excel, JSON. We support plain text (extensible later).
- Both use ChromaDB + OpenAI embeddings.

## Memory Tiers

```
┌─────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM                     │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  WORKING MEMORY (conversation)               │    │
│  │  = message history sent to LLM               │    │
│  │  Sliding window: last N messages              │    │
│  │  Same as Phase 1                              │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  SEMANTIC MEMORY (knowledge base)    [NEW]   │    │
│  │  = ChromaDB vector store                      │    │
│  │  Ingested documents → chunked → embedded      │    │
│  │  Retrieved via RAG tool when agent needs info │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

CrewAI has a third tier (episodic/long-term memory with importance
scoring and recency decay). We skip that — it adds complexity but
the learning value is mainly in the RAG pipeline.

## Web Search Design

We use `duckduckgo-search` (Python package) — no API key needed.
The agent decides when to search via the `web_search` tool.

```
Agent: "I need to find current info about X"
  → calls web_search(query="X")
    → DuckDuckGo API → top 5 results (title + snippet + URL)
  → Agent reads results, decides if it needs more info
  → May call web_search again with refined query
```
