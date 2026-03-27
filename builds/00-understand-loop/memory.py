"""
Memory — stores past task results for context retrieval.

BabyAGI used ChromaDB (a vector database) for memory:
  - Every result was embedded into a vector (via OpenAI embeddings API)
  - Retrieval used cosine similarity to find the most relevant past results
  - This is a primitive RAG (Retrieval-Augmented Generation) system
  - See DefaultResultsStorage at babyagi.py:190-246

BabyAGI 2o used NO external memory:
  - The entire conversation history (messages list) was sent to the LLM
  - Simple but expensive — token cost grows with every iteration
  - Limited by the LLM's context window

Our approach: simple list with last-N retrieval.
  - Store everything in a Python list
  - Retrieve the N most recent results (not semantic — just recency)
  - This is intentionally naive — you'll feel the limitations,
    which motivates upgrading to RAG in Phase 2.

Trade-offs:
  ┌─────────────┬──────────────────┬───────────────┬──────────────┐
  │             │ BabyAGI (vector) │ BabyAGI 2o    │ Ours (list)  │
  ├─────────────┼──────────────────┼───────────────┼──────────────┤
  │ Retrieval   │ By relevance     │ Full history  │ By recency   │
  │ Cost        │ Embedding API    │ Token growth  │ Free         │
  │ Scalability │ Good             │ Poor          │ Moderate     │
  │ Persistence │ On disk          │ None          │ None         │
  └─────────────┴──────────────────┴───────────────┴──────────────┘
"""


class Memory:
    def __init__(self, max_items: int = 20):
        self.entries: list[dict] = []
        self.max_items = max_items

    def add(self, task: str, result: str):
        """Store a completed task and its result."""
        self.entries.append({"task": task, "result": result})
        # Keep memory bounded — drop oldest if we exceed max
        if len(self.entries) > self.max_items:
            self.entries = self.entries[-self.max_items:]

    def get_recent(self, n: int = 5) -> list[dict]:
        """
        Return the N most recent entries.

        Compare to BabyAGI's context_agent() at babyagi.py:521-536:
          BabyAGI: results_storage.query(query=objective, top_results_num=5)
          → searches by semantic similarity to the objective

          Us: just return the last N items
          → simple but misses relevant older results

        This is the limitation you'll feel:
          If task 1 produced important context but task 8 is running,
          we might have already dropped task 1's result from our window.
          A vector DB wouldn't have this problem.
        """
        return self.entries[-n:]

    def get_all(self) -> list[dict]:
        """Return all entries (for debugging/display)."""
        return list(self.entries)

    def __len__(self):
        return len(self.entries)
