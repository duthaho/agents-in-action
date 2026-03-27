"""
RAG tools — let the agent search the knowledge base.

Key insight: RAG is just a tool.
The agent decides WHEN to search its knowledge base, just like
it decides when to use the calculator. This is more flexible than
always injecting RAG context into every prompt.

We expose two tools:
  - search_knowledge: search the knowledge base for relevant info
  - ingest_document: add new text to the knowledge base
"""

from .base import Tool

# We can't use the @tool decorator here because the RAG tools
# need a reference to the RAGEngine instance. Instead, we create
# Tool objects manually with closures.


def create_rag_tools(rag_engine) -> list[Tool]:
    """
    Create RAG tools bound to a specific RAGEngine instance.

    This is a factory pattern — we can't use @tool because these tools
    need to capture the rag_engine in a closure.
    """

    def search_knowledge(query: str) -> str:
        """Search the knowledge base for relevant information. Use this when you need to find specific facts, context, or details from ingested documents."""
        results = rag_engine.search(query, top_k=5)
        if not results:
            return "No relevant information found in the knowledge base."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] (score: {r['score']:.2f}, source: {r['source']})\n{r['text']}"
            )
        return "\n\n".join(formatted)

    def ingest_document(text: str, source: str) -> str:
        """Add a document to the knowledge base. The text will be chunked, embedded, and stored for future retrieval. Provide a source label to identify where this text came from."""
        count = rag_engine.ingest(text, source=source)
        stats = rag_engine.get_stats()
        return f"Ingested {count} chunks from '{source}'. Knowledge base now has {stats['total_chunks']} total chunks."

    search_tool = Tool(
        name="search_knowledge",
        description="Search the knowledge base for relevant information. Use this when you need to find specific facts, context, or details from ingested documents.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        func=search_knowledge,
    )

    ingest_tool = Tool(
        name="ingest_document",
        description="Add a document to the knowledge base. The text will be chunked, embedded, and stored for future retrieval.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["text", "source"],
        },
        func=ingest_document,
    )

    return [search_tool, ingest_tool]
