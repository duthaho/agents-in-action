"""
RAG Engine — chunk, embed, store, retrieve.

This is the equivalent of CrewAI's Knowledge + KnowledgeStorage system,
simplified to the core pipeline.

CrewAI's approach (knowledge/source/base_knowledge_source.py):
  - chunk_size=4000, chunk_overlap=200
  - Supports PDF, CSV, Excel, JSON, Text
  - ChromaDB storage with configurable embedders

Our approach:
  - chunk_size=1000, chunk_overlap=200 (smaller = more focused retrieval)
  - Plain text only (extensible later)
  - ChromaDB + OpenAI embeddings (text-embedding-3-small)

The pipeline:
  INGEST:   text → chunk → embed → store in ChromaDB
  RETRIEVE: query → embed → cosine similarity search → top-k chunks
"""

import os
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ─── Configuration ────────────────────────────────────────────────

CHUNK_SIZE = 1000       # Characters per chunk (CrewAI uses 4000)
CHUNK_OVERLAP = 200     # Overlap between chunks (same as CrewAI)
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cheap
COLLECTION_NAME = "knowledge"


class RAGEngine:
    """
    Simple RAG engine backed by ChromaDB.

    Compare to CrewAI's KnowledgeStorage (knowledge/storage/knowledge_storage.py):
      CrewAI: Async support, metadata filters, score thresholds, multiple backends
      Us:     Sync only, basic search, ChromaDB only
    """

    def __init__(self, persist_dir: str = "./chroma_data"):
        # ChromaDB client — persists to disk so knowledge survives restarts.
        # BabyAGI also used ChromaDB (babyagi.py:190-210).
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Embedding function — ChromaDB calls this automatically on add/query.
        # This is a key simplification: we don't manage embeddings ourselves,
        # ChromaDB handles it via this wrapper.
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=EMBEDDING_MODEL,
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def ingest(self, text: str, source: str = "unknown") -> int:
        """
        Ingest a document: chunk → embed → store.

        Args:
            text: The document text to ingest.
            source: Label for where this text came from (filename, URL, etc.)

        Returns:
            Number of chunks stored.

        Compare to CrewAI's BaseKnowledgeSource._save_documents():
          CrewAI chunks, then calls storage.save() which handles embedding.
          We do the same — ChromaDB handles embedding via the embedding_function.
        """
        chunks = self._chunk_text(text)

        if not chunks:
            return 0

        # Generate unique IDs for each chunk
        existing_count = self.collection.count()
        ids = [f"{source}_{existing_count + i}" for i in range(len(chunks))]

        # Store in ChromaDB — embedding happens automatically
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for relevant chunks.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of {"text": str, "source": str, "score": float}

        Compare to CrewAI's KnowledgeStorage.search():
          CrewAI: Supports metadata filters, score thresholds, async.
          Us:     Basic top-k search, no filters.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                # ChromaDB returns distances (lower = more similar for cosine)
                "score": 1 - results["distances"][0][i],
            })

        return output

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Compare to CrewAI's BaseKnowledgeSource._chunk_text():
          CrewAI: text[i : i + chunk_size] for i in range(0, len, chunk_size - overlap)
          Us:     Same algorithm, different parameters.

        The overlap ensures that content at chunk boundaries isn't lost.
        If a sentence spans two chunks, the overlap captures it in both.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def get_stats(self) -> dict:
        """Return stats about the knowledge base."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": COLLECTION_NAME,
        }
