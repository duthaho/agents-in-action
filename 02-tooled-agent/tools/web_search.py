"""
Web search tool — uses DuckDuckGo (no API key needed).

Requires: pip install ddgs

This gives the agent access to current information beyond
its training data. The agent decides when to search — it's
just another tool in the registry.
"""

from .base import tool


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns top 5 results with title, snippet, and URL. Use this to find current information, facts, or documentation."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs package not installed. Run: pip install ddgs"

    results = DDGS().text(query, max_results=5)

    if not results:
        return f"No results found for: {query}"

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}. {r['title']}\n"
            f"   {r['body']}\n"
            f"   URL: {r['href']}"
        )
    return "\n\n".join(formatted)
