"""
Tool registry — collects all tools and provides lookup.

Phase 1 had a flat list. Phase 2 organizes tools into a package
with a central registry. The agent gets all tools from here.
"""

from .base import Tool, tool
from .core_tools import calculator, get_current_time, python_repl
from .file_tools import read_file, write_file, list_directory
from .web_search import web_search
from .rag_tool import create_rag_tools

# Static tools (don't need external dependencies at init time)
STATIC_TOOLS: list[Tool] = [
    calculator,
    get_current_time,
    python_repl,
    read_file,
    write_file,
    list_directory,
    web_search,
]


def build_all_tools(rag_engine=None) -> list[Tool]:
    """Build the full tool list, including RAG tools if engine is provided."""
    tools = list(STATIC_TOOLS)
    if rag_engine:
        tools.extend(create_rag_tools(rag_engine))
    return tools


def get_tool_by_name(name: str, tools: list[Tool]) -> Tool | None:
    for t in tools:
        if t.name == name:
            return t
    return None
