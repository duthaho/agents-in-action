"""
Tools — async version of the Phase 2/3 Tool base class.

Differences from Phase 3:

    - Tool.execute is async. Tools can now use aiohttp, asyncio.sleep,
      run_in_executor, etc. without blocking the event loop.

    - Tool.execute raises ToolError on failure instead of returning
      a string like "Error: ...". The agent base class catches it and
      feeds the error back to the LLM as an observation — so the LLM
      is a first-class participant in recovery.

    - Tool has a `sensitive: bool` flag. In Step 6 the decorator will
      use it to gate execution behind the AsyncApprovalGate. Defined
      now (unused) so Step 6 is a pure additive change.

    - Tool.execute takes a RunContext as the first arg. In Step 6 the
      approval gate is read from ctx. Tool *functions* still take
      plain kwargs — the ctx plumbing stays on the Tool wrapper.

The `@tool` decorator supports both @tool and @tool(sensitive=True).
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, get_type_hints

from errors import ApprovalDenied, ToolError

PYTHON_TYPE_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Awaitable[Any]]
    sensitive: bool = False

    async def execute(self, ctx: Any, **kwargs) -> str:
        # Sensitive-flag enforcement lands in Step 6. For now the flag
        # is just stored. ApprovalDenied re-raise is pre-wired so the
        # agent base can catch it symmetrically with ToolError.
        try:
            result = await self.func(**kwargs)
        except ApprovalDenied:
            raise
        except Exception as e:
            raise ToolError(self.name, f"{type(e).__name__}: {e}") from e
        return str(result)

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(func: Callable[..., Awaitable[Any]] | None = None, *, sensitive: bool = False):
    """
    Decorator that turns a typed, documented async function into a Tool.

    Usage:

        @tool
        async def calculator(expression: str) -> str:
            '''Evaluate a math expression.'''
            ...

        @tool(sensitive=True)
        async def write_file(path: str, content: str) -> str:
            '''Write content to disk.'''
            ...
    """

    def wrap(f: Callable[..., Awaitable[Any]]) -> Tool:
        if not f.__doc__:
            raise ValueError(f"Tool '{f.__name__}' must have a docstring")
        if not inspect.iscoroutinefunction(f):
            raise ValueError(f"Tool '{f.__name__}' must be an async function")

        hints = get_type_hints(f)
        sig = inspect.signature(f)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            json_type = PYTHON_TYPE_TO_JSON.get(hints.get(pname, str), "string")
            properties[pname] = {"type": json_type}
            if p.default is inspect.Parameter.empty:
                required.append(pname)

        return Tool(
            name=f.__name__,
            description=f.__doc__.strip(),
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            func=f,
            sensitive=sensitive,
        )

    # Support both @tool and @tool(sensitive=True)
    if func is None:
        return wrap
    return wrap(func)


def get_tool_by_name(name: str, tools: list[Tool]) -> Tool | None:
    for t in tools:
        if t.name == name:
            return t
    return None


# ─── Built-in tools ────────────────────────────────────────────────

@tool
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns top 5 results with title, snippet, and URL. Use this to find current information and facts."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("ddgs package not installed. Run: pip install ddgs")

    # DDGS is sync — run in a thread so we don't block the event loop.
    def _run() -> list[dict]:
        return DDGS().text(query, max_results=5)

    results = await asyncio.to_thread(_run)
    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(
            f"**{r.get('title', 'No title')}**\n"
            f"{r.get('body', '')}\n"
            f"URL: {r.get('href', '')}"
        )
    return "\n\n".join(lines)


@tool
async def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports Python math syntax (no imports)."""
    import math
    allowed = {"__builtins__": {}}
    allowed.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
    return str(eval(expression, allowed))


# Tools available to the researcher agent. Step 6 appends write_file.
RESEARCH_TOOLS: list[Tool] = [web_search, calculator]
