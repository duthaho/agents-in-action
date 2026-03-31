"""
Tools for Phase 3 — web search + calculator.

Reuses the @tool pattern from Phase 1/2.
Only the researcher agent needs tools; router and writer are text-only.
"""

import inspect
import json
from typing import Any, Callable, get_type_hints

# ─── Type mapping ───
PYTHON_TYPE_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


class Tool:
    def __init__(self, name: str, description: str, parameters: dict, func: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def execute(self, **kwargs) -> str:
        try:
            return str(self.func(**kwargs))
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(func: Callable) -> Tool:
    """Decorator that turns a typed, documented function into a Tool."""
    if not func.__doc__:
        raise ValueError(f"Tool '{func.__name__}' must have a docstring")

    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        json_type = PYTHON_TYPE_TO_JSON.get(hints.get(name, str), "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return Tool(
        name=func.__name__,
        description=func.__doc__.strip(),
        parameters={"type": "object", "properties": properties, "required": required},
        func=func,
    )


def get_tool_by_name(name: str, tools: list[Tool]) -> Tool | None:
    for t in tools:
        if t.name == name:
            return t
    return None


# ─── Built-in Tools ───

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns top 5 results with title, snippet, and URL. Use this to find current information and facts."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs package not installed. Run: pip install ddgs"

    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."

    output = []
    for r in results:
        output.append(f"**{r.get('title', 'No title')}**\n{r.get('body', '')}\nURL: {r.get('href', '')}")
    return "\n\n".join(output)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports Python math syntax."""
    import math
    allowed = {"__builtins__": {}}
    allowed.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
    return str(eval(expression, allowed))


RESEARCH_TOOLS: list[Tool] = [web_search, calculator]
