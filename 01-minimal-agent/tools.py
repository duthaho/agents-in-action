"""
Tool system — @tool decorator + built-in tools.

Inspired by CrewAI's tool system (base_tool.py:542-621).
CrewAI uses Pydantic models to generate JSON schemas from type hints.
We do the same thing but with plain Python — no Pydantic needed.

The key insight: a "tool" is just:
  1. A function the agent can call
  2. A JSON schema that tells the LLM what arguments it expects
  3. A description that helps the LLM decide WHEN to use it

The @tool decorator auto-generates #2 and #3 from the function's
type annotations and docstring — same pattern as CrewAI.
"""

import inspect
import json
from datetime import datetime
from typing import Any, Callable, get_type_hints

# ─── Type mapping: Python types → JSON Schema types ───
# The LLM needs JSON Schema to understand parameter types.
PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    """
    A tool the agent can call.

    Compare to CrewAI's BaseTool (base_tool.py:57):
      CrewAI: Pydantic BaseModel with name, description, args_schema, _run(), _arun()
      Us:     Plain class with name, description, parameters, func

    We skip: async support, caching, usage counting, LangChain adapters.
    Those are production concerns — Phase 4 material.
    """

    def __init__(self, name: str, description: str, parameters: dict, func: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema dict
        self.func = func

    def execute(self, **kwargs) -> str:
        """Run the tool and return result as a string."""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            # Return errors as strings — let the LLM decide how to handle.
            # This is a key agent pattern: tools should never crash the loop.
            return f"Error: {type(e).__name__}: {e}"

    def to_openai_schema(self) -> dict:
        """
        Convert to OpenAI's tool format for the API.

        This is the format OpenAI expects in the `tools` parameter:
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "...",
                "parameters": { JSON Schema }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(func: Callable) -> Tool:
    """
    Decorator that turns a function into a Tool.

    Compare to CrewAI's @tool (base_tool.py:542-621):
      CrewAI: Supports @tool, @tool("name"), @tool(result_as_answer=True)
              Uses Pydantic create_model() for schema generation.
      Us:     Only supports @tool (no arguments).
              Uses inspect + get_type_hints() for schema generation.

    Requirements (same as CrewAI):
      - Function MUST have a docstring → becomes the tool description
      - Function MUST have type annotations → becomes the parameter schema
      - Function MUST return something → converted to string for the LLM

    Example:
        @tool
        def calculator(expression: str) -> str:
            '''Evaluate a mathematical expression.'''
            return str(eval(expression))

        # Auto-generates:
        # Tool(name="calculator",
        #      description="Evaluate a mathematical expression.",
        #      parameters={"type": "object", "properties": {"expression": {"type": "string"}}, ...})
    """
    if not func.__doc__:
        raise ValueError(f"Tool function '{func.__name__}' must have a docstring")

    # Get type hints for parameters
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    # Build JSON Schema from type annotations
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)
        json_type = PYTHON_TYPE_TO_JSON.get(param_type, "string")

        properties[param_name] = {"type": json_type}

        # If no default value, it's required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return Tool(
        name=func.__name__,
        description=func.__doc__.strip(),
        parameters=parameters_schema,
        func=func,
    )


# ─── Built-in Tools ───────────────────────────────────────────────


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '47 * 89 + 12'. Supports Python math syntax including +, -, *, /, **, %, and parentheses."""
    # Using eval with restricted builtins for safety.
    # In production, use a proper math parser — eval is dangerous.
    allowed_names = {"__builtins__": {}}
    import math
    allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
    return str(eval(expression, allowed_names))


@tool
def get_current_time() -> str:
    """Get the current date and time. Returns the current datetime in ISO format."""
    return datetime.now().isoformat()


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output. The code runs in an isolated namespace. Print statements will be captured. Use this for computations, data processing, or any Python task."""
    import io
    import contextlib

    output = io.StringIO()
    namespace: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(output):
            exec(code, namespace)
        result = output.getvalue()
        return result if result else "(code executed successfully, no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ─── Tool Registry ────────────────────────────────────────────────


# Default set of tools — the agent picks from these.
DEFAULT_TOOLS: list[Tool] = [calculator, get_current_time, python_repl]


def get_tool_by_name(name: str, tools: list[Tool] | None = None) -> Tool | None:
    """Look up a tool by name."""
    for t in (tools or DEFAULT_TOOLS):
        if t.name == name:
            return t
    return None
