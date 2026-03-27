"""Core tools from Phase 1 — calculator, time, python_repl."""

import io
import math
import contextlib
from datetime import datetime
from typing import Any

from .base import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '47 * 89 + 12'. Supports Python math syntax including +, -, *, /, **, %, parentheses, and math functions like sqrt, sin, cos, pi."""
    allowed_names = {"__builtins__": {}}
    allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
    return str(eval(expression, allowed_names))


@tool
def get_current_time() -> str:
    """Get the current date and time in ISO format."""
    return datetime.now().isoformat()


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output. Print statements will be captured. Use for computations, data processing, or any Python task."""
    output = io.StringIO()
    namespace: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(output):
            exec(code, namespace)
        result = output.getvalue()
        return result if result else "(code executed successfully, no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
