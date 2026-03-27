"""
Tool base class + @tool decorator.
Carried forward from Phase 1 — same code, now in a package.
"""

import inspect
from typing import Any, Callable, get_type_hints

PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    def __init__(self, name: str, description: str, parameters: dict, func: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def execute(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return str(result)
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
        raise ValueError(f"Tool function '{func.__name__}' must have a docstring")

    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)
        json_type = PYTHON_TYPE_TO_JSON.get(param_type, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return Tool(
        name=func.__name__,
        description=func.__doc__.strip(),
        parameters={"type": "object", "properties": properties, "required": required},
        func=func,
    )
