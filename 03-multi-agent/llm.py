"""
LLM wrapper — reused from Phase 1.
See builds/01-minimal-agent/llm.py for detailed comments.
"""

import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

client = OpenAI()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def llm_call(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0.7,
) -> ChatCompletionMessage:
    """Send messages to the LLM and return the response message."""
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message
