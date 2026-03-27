"""
LLM wrapper — thin abstraction over the OpenAI Chat Completions API.

Why a separate file?
  The agent logic shouldn't know or care which provider we use.
  Swap this file and the rest of the code still works.

Key design choice:
  We use the *chat completions* endpoint (messages format),
  not the legacy completions endpoint.  This is what BabyAGI 2o
  does and what modern agents should use.
"""

import os
from openai import OpenAI

# Initialize client once — reused across all calls.
# Reads OPENAI_API_KEY from environment automatically.
client = OpenAI()

# Default model — can override per call.
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def llm_call(prompt: str, model: str | None = None, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    Send a prompt to the LLM and return the response text.

    This is deliberately simple — one prompt in, one string out.
    No streaming, no tool calls, no structured output (yet).

    Compare to BabyAGI's openai_call() at babyagi.py:333-416:
      - BabyAGI handles 6 different error types with retry loops
      - BabyAGI supports GPT, Llama, and human input modes
      - We keep it minimal — just the OpenAI chat path
      - We let exceptions propagate (the caller can handle them)
    """
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
