"""
LLM wrapper — now with native function calling support.

Phase 0 llm.py was "string in, string out".
Phase 1 upgrades to "messages + tools in, ChatCompletionMessage out".

The key change: instead of returning just the text, we return the full
message object so the agent can inspect tool_calls.

Compare to CrewAI's get_llm_response() in crew_agent_executor.py:355-365:
  CrewAI passes messages, callbacks, response_model, and verbose flags.
  We keep it simple — just messages and tools.
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
    """
    Send messages to the LLM and return the response message.

    Phase 0 difference:
      Phase 0: prompt (str) → response text (str)
      Phase 1: messages (list) + tools (list) → ChatCompletionMessage

    Why return the full message object?
      The message might contain either:
        - content (text response, i.e., final answer)
        - tool_calls (list of tools the LLM wants to invoke)
      The agent needs to inspect which one to decide what to do next.

    The tools parameter uses OpenAI's format:
      [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    # Only pass tools if we have them — some calls (like summarization) don't need tools
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message
