"""
BaseAgent — a reusable ReAct agent from Phase 1, adapted for multi-agent use.

The key change from Phase 1: agents here are stateless per-call.
They don't maintain conversation history across calls — each invocation
is a fresh conversation. The orchestrator manages state between agents.

Compare to CrewAI's Agent (agent/core.py):
  CrewAI Agent has: role, goal, backstory, tools, llm, max_iter
  Our BaseAgent has: system_prompt, tools, max_iterations
  Same concept — the system_prompt encodes role/goal/backstory.
"""

import json

from llm import llm_call
from tools import Tool, get_tool_by_name


class BaseAgent:
    def __init__(self, system_prompt: str, tools: list[Tool] | None = None, max_iterations: int = 10):
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.tool_schemas = [t.to_openai_schema() for t in self.tools] if self.tools else None

    def run(self, task: str) -> str:
        """
        Execute a task and return the result.

        This is a single-shot ReAct loop (Phase 1 pattern):
          1. Send system prompt + task + tool defs to LLM
          2. If LLM returns tool_calls → execute, feed back, repeat
          3. If LLM returns text → that's the answer

        Compare to CrewAI's execute_task() in agent/core.py:664
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        for iteration in range(self.max_iterations):
            response = llm_call(messages=messages, tools=self.tool_schemas)

            if response.tool_calls:
                # Add assistant message with tool calls
                msg = {"role": "assistant"}
                if response.content:
                    msg["content"] = response.content
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in response.tool_calls
                ]
                messages.append(msg)

                # Execute each tool call
                for tc in response.tool_calls:
                    tool = get_tool_by_name(tc.function.name, self.tools)
                    if tool:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        result = tool.execute(**args)
                    else:
                        result = f"Error: Unknown tool '{tc.function.name}'"

                    print(f"    [{self.__class__.__name__}] Tool: {tc.function.name} → {result[:150]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": result,
                    })
                continue

            # No tool calls — final answer
            return response.content or "(no response)"

        return "(max iterations reached)"
