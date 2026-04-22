"""
BaseAgent — the async ReAct loop shared by router, researcher, writer.

This is the core "reason + act" loop, Phase 1 style, now async:

    1. Send the system prompt + task to the LLM.
    2. If the response contains tool_calls, execute each one in
       order, append the results as tool messages, and loop.
    3. If the response is plain text, that's the answer — return it.

What this file knows about production features:

    - It publishes ToolCallStarted/Completed/Failed around every
      tool invocation. (LLMCall events are published by llm.py.)

    - It catches ToolError AND ApprovalDenied and feeds them back
      to the LLM as tool-result observations. This is the
      "self-healing" behavior: the LLM sees "ERROR: ..." or
      "DENIED: ..." and chooses the next action.

    - It sets ctx.current_agent before each tool call so the
      approval gate (Step 6) and downstream events know who is
      asking.

What this file does NOT know about:

    - Retries (handled by the wrapper around llm_call in Step 4).
    - Checkpointing (handled by the orchestrator via StateTransition
      events — agents are stateless per-call).
    - Cost tracking (emitted from llm.py).

Step 7 (streaming) adds an alternate path that uses llm_stream when
ctx.stream is True. The rest of this loop stays the same.
"""
from __future__ import annotations

import json
from typing import Any

from context import RunContext
from errors import ApprovalDenied, ToolError
from events import ToolCallCompleted, ToolCallFailed, ToolCallStarted, now
from llm import llm_call
from tools import Tool, get_tool_by_name


class BaseAgent:
    # Override in subclasses so events/logs carry a useful agent name.
    name: str = "base"

    def __init__(
        self,
        system_prompt: str,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        model: str | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.model = model
        self.tool_schemas = (
            [t.to_openai_schema() for t in self.tools] if self.tools else None
        )

    async def run(self, ctx: RunContext, task: str) -> str:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        for _ in range(self.max_iterations):
            message = await llm_call(
                ctx,
                messages=messages,
                tools=self.tool_schemas,
                model=self.model,
                agent_name=self.name,
            )

            tool_calls = message.tool_calls or []

            if tool_calls:
                # Record the assistant turn that issued the tool calls.
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if message.content:
                    assistant_msg["content"] = message.content
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_msg)

                # Execute each tool call and feed results back.
                for tc in tool_calls:
                    result = await self._invoke_tool(
                        ctx, tc.function.name, tc.function.arguments
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": result,
                    })
                continue

            # No tool calls — plain text is the final answer.
            return message.content or "(no response)"

        return "(max iterations reached)"

    async def _invoke_tool(self, ctx: RunContext, name: str, raw_args: str) -> str:
        """
        Run one tool call, publishing lifecycle events and converting
        ToolError / ApprovalDenied into observations for the LLM.
        """
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}

        tool_obj = get_tool_by_name(name, self.tools)
        if tool_obj is None:
            msg = f"Unknown tool '{name}'"
            await ctx.bus.publish(ToolCallFailed(
                run_id=ctx.run_id, timestamp=now(),
                agent=self.name, tool=name, error=msg,
            ))
            return f"ERROR: {msg}"

        ctx.current_agent = self.name
        await ctx.bus.publish(ToolCallStarted(
            run_id=ctx.run_id, timestamp=now(),
            agent=self.name, tool=name, args=args,
        ))

        try:
            result = await tool_obj.execute(ctx, **args)
            await ctx.bus.publish(ToolCallCompleted(
                run_id=ctx.run_id, timestamp=now(),
                agent=self.name, tool=name, result_preview=result[:200],
            ))
            return result

        except ApprovalDenied as e:
            await ctx.bus.publish(ToolCallFailed(
                run_id=ctx.run_id, timestamp=now(),
                agent=self.name, tool=name, error=f"denied: {e}",
            ))
            return f"DENIED: {e}. Choose a different approach."

        except ToolError as e:
            await ctx.bus.publish(ToolCallFailed(
                run_id=ctx.run_id, timestamp=now(),
                agent=self.name, tool=name, error=str(e),
            ))
            return f"ERROR: {e}"
