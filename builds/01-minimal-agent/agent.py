"""
ReAct Agent — the core loop with tool execution.

This is the Phase 1 equivalent of CrewAI's CrewAgentExecutor._invoke_loop_native_tools()
(crew_agent_executor.py:305-326).

The loop:
    1. Send messages + tool definitions to the LLM
    2. If LLM returns tool_calls → execute each tool, add results to memory, go to 1
    3. If LLM returns content (no tool_calls) → that's the final answer, stop

This is fundamentally different from Phase 0:
    Phase 0: fixed 2 LLM calls per iteration (execute + reflect)
    Phase 1: variable LLM calls — loops until the LLM decides to stop

The LLM is now in control of the loop. It decides:
    - Which tool to call (or none)
    - What arguments to pass
    - When to stop and give the final answer
"""

import json

from llm import llm_call
from memory import ConversationMemory
from tools import Tool, DEFAULT_TOOLS, get_tool_by_name

# System prompt — defines who the agent is and how it should behave.
# This is the equivalent of BabyAGI's execution prompt, but for a
# conversational agent with tools.
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

When you need to perform calculations, get the current time, or execute code,
use the appropriate tool. You can call multiple tools in sequence to solve
complex problems.

Guidelines:
- Use tools when they would give a more accurate answer than your knowledge alone
- For math: always use the calculator tool instead of mental math
- For code: use python_repl when the user asks to run or test code
- Think step by step for complex problems
- Give clear, concise final answers"""


class Agent:
    def __init__(self, tools: list[Tool] | None = None, max_iterations: int = 10):
        """
        Create a ReAct agent.

        Args:
            tools: List of tools the agent can use. Defaults to built-in tools.
            max_iterations: Safety limit on tool-call loops to prevent infinite loops.
                           CrewAI has the same concept (max_iter in agent/core.py).
        """
        self.tools = tools or DEFAULT_TOOLS
        self.max_iterations = max_iterations
        self.memory = ConversationMemory(system_prompt=SYSTEM_PROMPT)

        # Pre-compute tool schemas for the LLM — done once, reused every call.
        self.tool_schemas = [t.to_openai_schema() for t in self.tools]

    def chat(self, user_input: str) -> str:
        """
        Send a message to the agent and get a response.

        This is the main entry point. It:
        1. Adds the user message to memory
        2. Runs the ReAct loop (LLM → tool calls → LLM → ... → final answer)
        3. Returns the final answer text

        Compare to CrewAI's CrewAgentExecutor.invoke() (crew_agent_executor.py:209):
          CrewAI: Sets up messages, calls _invoke_loop(), returns AgentFinish
          Us:     Same pattern, but simpler — no callbacks, events, or response models
        """
        self.memory.add_user_message(user_input)

        # ── The ReAct Loop ──────────────────────────────────────────
        # Compare to CrewAI's _invoke_loop_native_tools()
        #
        # CrewAI's loop (crew_agent_executor.py:328-339):
        #   while not isinstance(formatted_answer, AgentFinish):
        #       answer = get_llm_response(llm, messages, callbacks, ...)
        #       formatted_answer = process_llm_response(answer)
        #       if isinstance(formatted_answer, AgentAction):
        #           tool_result = execute_tool(...)
        #
        # Our loop — same structure, less ceremony:

        for iteration in range(self.max_iterations):
            # Step 1: Call the LLM with conversation history + tool definitions
            response = llm_call(
                messages=self.memory.get_messages(),
                tools=self.tool_schemas,
            )

            # Step 2: Check if the LLM wants to use tools
            if response.tool_calls:
                # LLM wants to call one or more tools
                # Add the assistant's message (with tool_calls) to memory
                self.memory.add_assistant_message(response)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    self._execute_tool_call(tool_call, iteration)
                # Loop back to Step 1 — the LLM will see the tool results
                continue

            # Step 3: No tool calls — this is the final answer
            self.memory.add_assistant_message(response)
            return response.content or "(no response)"

        # Safety net — max iterations reached
        return f"(reached max {self.max_iterations} tool-call iterations without a final answer)"

    def _execute_tool_call(self, tool_call, iteration: int):
        """
        Execute a single tool call and add the result to memory.

        Compare to CrewAI's execute_tool_and_check_finality()
        (crew_agent_executor.py:414-424):
          CrewAI: Looks up tool, validates fingerprint, calls with hooks
          Us:     Look up tool, call it, add result to memory
        """
        func_name = tool_call.function.name
        tool = get_tool_by_name(func_name, self.tools)

        if not tool:
            result = f"Error: Unknown tool '{func_name}'"
        else:
            # Parse arguments from JSON string
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
                result = f"Error: Could not parse arguments: {tool_call.function.arguments}"
            else:
                result = tool.execute(**args)

        # Print trace — so we can see what the agent is doing
        print(f"  [{iteration+1}] Tool: {func_name}({tool_call.function.arguments}) → {result[:200]}")

        # Add tool result to memory — the LLM will see this on the next iteration
        self.memory.add_tool_result(
            tool_call_id=tool_call.id,
            tool_name=func_name,
            result=result,
        )
