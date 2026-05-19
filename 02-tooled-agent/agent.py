"""
ReAct Agent — same loop as Phase 1, now with more tools + RAG.

The agent code itself barely changes between Phase 1 and Phase 2.
That's the beauty of the tool-based architecture: adding new capabilities
is just adding new tools to the registry. The loop stays the same.
"""

import json

from llm import llm_call
from memory import ConversationMemory
from tools import Tool, get_tool_by_name

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

You can:
- Calculate math expressions
- Get the current time
- Execute Python code
- Read, write, and list files
- Search the web for current information
- Search a knowledge base of ingested documents
- Ingest new documents into the knowledge base

Guidelines:
- Use tools when they would give a more accurate answer
- For math: always use the calculator tool
- For current events or facts: use web_search
- For questions about ingested documents: use search_knowledge
- Think step by step for complex problems
- Give clear, concise final answers"""


class Agent:
    def __init__(self, tools: list[Tool], max_iterations: int = 15):
        self.tools = tools
        self.max_iterations = max_iterations
        self.memory = ConversationMemory(system_prompt=SYSTEM_PROMPT)
        self.tool_schemas = [t.to_openai_schema() for t in self.tools]

    def chat(self, user_input: str) -> str:
        """Same ReAct loop as Phase 1 — tools are the only difference."""
        self.memory.add_user_message(user_input)

        for iteration in range(self.max_iterations):
            response = llm_call(
                messages=self.memory.get_messages(),
                tools=self.tool_schemas,
            )

            if response.tool_calls:
                self.memory.add_assistant_message(response)
                for tool_call in response.tool_calls:
                    self._execute_tool_call(tool_call, iteration)
                continue

            self.memory.add_assistant_message(response)
            return response.content or "(no response)"

        return f"(reached max {self.max_iterations} iterations)"

    def _execute_tool_call(self, tool_call, iteration: int):
        func_name = tool_call.function.name
        tool = get_tool_by_name(func_name, self.tools)

        if not tool:
            result = f"Error: Unknown tool '{func_name}'"
        else:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
                result = f"Error: Could not parse arguments"
            else:
                result = tool.execute(**args)

        # Truncate long results for display
        display = result[:200] + "..." if len(result) > 200 else result
        print(f"  [{iteration+1}] {func_name}(...) → {display}")

        self.memory.add_tool_result(
            tool_call_id=tool_call.id,
            tool_name=func_name,
            result=result,
        )
