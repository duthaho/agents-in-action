"""
Memory — conversation message history with sliding window.

Phase 0 memory stored {task, result} pairs — a simple log.
Phase 1 memory stores the full OpenAI message history — this is what
the LLM actually sees, including tool calls and tool results.

This is closer to BabyAGI 2o's approach (conversation as memory)
than BabyAGI's approach (vector DB as memory).

The sliding window keeps the conversation bounded:
  - Always keep the system message (first message)
  - Keep the last N messages
  - Drop older messages when we exceed the limit

This is the simplest form of context management.
In Phase 2 we'll upgrade to RAG for better long-term memory.
"""


class ConversationMemory:
    def __init__(self, system_prompt: str, max_messages: int = 50):
        """
        Initialize with a system prompt.

        The system prompt defines the agent's personality and capabilities.
        It's always the first message and never gets evicted.
        """
        self.system_message = {"role": "system", "content": system_prompt}
        self.messages: list[dict] = [self.system_message]
        self.max_messages = max_messages

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, message):
        """
        Add an assistant response to the conversation.

        Accepts either a dict or an OpenAI ChatCompletionMessage object.
        We need to handle both because:
          - The LLM returns a ChatCompletionMessage object
          - We need to serialize it to a dict for the message history
          - Tool calls need special serialization
        """
        if hasattr(message, "model_dump"):
            # OpenAI ChatCompletionMessage → convert to dict
            msg = {"role": "assistant"}
            if message.content:
                msg["content"] = message.content
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            self.messages.append(msg)
        else:
            # Already a dict
            self.messages.append(message)
        self._trim()

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """
        Add a tool execution result to the conversation.

        OpenAI requires tool results to reference the tool_call_id
        from the assistant's message. This is how the LLM knows which
        tool call this result corresponds to.
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        self._trim()

    def get_messages(self) -> list[dict]:
        """Return the full message history for the LLM."""
        return self.messages

    def _trim(self):
        """
        Sliding window: keep system message + last N messages.

        This is the simplest way to manage context length.
        More sophisticated approaches:
          - Token counting (count actual tokens, not messages)
          - Summarization (compress old messages into a summary)
          - RAG (store old messages in vector DB, retrieve relevant ones)
        We'll explore these in later phases.
        """
        if len(self.messages) > self.max_messages:
            # Keep system message + most recent messages
            self.messages = [self.system_message] + self.messages[-(self.max_messages - 1):]

    def __len__(self):
        return len(self.messages)
