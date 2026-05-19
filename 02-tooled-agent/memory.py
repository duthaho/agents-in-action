"""Conversation memory — same as Phase 1."""


class ConversationMemory:
    def __init__(self, system_prompt: str, max_messages: int = 50):
        self.system_message = {"role": "system", "content": system_prompt}
        self.messages: list[dict] = [self.system_message]
        self.max_messages = max_messages

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, message):
        if hasattr(message, "model_dump"):
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
            self.messages.append(message)
        self._trim()

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        self._trim()

    def get_messages(self) -> list[dict]:
        return self.messages

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.messages = [self.system_message] + self.messages[-(self.max_messages - 1):]
