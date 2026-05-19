# Phase 1 — Architecture: ReAct Agent with Tool Use

## What changed from Phase 0

Phase 0 agent could only **generate text**. It had no way to interact
with the world — no calculator, no file access, no code execution.

Phase 1 adds **tool use** via the LLM's native function calling API.
This is the single biggest capability upgrade an agent can get.

## The ReAct Pattern

ReAct = **Re**asoning + **Act**ing (Yao et al., 2022)

```
User: "What is 47 * 89 + 12?"

┌─────────────────────────────────────────────────┐
│ Iteration 1                                     │
│                                                 │
│ THINK: I need to calculate 47 * 89 first.       │
│ ACT:   call calculator(expression="47 * 89")    │
│ OBSERVE: 4183                                   │
│                                                 │
├─────────────────────────────────────────────────┤
│ Iteration 2                                     │
│                                                 │
│ THINK: Now I add 12 to 4183.                    │
│ ACT:   call calculator(expression="4183 + 12")  │
│ OBSERVE: 4195                                   │
│                                                 │
├─────────────────────────────────────────────────┤
│ Iteration 3                                     │
│                                                 │
│ THINK: I have the answer.                       │
│ RESPOND: "47 * 89 + 12 = 4195"                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Phase 0 loop vs Phase 1 loop

```
PHASE 0 (text only):              PHASE 1 (with tools):
─────────────────────              ──────────────────────
while not done:                    while not done:
  result = llm(prompt)               response = llm(messages, tools)
  memory.add(result)                 if response has tool_calls:
  reflection = llm(result)               result = execute_tool(call)
  done = reflection["done"]             messages.append(result)
                                     else:
                                         done = True  # final answer
```

Key difference: Phase 0 always made a fixed number of LLM calls per
iteration (execute + reflect). Phase 1 loops **until the LLM decides
to stop calling tools and give a final answer**.

## Native Function Calling vs ReAct Text Parsing

There are two ways an LLM can use tools:

### Option A: Native function calling (what we'll use)
```python
# LLM returns structured tool_calls in the API response
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[{                          # ← tool definitions
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
    }]
)
# response.choices[0].message.tool_calls = [
#   {"id": "call_123", "function": {"name": "calculator", "arguments": '{"expression": "47*89"}'}}
# ]
```

### Option B: ReAct text parsing (what CrewAI falls back to)
```
LLM output:
  Thought: I need to calculate 47 * 89
  Action: calculator
  Action Input: {"expression": "47 * 89"}

Code parses this with regex → extracts tool name and args
```

**We use Option A** because:
- More reliable (structured output, no regex needed)
- All modern LLMs support it (OpenAI, Anthropic, etc.)
- CrewAI prefers it too (see crew_agent_executor.py:315-323)

## Tool System Design

Inspired by CrewAI's @tool decorator (base_tool.py:542-621):

```python
# CrewAI's approach — reads docstring + type hints automatically:
@tool
def calculator(expression: str) -> str:
    '''Evaluate a mathematical expression.'''
    return str(eval(expression))

# Under the hood, this generates:
# {
#   "name": "calculator",
#   "description": "Evaluate a mathematical expression.",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "expression": {"type": "string"}
#     },
#     "required": ["expression"]
#   }
# }
```

We'll implement the same pattern — a `@tool` decorator that turns
any typed, documented function into a tool the LLM can call.

## File Structure

```
builds/01-minimal-agent/
├── tools.py      # @tool decorator + built-in tools (calculator, time, python)
├── llm.py        # OpenAI wrapper WITH function calling support
├── memory.py     # Conversation message history (not just results)
├── agent.py      # ReAct loop: send messages+tools → execute tool calls → repeat
└── main.py       # Interactive REPL — chat with the agent
```

## Data Flow

```
User input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Agent.chat(user_input)                              │
│                                                     │
│  1. memory.add(role="user", content=user_input)     │
│                                                     │
│  2. while True:  ◄─────────────────────────┐        │
│       │                                    │        │
│       ▼                                    │        │
│     response = llm_call(                   │        │
│       messages=memory.messages,            │        │
│       tools=tool_schemas                   │        │
│     )                                      │        │
│       │                                    │        │
│       ├── has tool_calls?                  │        │
│       │     YES:                           │        │
│       │       for each call:               │        │
│       │         result = tool.execute()    │        │
│       │         memory.add(tool_result)    │        │
│       │         ──────────────────────► loop│        │
│       │                                    │        │
│       │     NO (final answer):             │        │
│       │       memory.add(assistant_msg)    │        │
│       │       return content ──────────────┘        │
│                                                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
Display answer to user
```
