# Phase 0 — Architecture: Understanding Agent Loops

## Part 1: Two Agent Architectures Compared

### Architecture A: BabyAGI (Task Decomposition Loop)

```
┌─────────────────────────────────────────────────────┐
│                    OBJECTIVE                         │
│           "Solve world hunger"                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │ INITIAL TASK │
              │  (seed task) │
              └──────┬──────┘
                     │
    ┌────────────────▼────────────────────┐
    │            MAIN LOOP                │
    │                                     │
    │  ┌───────────────────────────┐      │
    │  │ 1. POP next task from     │      │
    │  │    priority queue         │      │
    │  └────────────┬──────────────┘      │
    │               ▼                     │
    │  ┌───────────────────────────┐      │
    │  │ 2. EXECUTION AGENT        │      │
    │  │    prompt = objective      │      │     ┌──────────────┐
    │  │          + task            │◄─────┼─────│  VECTOR DB   │
    │  │          + context (RAG)   │      │     │  (ChromaDB)  │
    │  │    → LLM call #1          │──────┼────►│  stores all  │
    │  └────────────┬──────────────┘      │     │  results     │
    │               ▼                     │     └──────────────┘
    │  ┌───────────────────────────┐      │
    │  │ 3. TASK CREATION AGENT    │      │
    │  │    prompt = result         │      │
    │  │          + objective       │      │
    │  │          + incomplete tasks│      │
    │  │    → LLM call #2          │      │
    │  │    → returns new tasks     │      │
    │  └────────────┬──────────────┘      │
    │               ▼                     │
    │  ┌───────────────────────────┐      │
    │  │ 4. PRIORITIZATION AGENT   │      │
    │  │    prompt = all tasks      │      │
    │  │          + objective       │      │
    │  │    → LLM call #3          │      │
    │  │    → reordered task list   │      │
    │  └────────────┬──────────────┘      │
    │               │                     │
    │               └─────► loop ─────────┘
    │                                     │
    │  EXIT: when task queue is empty     │
    └─────────────────────────────────────┘
```

**Data flow per iteration:**
```
task_queue.pop() → execution_agent(objective, task, context_from_vectordb)
                       │
                       ▼ result
                 vector_db.store(result)
                       │
                       ▼ result
                 task_creation_agent(objective, result, incomplete_tasks)
                       │
                       ▼ new_tasks
                 task_queue.extend(new_tasks)
                       │
                       ▼
                 prioritization_agent(all_tasks, objective)
                       │
                       ▼ reordered_tasks
                 task_queue.replace(reordered_tasks)
```

**Key properties:**
- 3 LLM calls per iteration (expensive but thorough)
- Memory is external (vector DB) — survives across iterations
- Task list grows and shrinks dynamically
- Stops when no more tasks are generated
- No tool use — the LLM only generates text, never executes code

---

### Architecture B: BabyAGI 2o (Self-Building Tool Loop)

```
┌─────────────────────────────────────────────────────┐
│                   USER TASK                          │
│           "Build me a weather app"                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │            MAIN LOOP                   │
    │                                        │
    │  ┌──────────────────────────────┐      │
    │  │  LLM decides next action     │      │
    │  │  (has full conversation       │      │
    │  │   history as context)        │      │
    │  └────────────┬─────────────────┘      │
    │               │                        │
    │          ┌────┴────┐                   │
    │          ▼         ▼                   │
    │    ┌──────────┐ ┌──────────────┐       │
    │    │ USE a    │ │ CREATE a new │       │
    │    │ tool     │ │ tool         │       │     ┌──────────────────┐
    │    │          │ │              │       │     │  TOOL REGISTRY   │
    │    │ call_tool│ │ exec(code)   │◄──────┼────►│                  │
    │    │ (name,   │ │ register_tool│       │     │ • create_tool    │
    │    │  args)   │ │ (name, func, │       │     │ • install_pkg    │
    │    │          │ │  desc, params│       │     │ • task_completed │
    │    └────┬─────┘ └──────┬───────┘       │     │ • [dynamic...]   │
    │         │              │               │     └──────────────────┘
    │         ▼              ▼               │
    │    ┌────────────────────────┐          │
    │    │ Result appended to     │          │
    │    │ conversation messages  │          │
    │    └────────────┬───────────┘          │
    │                 │                      │
    │                 └──► loop ─────────────┘
    │                                        │
    │  EXIT: LLM calls task_completed()      │
    │     or max_iterations (50) reached     │
    └────────────────────────────────────────┘
```

**Data flow per iteration:**
```
messages (full history) → LLM → tool_calls[]
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                    create_tool()        use_tool()
                          │                   │
                          ▼                   ▼
                    register in           execute and
                    tool registry         get result
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                          append result to messages
```

**Key properties:**
- 1 LLM call per iteration (cheaper)
- Memory is the conversation history (messages list) — no external DB
- The agent builds its own tools at runtime via `exec()`
- Uses native function calling (tool_calls) instead of text parsing
- Stops when LLM explicitly signals completion
- Can install packages and execute arbitrary code

---

## Part 2: Core Patterns Extracted

### Pattern 1: The Agent Loop (shared by both)

Every AI agent follows this fundamental pattern:

```
initialize(state)
while not done:
    observation = perceive(state)       # what do I know?
    decision    = think(observation)     # what should I do?  (LLM call)
    result      = act(decision)          # do it
    state       = update(state, result)  # remember what happened
```

The differences between agents are:
- **What is "state"?** → task queue + vector DB (A) vs. conversation history + tool registry (B)
- **What is "perceive"?** → RAG lookup (A) vs. full message history (B)
- **What is "act"?** → just text output (A) vs. tool execution (B)
- **When is "done"?** → empty queue (A) vs. explicit signal (B)

### Pattern 2: Memory

| Aspect | BabyAGI | BabyAGI 2o |
|--------|---------|------------|
| Storage | Vector DB (ChromaDB) | Conversation messages list |
| Retrieval | Semantic similarity (RAG) | Full history sent to LLM |
| Persistence | Survives restarts (on disk) | Lost when process ends |
| Scalability | Scales to many results | Limited by context window |
| Cost | Embedding cost per store | Token cost grows per call |

### Pattern 3: Task Management

| Aspect | BabyAGI | BabyAGI 2o |
|--------|---------|------------|
| Planning | Explicit (LLM creates + prioritizes tasks) | Implicit (LLM decides next step each iteration) |
| Decomposition | Formal task list with IDs | LLM reasons about what to do next |
| Adaptability | Re-prioritizes every loop | Fully dynamic — can change approach anytime |
| Visibility | Can inspect task queue | Opaque — reasoning is in LLM's head |

### Pattern 4: Tool Use

| Aspect | BabyAGI | BabyAGI 2o |
|--------|---------|------------|
| Tools | None — text only | Dynamic tool creation + execution |
| Capability growth | Fixed | Self-expanding (creates tools at runtime) |
| Safety | Safe (no code execution) | Risky (`exec()` runs arbitrary code) |
| Power | Limited to what LLM can say | Can install packages, call APIs, run code |

---

## Part 3: Our Build — Simplified Agent Architecture

We'll build a minimal agent that combines the best ideas from both:

```
┌─────────────────────────────────────────────────────┐
│                   USER OBJECTIVE                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │            AGENT LOOP                  │
    │                                        │
    │  State:                                │
    │  ├── objective (str)                   │
    │  ├── task_queue (deque)                │
    │  ├── memory (list of past results)     │
    │  └── iteration_count (int)             │
    │                                        │
    │  ┌──────────────────────────────┐      │
    │  │ 1. POP task from queue       │      │
    │  │    (or use objective if empty)│      │
    │  └────────────┬─────────────────┘      │
    │               ▼                        │
    │  ┌──────────────────────────────┐      │
    │  │ 2. EXECUTE                    │      │
    │  │    prompt = objective          │      │
    │  │          + task               │      │
    │  │          + last N results     │      │
    │  │    → LLM call                 │      │
    │  └────────────┬─────────────────┘      │
    │               ▼                        │
    │  ┌──────────────────────────────┐      │
    │  │ 3. STORE result in memory    │      │
    │  └────────────┬─────────────────┘      │
    │               ▼                        │
    │  ┌──────────────────────────────┐      │
    │  │ 4. REFLECT                    │      │
    │  │    prompt = result            │      │
    │  │          + objective          │      │
    │  │          + remaining tasks    │      │
    │  │    → LLM returns:            │      │
    │  │      { done: bool,           │      │
    │  │        new_tasks: [...],     │      │
    │  │        summary: str }        │      │
    │  └────────────┬─────────────────┘      │
    │               │                        │
    │          ┌────┴────┐                   │
    │          ▼         ▼                   │
    │       done?     add new tasks          │
    │       EXIT      to queue → loop ───────┘
    └────────────────────────────────────────┘
```

### Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Memory | Simple list (last N results) | No vector DB dependency, easy to understand. Upgrade to RAG later in Phase 2. |
| LLM calls per loop | 2 (execute + reflect) | Simpler than BabyAGI's 3 calls. Combine task creation + prioritization into one "reflect" step. |
| Task management | Explicit queue (like BabyAGI) | Visible and debuggable, unlike 2o's implicit approach. |
| Tool use | None for now | Keep Phase 0 focused on the loop pattern. Add tools in Phase 1. |
| Structured output | JSON from reflect step | Parse `{ done, new_tasks, summary }` — cleaner than BabyAGI's text parsing with regex. |
| Exit condition | LLM says done OR max iterations | Safety net prevents infinite loops. |
| LLM provider | OpenAI or Anthropic (configurable) | Use native SDK, no framework dependency. |

### File Structure

```
builds/00-understand-loop/
├── agent.py          # The loop: execute → store → reflect → repeat
├── llm.py            # Thin wrapper around OpenAI/Anthropic API calls
├── memory.py         # Simple list-based memory with last-N retrieval
├── prompts.py        # All prompt templates (execution + reflection)
└── main.py           # Entry point: set objective, run loop, print trace
```

### Pseudocode

```python
# main.py
objective = "Research and summarize the history of AI agents"
agent = Agent(objective=objective, max_iterations=10)
agent.run()

# agent.py
class Agent:
    def __init__(self, objective, max_iterations=10):
        self.objective = objective
        self.task_queue = deque([objective])  # seed with the objective itself
        self.memory = Memory(max_items=20)
        self.max_iterations = max_iterations

    def run(self):
        for i in range(self.max_iterations):
            if not self.task_queue:
                print("No more tasks. Done.")
                break

            task = self.task_queue.popleft()
            print(f"[Iteration {i+1}] Executing: {task}")

            # Step 1: Execute
            result = self.execute(task)
            print(f"  Result: {result[:200]}...")

            # Step 2: Store
            self.memory.add(task=task, result=result)

            # Step 3: Reflect
            reflection = self.reflect(result)
            print(f"  Done: {reflection['done']}")
            print(f"  New tasks: {reflection['new_tasks']}")

            if reflection["done"]:
                print(f"Objective complete. Summary: {reflection['summary']}")
                break

            for new_task in reflection["new_tasks"]:
                self.task_queue.append(new_task)

    def execute(self, task):
        context = self.memory.get_recent(n=5)
        prompt = build_execution_prompt(self.objective, task, context)
        return llm_call(prompt)

    def reflect(self, result):
        remaining = list(self.task_queue)
        prompt = build_reflection_prompt(self.objective, result, remaining)
        response = llm_call(prompt)  # expects JSON
        return parse_json(response)
```

### What This Teaches (vs. BabyAGI)

| Concept | BabyAGI | Our version |
|---------|---------|-------------|
| The loop | 3 separate agents | 2 steps (execute + reflect) — same idea, less ceremony |
| Memory | Vector DB + embeddings | Simple list — see why RAG matters when we hit limits |
| Task management | Separate creation + prioritization agents | Combined into one reflect step with structured output |
| Observability | Print statements | Print trace at each step — see the full reasoning chain |
| Complexity | ~600 lines with config | Target ~150 lines of core logic |
