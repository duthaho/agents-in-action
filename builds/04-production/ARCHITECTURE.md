# Phase 4 — Architecture: Production Patterns

## What changed from Phase 3

Phase 3 was a sync multi-agent research pipeline: router → researcher
→ writer, sharing a `ResearchState`. It worked, but it was naive —
no observability, no persistence, no retries, no cost tracking, no
human oversight, no streaming. Every call was fire-and-forget and
every failure was a stack trace.

Phase 4 keeps the same three agent roles and the same state shape
but rebuilds the runtime from scratch to demonstrate six production
patterns:

| # | Pattern             | What it buys you                                          |
|---|---------------------|-----------------------------------------------------------|
| 1 | Observability       | A replayable JSONL log of every LLM call, tool call, and state transition |
| 2 | Cost tracking       | Per-call token counts and USD estimates, rolled up per agent and overall |
| 3 | Error handling      | LLM errors retry with backoff; tool errors are fed back to the LLM to adapt |
| 4 | State persistence   | Crash the process, resume from the last checkpoint, continue the run |
| 5 | HITL approval       | Sensitive tool calls block for human approval before executing           |
| 6 | Streaming           | LLM tokens yielded as they arrive, not buffered until the call completes |

## The two decisions that shape everything

### 1. Async from day 1

The whole runtime runs on `asyncio`. LLM calls, tool execution,
event publishing, and the approval gate are all async. We use
`openai.AsyncOpenAI` as the sole client.

Why async-first? Because streaming (feature #6) is trivially async
and everything else is easy to make async. If we start sync and add
streaming later, we'd rewrite every file. Start async, pay the small
cognitive tax upfront, cash it in when streaming lands.

### 2. Event bus at the center

One `AsyncEventBus`. Every significant action publishes an `Event`
onto it:

- LLM call started / completed / failed
- Tool call started / completed / failed
- State transition (before each agent step)
- Approval requested / resolved
- Token chunk (during streaming)
- Cost recorded (after each LLM call)

Production features are **subscribers** to this bus, not
instrumentation sprinkled through agent code:

```
              ┌─────────────┐
              │   Agents    │  publish events
              │   + Tools   │        │
              │  + LLM wrap │        ▼
              └─────────────┘  ┌──────────┐
                               │ EventBus │
                               └────┬─────┘
                                    │ fan-out
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        ┌──────────┐         ┌──────────┐         ┌─────────────┐
        │  Logger  │         │   Cost   │         │Checkpointer │
        │ (JSONL)  │         │ Tracker  │         │  (SQLite)   │
        └──────────┘         └──────────┘         └─────────────┘
```

The payoff: agent code doesn't know logging, cost tracking, or
checkpointing exist. Adding a new subscriber (say, an OpenTelemetry
exporter) is one new file and one `bus.subscribe()` call in `main.py`.
Removing one is the reverse. No agent file changes.

A `RunContext` object is threaded through every call to carry the
run-scoped dependencies: `run_id`, `bus`, `checkpointer`,
`approval_gate`, `stream`, `current_agent`. Nothing reaches for
globals.

## The six features, mapped to files

| Feature                 | Implementation                                               |
|-------------------------|--------------------------------------------------------------|
| Observability           | `logger.py` — a `JsonlLogger` subscriber, one file per run   |
| Cost tracking           | `cost.py` subscriber + `CostRecorded` emission in `llm.py`   |
| Error handling + retries| `errors.py` types + retry wrapper in `llm.py` + error feedback in `agents/base.py` |
| State persistence       | `checkpoint.py` `SqliteCheckpointer` subscriber + `resume()` in `orchestrator.py` |
| HITL approval           | `approval.py` + `sensitive` flag in `tools.py`               |
| Streaming               | `llm_stream` async generator in `llm.py` + streaming branch in `agents/base.py` |

## File structure

```
builds/04-production/
├── ARCHITECTURE.md       ← this file
├── events.py             Event dataclasses + AsyncEventBus
├── state.py              ResearchState + to_dict/from_dict
├── errors.py             ToolError, LLMError, MaxRetriesExceeded, ApprovalDenied
├── context.py            RunContext
├── llm.py                Async LLM wrapper: call, stream, retries, cost emission
├── logger.py             JsonlLogger subscriber
├── cost.py               CostTracker subscriber
├── checkpoint.py         SqliteCheckpointer subscriber
├── approval.py           AsyncApprovalGate (CLI prompt today, HTTP-ready shape)
├── tools.py              Tool + web_search + calculator + write_file
├── agents/
│   ├── __init__.py
│   ├── base.py           Async ReAct loop, publishes events, honors approvals
│   ├── router.py         plan / evaluate
│   ├── researcher.py     gather facts via tools
│   └── writer.py         compose final report
├── orchestrator.py       SequentialPipeline + GraphOrchestrator (both async)
├── runs/                 Per-run artifacts: <run_id>/events.jsonl + state.db
└── main.py               CLI: --query, --resume, --pattern, --stream
```

Each file has one responsibility. Read them in this order:

`events.py → state.py → errors.py → context.py → llm.py → logger.py →
cost.py → checkpoint.py → approval.py → tools.py → agents/base.py →
router/researcher/writer → orchestrator.py → main.py`

This order mirrors the dependency graph — each file only imports
things defined earlier in the order.

## Build order (≠ read order)

We don't scaffold all 13 files at once. Features land incrementally,
and each intermediate state is runnable:

| Step | What lands                                             | Demo                                      |
|------|--------------------------------------------------------|-------------------------------------------|
| 1    | Async baseline (everything except the 4 subscribers)  | Runs end-to-end silently, no log written  |
| 2    | `logger.py` + wire into `main.py`                     | `runs/<id>/events.jsonl` has full trace   |
| 3    | `cost.py` + price table + `CostRecorded` emission      | Cost report prints at end of each run     |
| 4    | Retry wrapper in `llm.py` + tool error feedback       | Broken network retries cleanly; bad tools adapt |
| 5    | `checkpoint.py` + `--resume` in `main.py`             | Crash mid-run, resume from last step      |
| 6    | `approval.py` + `write_file` sensitive tool           | CLI approval prompt; denial replans       |
| 7    | `llm_stream` + `--stream` flag                        | Tokens print progressively                |

After each step you can `diff` against Phase 3 (`builds/03-multi-agent/`)
to see what a "production hardening" pass actually looks like.

## ResearchState

Same shape as Phase 3, with serialization helpers for the checkpointer:

```python
@dataclass
class ResearchState:
    query: str
    research_plan: str = ""
    findings: list[str] = field(default_factory=list)
    report: str = ""
    status: str = "planning"           # planning | researching | writing | done
    iteration: int = 0
    max_iterations: int = 3

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "ResearchState": ...
```

The `status` field is the resume key — the orchestrator dispatches
the next step based on it.

## Agents

Three specialist agents, each a thin subclass of `BaseAgent` (an
async ReAct loop):

- **Router** — plans the research (text-only, no tools) and, in the
  graph orchestrator, re-evaluates whether findings are sufficient
  after each research pass.
- **Researcher** — uses `web_search`, `calculator`, and `write_file`
  (sensitive) to gather facts. Increments `state.iteration`.
- **Writer** — turns the findings list into a polished markdown
  report (text-only, no tools).

## Orchestrators

Both Phase 3 patterns carry over, now async:

- **SequentialPipeline** — fixed order: plan → research → write.
  Predictable; doesn't loop.
- **GraphOrchestrator** — state machine. After each research pass
  the router evaluates findings and either loops back
  (`researching`) or advances (`writing`). Capped by
  `max_iterations` as a safety net.

Both implement `run(ctx, query)` and `resume(ctx, state)`, and both
publish a `StateTransition` event before each agent runs. The
checkpointer subscribes to those and persists the full state — so
resume is simply "load the last row, call `resume(ctx, state)`".

## Event types

Defined in `events.py` as dataclasses. All share `run_id`,
`timestamp`, `kind`:

```
Event
├── LLMCallStarted / LLMCallCompleted / LLMCallFailed
├── ToolCallStarted / ToolCallCompleted / ToolCallFailed
├── StateTransition
├── ApprovalRequested / ApprovalResolved
├── TokenChunk
└── CostRecorded
```

`AsyncEventBus` fan-out is sequential (`for h in subs: await h(e)`),
not `asyncio.gather`. Sequential preserves event ordering — the
logger and checkpointer must see events in the order they happened,
which matters for replay and for crash-safety.

## Retry policy

Only the non-streaming `llm_call` path has retries. Exponential
backoff with jitter: 0.5s → 1s → 2s → 4s, ±250ms. Four attempts max.
Retryable errors are explicitly enumerated:

- `APITimeoutError`
- `APIConnectionError`
- `RateLimitError`
- `InternalServerError`

Everything else (`BadRequestError`, `AuthenticationError`) is
permanent and surfaces immediately — retrying is wasted money.

Tool calls are not retried at the plumbing level. A failing tool
raises `ToolError`, the agent catches it, feeds the error string
back to the LLM as the tool result, and the LLM decides whether to
retry or pick a different approach. The LLM *is* the retry policy
for tools — that's the "self-healing" property that makes agents
feel magical.

## HITL approval

The `AsyncApprovalGate` is a per-call gate on sensitive tools. When
a tool flagged `sensitive=True` is invoked:

1. The tool publishes `ApprovalRequested(agent, tool, args)`.
2. The gate prints a CLI prompt (`input()` wrapped in a threadpool
   executor so the event loop stays free).
3. The user answers `y` / `n`.
4. The gate publishes `ApprovalResolved(granted=bool)`.
5. If granted, the tool executes. If denied, `ApprovalDenied` is
   raised and the agent loop feeds a `"DENIED: ..."` observation
   back to the LLM, which replans.

Approval is per-call with actual args visible, not per-tool. An
agent may call the same sensitive tool three times in one run with
different args and each call gets a fresh approval.

CLI today. Swapping in an HTTP gate is one method: replace
`_prompt()` with an HTTP endpoint that resolves a `Future` when the
user answers.

## Streaming

`llm_stream(ctx, ...)` is an async generator. It opens a streaming
completion with `stream=True, stream_options={"include_usage": True}`
and yields `StreamChunk` objects as content deltas arrive. It also
publishes `TokenChunk` events so a simple stdout subscriber can
print tokens progressively.

Sharp edges:

- **Tool calls still execute after the stream closes.** OpenAI
  streams tool-call JSON in fragments; running a tool on half-parsed
  args is unsafe. We accumulate the tool calls and execute them only
  once the stream is complete.
- **No retries on the streaming path.** Retry-after-partial-stream
  is a rabbit hole. Streaming is best-effort.
- **Cost events fire once per stream**, from the terminal usage
  chunk (`include_usage=True`).

## Known trade-offs (learning-scoped limits)

- **Sequential subscriber fan-out.** A slow subscriber back-pressures
  the agent loop. A production system would hand events to a
  background queue.
- **Step-level resume granularity.** A run that crashes after an
  expensive tool call re-runs that tool on resume. Tool-level resume
  would require serializing the in-flight message list.
- **No streaming retries.** See above.
- **Hardcoded price table** in `llm.py`. A real system reads prices
  from config or an API.
- **Approval state isn't checkpointed.** A crash mid-approval
  restarts the containing agent step.
- **CLI-only approval gate.** HTTP is a one-method swap.

These aren't hidden — the point of the phase is to show the pattern,
not to build a distributed production system.
