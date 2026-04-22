"""
RunContext — the bundle of run-scoped dependencies threaded through
every agent / tool / LLM call.

Exists so that nothing needs globals. A single object carries:

    run_id           Unique per run. Used to scope log files, the
                     SQLite checkpoint DB, and every event.
    bus              The AsyncEventBus. Agents, tools, and the LLM
                     wrapper publish events to it.
    checkpointer     Optional. Populated in Step 5 when --resume or
                     crash-safe persistence is wanted.
    approval_gate    Optional. Populated in Step 6 when sensitive
                     tools need human approval.
    stream           Flag for Step 7. When True, agents use
                     llm_stream instead of llm_call.
    current_agent    Mutated by agents/base.py before each tool call
                     so the approval prompt / events can show which
                     agent is asking. Kept mutable because passing
                     it through every tool call would be noisier.

Why a dataclass instead of a dict: it's self-documenting, type-checked
in IDEs, and the forward references keep the import graph clean
(checkpointer, approval, events are imported only for type checking).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from approval import AsyncApprovalGate
    from checkpoint import SqliteCheckpointer
    from events import AsyncEventBus


@dataclass
class RunContext:
    run_id: str
    bus: "AsyncEventBus"
    checkpointer: "SqliteCheckpointer | None" = None
    approval_gate: "AsyncApprovalGate | None" = None
    stream: bool = False
    current_agent: str = ""
