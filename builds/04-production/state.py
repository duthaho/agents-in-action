"""
Shared state for the research pipeline.

Same shape as Phase 3, now with (de)serialization helpers so the
SqliteCheckpointer (Step 5) can persist the full state on each
agent transition.

The `status` field is the resume key. After a crash, the orchestrator
loads the last row from the checkpoint DB, reads `status`, and
dispatches the next step:

    planning    → router plans, advances to researching
    researching → researcher gathers, advances to writing (seq) or
                  routes back to router for evaluation (graph)
    writing     → writer produces the report, advances to done
    done        → run complete, nothing to do

Resume granularity is step-level, not tool-call-level. If a crash
happens mid-researcher, resume re-runs the entire researcher step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ResearchState:
    query: str
    research_plan: str = ""
    findings: list[str] = field(default_factory=list)
    report: str = ""
    status: str = "planning"           # planning | researching | writing | done
    iteration: int = 0
    max_iterations: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchState":
        return cls(**d)
