"""
SqliteCheckpointer — a subscriber that persists ResearchState on
every StateTransition event.

Design:

    - One SQLite DB per run: runs/<run_id>/state.db.
    - One table: checkpoints (seq, run_id, ts, status, state JSON).
    - Append-only — we never UPDATE. Full history of every transition
      is preserved for post-mortem debugging.
    - Commit on every row. Each row is a crash-safe savepoint.

Resume semantics:

    The checkpoint is written BEFORE the agent executes (see
    orchestrator._transition). So `latest()` returns the state the
    last agent was *about to* process. On resume, the orchestrator
    dispatches on state.status and re-runs that agent from the top —
    no inconsistent partial state.

    Granularity is step-level. A crash mid-agent re-runs that entire
    step on resume (including any expensive tool calls inside it).

Why SQLite over pickle:
    - Atomic commits.
    - Full history via ORDER BY seq.
    - Zero external deps (ships with Python).
    - Inspectable with any sqlite client while the run is paused.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from events import Event, StateTransition
from state import ResearchState


class SqliteCheckpointer:
    def __init__(self, run_id: str, runs_dir: Path) -> None:
        self.run_id = run_id
        self.db_path = runs_dir / run_id / "state.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                seq     INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id  TEXT NOT NULL,
                ts      REAL NOT NULL,
                status  TEXT NOT NULL,
                state   TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    async def handle(self, event: Event) -> None:
        if not isinstance(event, StateTransition):
            return
        self._conn.execute(
            "INSERT INTO checkpoints(run_id, ts, status, state) VALUES (?, ?, ?, ?)",
            (event.run_id, event.timestamp, event.to_status, json.dumps(event.snapshot)),
        )
        self._conn.commit()

    def latest(self) -> ResearchState | None:
        row = self._conn.execute(
            "SELECT state FROM checkpoints WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
            (self.run_id,),
        ).fetchone()
        if row is None:
            return None
        return ResearchState.from_dict(json.loads(row[0]))

    def close(self) -> None:
        self._conn.close()
