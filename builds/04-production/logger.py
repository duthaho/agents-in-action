"""
JsonlLogger — the first production feature. A subscriber that writes
every event to runs/<run_id>/events.jsonl, one JSON line per event.

Design choices:

    - JSONL, not stdlib logging. We want exact control over the wire
      format. `jq` and `grep` work out of the box.

    - One file per run. A run is the replay/debug unit. Global logs
      across runs make debugging harder, not easier.

    - Flush after every write. Crash-safety matters because the
      checkpointer (Step 5) also writes on events — we want the log
      to be trustworthy post-crash. The cost is small at human-debug
      rates.

    - No levels, no filters. Every event is interesting. This is an
      agent runtime, not a web server.

    - Subscriber runs inline with publish(). Back-pressure on slow
      I/O is accepted as a learning trade-off; production would hand
      events to a background queue.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from events import Event


class JsonlLogger:
    def __init__(self, run_id: str, runs_dir: Path) -> None:
        self.run_id = run_id
        self.path = runs_dir / run_id / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so --resume (Step 5) continues the same log.
        self._fh = self.path.open("a", encoding="utf-8")

    async def handle(self, event: Event) -> None:
        self._fh.write(json.dumps(asdict(event), default=_json_default) + "\n")
        self._fh.flush()

    async def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _json_default(obj: Any) -> Any:
    """Fallback for anything dataclasses.asdict can't serialize natively."""
    try:
        return str(obj)
    except Exception:
        return repr(obj)
