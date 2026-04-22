"""
Entry point — run the multi-agent research system.

Baseline usage (Step 1):
    export OPENAI_API_KEY="sk-..."
    python main.py --query "What are the latest trends in AI agents?"
    python main.py --query "..." --pattern graph
    python main.py --query "..." --pattern sequential

Later steps add CLI flags without changing this file's shape:
    --resume <run_id>   (Step 5: load last checkpoint and continue)
    --stream            (Step 7: print tokens as they arrive)

This file is the ONLY place where bus subscribers are wired up. Every
production feature (logger, cost tracker, checkpointer, token printer)
plugs in here as a `ctx.bus.subscribe(...)` call. Agents and tools
remain unaware of which subscribers exist in a given run.
"""
from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from context import RunContext
from cost import CostTracker
from events import AsyncEventBus
from logger import JsonlLogger
from orchestrator import GraphOrchestrator, SequentialPipeline

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def build_ctx(run_id: str) -> RunContext:
    bus = AsyncEventBus()
    return RunContext(run_id=run_id, bus=bus)


async def run(args: argparse.Namespace) -> None:
    run_id = args.resume or uuid.uuid4().hex[:12]
    ctx = build_ctx(run_id)

    # Step 2: JsonlLogger — writes runs/<run_id>/events.jsonl
    logger = JsonlLogger(run_id=run_id, runs_dir=RUNS_DIR)
    ctx.bus.subscribe(logger.handle)

    # Step 3: CostTracker — aggregates CostRecorded events
    tracker = CostTracker()
    ctx.bus.subscribe(tracker.handle)

    # Step 5 adds:  SqliteCheckpointer subscribed + ctx.checkpointer set
    # Step 6 adds:  AsyncApprovalGate attached to ctx.approval_gate
    # Step 7 adds:  ctx.stream = args.stream + TokenChunk stdout printer

    orchestrator = (
        GraphOrchestrator() if args.pattern == "graph" else SequentialPipeline()
    )

    try:
        state = await orchestrator.run(ctx, args.query)

        print("\n" + "=" * 60)
        print(f"  FINAL REPORT  (run_id: {run_id})")
        print("=" * 60)
        print(state.report)
        print("\n" + "-" * 60)
        print(f"  {state.iteration} research pass(es), "
              f"{len(state.findings)} findings, "
              f"{len(state.report)} chars in report")
        print(f"  Event log: runs/{run_id}/events.jsonl")
        print("-" * 60)
        print("  COST")
        print("-" * 60)
        print(tracker.report())
        print("-" * 60)
    finally:
        await logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: production-shaped multi-agent research system",
    )
    parser.add_argument("--query", default=None, help="user query (required unless --resume)")
    parser.add_argument("--pattern", choices=["sequential", "graph"], default="graph")
    parser.add_argument("--resume", default=None, help="resume a prior run_id (Step 5)")
    parser.add_argument("--stream", action="store_true", help="stream tokens (Step 7)")
    args = parser.parse_args()

    if not args.resume and not args.query:
        parser.error("--query is required unless --resume is given")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
