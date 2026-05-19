"""
Agent — the core loop that ties everything together.

This is the equivalent of BabyAGI's main() at babyagi.py:548-609.

The pattern (shared by virtually every AI agent):

    initialize state
    while not done:
        observe  →  what do I know?      (memory retrieval)
        think    →  what should I do?     (LLM call)
        act      →  do it                 (execute task)
        update   →  remember what happened (store result)

BabyAGI's loop made 3 LLM calls per iteration:
    1. execution_agent()       — do the task
    2. task_creation_agent()   — generate new tasks from result
    3. prioritization_agent()  — reorder the task queue

Our loop makes 2 LLM calls per iteration:
    1. execute()  — do the task (same as BabyAGI)
    2. reflect()  — generate new tasks AND decide if done (combines 2 & 3)
"""

import json
from collections import deque

from llm import llm_call
from memory import Memory
from prompts import build_execution_prompt, build_reflection_prompt


class Agent:
    def __init__(self, objective: str, max_iterations: int = 10):
        self.objective = objective
        self.max_iterations = max_iterations

        # Task queue — same as BabyAGI's SingleTaskListStorage (babyagi.py:279-301)
        # BabyAGI used a deque with task_id tracking.
        # We use a simple deque of strings — no IDs needed at this scale.
        self.task_queue: deque[str] = deque()

        # Track completed task names so we can deduplicate.
        # BabyAGI didn't have this problem because its task_creation_agent
        # received the incomplete task list and was told not to overlap —
        # but LLMs don't reliably follow that instruction!
        # Lesson: never trust the LLM to manage state — enforce it in code.
        self.completed_tasks: set[str] = set()

        # Memory — stores completed task results for context
        self.memory = Memory(max_items=20)

        # Iteration counter for display and safety limit
        self.iteration = 0

    def run(self):
        """
        The main agent loop.

        Compare to BabyAGI's main() at babyagi.py:548-609:

        BabyAGI:                          Us:
        ─────────────────────────────     ────────────────────────────
        while loop:                       for i in range(max):
          task = queue.popleft()            task = queue.popleft()
          result = execution_agent()        result = execute(task)
          results_storage.add(result)       memory.add(task, result)
          new = task_creation_agent()       reflection = reflect(result)
          queue.extend(new)                 queue.extend(reflection.new_tasks)
          prioritization_agent()            (prioritization is implicit)
          sleep(5)                          (no sleep needed)

        Key difference: BabyAGI runs until the queue is empty.
        We add a max_iterations safety net to prevent infinite loops.
        """

        # Seed the queue — BabyAGI did this at babyagi.py:540-545
        # BabyAGI had OBJECTIVE and INITIAL_TASK as separate concepts.
        # INITIAL_TASK was something like "Develop a task list" — a first
        # step that naturally decomposes into sub-tasks.
        #
        # If we seed with the objective itself, the LLM solves it in one
        # shot and the reflect step says "done!" on the first iteration.
        # Instead, we seed with a planning task that forces decomposition.
        self.task_queue.append(
            f"Break down this objective into 2-4 concrete sub-tasks: {self.objective}"
        )

        print(f"\n{'='*60}")
        print(f"  AGENT STARTED")
        print(f"  Objective: {self.objective}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"{'='*60}\n")

        for self.iteration in range(1, self.max_iterations + 1):
            if not self.task_queue:
                print("\n✓ Task queue is empty. Agent finished.")
                break

            # --- Step 1: Pop the next task ---
            task = self.task_queue.popleft()
            self._print_status(task)

            # --- Step 2: Execute the task (LLM call #1) ---
            result = self._execute(task)
            print(f"\n  Result (preview): {result[:300]}{'...' if len(result) > 300 else ''}")

            # --- Step 3: Store in memory ---
            self.memory.add(task=task, result=result)
            self.completed_tasks.add(task.lower().strip())

            # --- Step 4: Reflect on progress (LLM call #2) ---
            reflection = self._reflect(task, result)

            if reflection is None:
                print("\n  ⚠ Reflection failed to parse. Continuing with empty queue.")
                continue

            print(f"\n  Done: {reflection['done']}")
            print(f"  Summary: {reflection['summary']}")

            if reflection["done"]:
                print(f"\n{'='*60}")
                print(f"  ✓ OBJECTIVE COMPLETE")
                print(f"  Summary: {reflection['summary']}")
                print(f"  Iterations used: {self.iteration}")
                print(f"{'='*60}\n")
                return

            # --- Step 5: Add new tasks to queue (with deduplication) ---
            # This is a lesson BabyAGI learned the hard way too —
            # LLMs will repeat tasks even when told not to.
            # We normalize and check against both completed and queued tasks.
            queued = {t.lower().strip() for t in self.task_queue}
            for new_task in reflection.get("new_tasks", []):
                normalized = new_task.lower().strip()
                if normalized in self.completed_tasks:
                    print(f"  ~ Skipped (already done): {new_task}")
                elif normalized in queued:
                    print(f"  ~ Skipped (already queued): {new_task}")
                else:
                    print(f"  + New task: {new_task}")
                    self.task_queue.append(new_task)
                    queued.add(normalized)

        else:
            # max_iterations reached without completing
            print(f"\n{'='*60}")
            print(f"  ⚠ MAX ITERATIONS ({self.max_iterations}) REACHED")
            print(f"  The objective may not be fully complete.")
            print(f"  Tasks remaining in queue: {len(self.task_queue)}")
            print(f"{'='*60}\n")

    def _execute(self, task: str) -> str:
        """
        Execute a single task using the LLM.

        Compare to BabyAGI's execution_agent() at babyagi.py:496-517:
          BabyAGI: context = context_agent(query=objective, top_results_num=5)
                   prompt = objective + context + task
                   return openai_call(prompt)

          Us:      context = memory.get_recent(n=5)
                   prompt = build_execution_prompt(objective, task, context)
                   return llm_call(prompt)

        Same structure — the only difference is HOW we retrieve context
        (recency vs. semantic similarity).
        """
        context = self.memory.get_recent(n=5)
        prompt = build_execution_prompt(self.objective, task, context)
        return llm_call(prompt)

    def _reflect(self, task: str, result: str) -> dict | None:
        """
        Reflect on the result and decide next steps.

        This replaces BabyAGI's task_creation_agent + prioritization_agent.

        Returns: {"done": bool, "summary": str, "new_tasks": [str, ...]}
        Or None if JSON parsing fails.
        """
        remaining = list(self.task_queue)
        prompt = build_reflection_prompt(self.objective, task, result, remaining)
        response = llm_call(prompt)

        # Parse JSON from LLM response.
        # BabyAGI used regex to parse numbered lists (babyagi.py:447-453).
        # JSON is more structured but LLMs sometimes wrap it in markdown.
        try:
            # Handle case where LLM wraps JSON in ```json ... ```
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]  # remove first line
                cleaned = cleaned.rsplit("```", 1)[0]  # remove last ```
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"\n  ⚠ Failed to parse reflection JSON: {e}")
            print(f"  Raw response: {response[:200]}")
            return None

    def _print_status(self, current_task: str):
        """Print current iteration status — our version of BabyAGI's colored output."""
        print(f"\n{'─'*60}")
        print(f"  Iteration {self.iteration}/{self.max_iterations}")
        print(f"  Current task: {current_task}")
        print(f"  Tasks in queue: {len(self.task_queue)}")
        print(f"  Memory entries: {len(self.memory)}")
        print(f"{'─'*60}")
