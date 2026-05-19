"""
Prompt templates — the "brains" of each agent step.

Key insight from BabyAGI:
  BabyAGI has 3 separate "agents" but they're really just 3 different
  prompts fed to the same LLM.  The prompt IS the agent's personality.

  - execution_agent prompt:  "Perform this task, here's context"
  - task_creation prompt:    "Given this result, what tasks remain?"
  - prioritization prompt:   "Reorder these tasks by importance"

We simplify to 2 prompts:
  - EXECUTION:  same idea as BabyAGI's execution agent
  - REFLECTION: combines task creation + prioritization into one step,
                and uses JSON output instead of BabyAGI's numbered-list
                parsing (which was fragile — see the regex at babyagi.py:447-453)
"""


def build_execution_prompt(objective: str, task: str, context: list[dict]) -> str:
    """
    Build the prompt for the execution step.

    Compare to BabyAGI's execution_agent() at babyagi.py:496-517:
      BabyAGI: "Perform one task based on the following objective: {objective}"
      Us:      Same structure, but we format context more clearly.

    The context parameter is a list of previous {task, result} dicts.
    BabyAGI retrieved these from a vector DB (semantic search).
    We just use the last N results (simpler, works for small runs).
    """
    prompt = f"""You are an AI assistant working toward an objective.

OBJECTIVE: {objective}

YOUR CURRENT TASK: {task}
"""

    if context:
        prompt += "\nPREVIOUSLY COMPLETED WORK (use this as context):\n"
        for i, entry in enumerate(context, 1):
            prompt += f"\n--- Task {i}: {entry['task']} ---\n"
            prompt += f"{entry['result']}\n"

    prompt += """
Complete the current task thoroughly. Be specific and detailed in your response.
Focus only on this task — do not try to complete the entire objective at once.
"""
    return prompt


def build_reflection_prompt(objective: str, task: str, result: str, remaining_tasks: list[str]) -> str:
    """
    Build the prompt for the reflection step.

    This replaces TWO agents from BabyAGI:
      1. task_creation_agent (babyagi.py:419-457) — generated new tasks
      2. prioritization_agent (babyagi.py:460-492) — reordered task queue

    BabyAGI parsed new tasks from numbered lists using regex.
    We ask for JSON instead — more reliable and easier to handle.

    The LLM decides:
      - Are we done with the objective?
      - If not, what tasks should we do next? (already in priority order)
      - A brief summary of progress so far
    """
    prompt = f"""You are evaluating progress toward an objective.

OBJECTIVE: {objective}

TASK JUST COMPLETED: {task}
RESULT:
{result}
"""

    if remaining_tasks:
        prompt += f"\nTASKS STILL IN QUEUE: {', '.join(remaining_tasks)}\n"
    else:
        prompt += "\nTASKS STILL IN QUEUE: (none)\n"

    prompt += """
Analyze the progress and respond in EXACTLY this JSON format (no other text):

{
    "done": true/false,
    "summary": "Brief summary of progress so far",
    "new_tasks": ["task 1", "task 2"]
}

Rules:
- Set "done" to true ONLY if the objective has been fully achieved through
  MULTIPLE completed tasks with concrete results (not just a plan or outline)
- A task that only produces a plan or list of sub-tasks does NOT count as completing the objective
- "new_tasks" should contain 1-3 NEW concrete, actionable tasks to progress toward the objective
- Do NOT repeat tasks that were already completed or are still in the queue
- Order new_tasks by priority (most important first)
- If done is true, new_tasks should be empty
"""
    return prompt
