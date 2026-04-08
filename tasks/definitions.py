"""Task metadata for the LLMFleet-SRE benchmark."""

TASKS = ["task_easy", "task_medium", "task_hard", "task_longhaul"]

TASK_METADATA = [
    {
        "name": "task_easy",
        "difficulty": "easy",
        "has_grader": True,
        "description": "Route 5 queued chat requests to a node that already has the model loaded.",
    },
    {
        "name": "task_medium",
        "difficulty": "medium",
        "has_grader": True,
        "description": "Recover an OOM-crashed node and clear a backing-up request queue under latency pressure.",
    },
    {
        "name": "task_hard",
        "difficulty": "hard",
        "has_grader": True,
        "description": "Evict chat models, load a code model, and serve a mixed premium/best-effort queue.",
    },
    {
        "name": "task_longhaul",
        "difficulty": "hard",
        "has_grader": True,
        "description": "Sustain cluster performance across a 50-step episode with a quiet-to-spike-to-quiet traffic shift.",
    },
]
