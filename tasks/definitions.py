"""Task metadata for the LLMFleet-SRE benchmark."""

TASKS = ["easy", "medium", "hard", "loghaul"]

TASK_NAME_ALIASES = {
    "task_easy": "easy",
    "task_medium": "medium",
    "task_hard": "hard",
    "task_longhaul": "loghaul",
    "med": "medium",
}


def normalize_task_name(task_name: str) -> str:
    """Map legacy task ids to canonical short ids."""
    return TASK_NAME_ALIASES.get(task_name, task_name)

TASK_METADATA = [
    {
        "id": "easy",
        "name": "easy",
        "difficulty": "easy",
        "has_grader": True,
        "ideal_action": "route_batch",
        "steps": 5,
        "description": "Route 5 queued chat requests to node_a which already has llama3-8b-chat loaded. No OOM crashes allowed.",
    },
    {
        "id": "medium",
        "name": "medium",
        "difficulty": "medium",
        "has_grader": True,
        "ideal_action": "restart_node",
        "steps": 10,
        "description": "Recover an OOM-crashed node and clear a backing-up request queue under latency pressure.",
    },
    {
        "id": "hard",
        "name": "hard",
        "difficulty": "hard",
        "has_grader": True,
        "ideal_action": "load_model",
        "steps": 30,
        "description": "Evict chat models, load a code model, and serve a mixed premium/best-effort queue.",
    },
    {
        "id": "loghaul",
        "name": "loghaul",
        "difficulty": "hard",
        "has_grader": True,
        "ideal_action": "route_batch",
        "steps": 50,
        "description": "Sustain cluster performance across a 50-step episode with a quiet-to-spike-to-quiet traffic shift.",
    },
]
