"""Task metadata for the LLMFleet-SRE benchmark."""

TASKS = ["task_easy", "task_medium", "task_hard", "task_longhaul"]

TASK_METADATA = [
    {
        "id": "task_easy",
        "name": "task_easy",
        "difficulty": "easy",
        "has_grader": True,
        "ideal_action": "route_batch",
        "steps": 5,
        "description": "Route 5 queued chat requests to node_a which already has llama3-8b-chat loaded. No OOM crashes allowed.",
    },
    {
        "id": "task_medium",
        "name": "task_medium",
        "difficulty": "medium",
        "has_grader": True,
        "ideal_action": "restart_node",
        "steps": 10,
        "description": "Recover an OOM-crashed node and clear a backing-up request queue under latency pressure.",
    },
    {
        "id": "task_hard",
        "name": "task_hard",
        "difficulty": "hard",
        "has_grader": True,
        "ideal_action": "load_model",
        "steps": 30,
        "description": "Evict chat models, load a code model, and serve a mixed premium/best-effort queue.",
    },
    {
        "id": "task_longhaul",
        "name": "task_longhaul",
        "difficulty": "hard",
        "has_grader": True,
        "ideal_action": "route_batch",
        "steps": 50,
        "description": "Sustain cluster performance across a 50-step episode with a quiet-to-spike-to-quiet traffic shift.",
    },
]
