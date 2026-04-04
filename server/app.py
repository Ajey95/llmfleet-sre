"""
FastAPI server for LLMFleet-SRE.

Wraps the LLMFleetEnvironment and exposes it over HTTP and WebSockets
compatibly with openenv-core 0.2.x framework components.
"""

from __future__ import annotations
from fastapi import FastAPI
from openenv.core.env_server import create_app

try:
    from ..models import LLMFleetAction, LLMFleetObservation
    from .environment import LLMFleetEnvironment
    from .tasks import TASKS
except ImportError:
    from models import LLMFleetAction, LLMFleetObservation
    from server.environment import LLMFleetEnvironment
    from server.tasks import TASKS

def _env_factory():
    return LLMFleetEnvironment(task_name=TASKS[0], step_budget=30)

# Create the standard OpenEnv FastAPI app
app = create_app(
    env=_env_factory,
    action_cls=LLMFleetAction,
    observation_cls=LLMFleetObservation,
    env_name="llmfleet-sre"
)

app.title = "LLMFleet-SRE"
app.description = (
    "An LLM agent manages a simulated GPU inference cluster  "
    "loading models, routing requests, recovering crashed nodes, "
    "and meeting SLA tiers. All compute is simulated; no GPU required."
)
app.version = "1.0.0"

@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {"name": "task_easy",   "difficulty": "easy",   "description": "Route 5 queued chat requests to a node that already has the model loaded."},
            {"name": "task_medium", "difficulty": "medium",  "description": "Recover an OOM-crashed node and clear a backing-up request queue under latency pressure."},
            {"name": "task_hard",   "difficulty": "hard",    "description": "Evict chat models, load a code model, and serve a mixed premium/best-effort queue."},
        ]
    }

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

