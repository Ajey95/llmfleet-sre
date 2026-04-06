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

_envs: dict[str, LLMFleetEnvironment] = {}
_current_task = TASKS[0]

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
            {"name": "task_longhaul", "difficulty": "hard", "description": "Sustain cluster performance across a 50-step episode with a quiet-to-spike-to-quiet traffic shift."},
        ]
    }


@app.post("/reset", response_model=LLMFleetObservation)
async def reset(task_name: str = "task_easy", seed: int | None = None):
    global _current_task
    _current_task = task_name
    # Recreate env with correct step budget for this task
    _envs[task_name] = LLMFleetEnvironment(
        task_name=task_name,
        step_budget=50 if task_name == "task_longhaul" else 30,
    )
    obs = _envs[task_name].reset(seed=seed)
    return obs

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

