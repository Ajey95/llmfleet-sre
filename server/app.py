"""
FastAPI server for LLMFleet-SRE.

Wraps the LLMFleetEnvironment and exposes it over HTTP and WebSockets
compatibly with openenv-core 0.2.x framework components.
"""

from __future__ import annotations
import json
from fastapi import FastAPI
from openenv.core.env_server import create_app
from urllib.parse import parse_qs

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


class ResetQueryToBodyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/reset":
            query = parse_qs(scope.get("query_string", b"").decode())
            payload = {}
            if query.get("task_name"):
                payload["task_name"] = query["task_name"][0]
            if query.get("seed"):
                seed_value = query["seed"][0]
                try:
                    payload["seed"] = int(seed_value)
                except ValueError:
                    payload["seed"] = seed_value

            if payload:
                body = json.dumps(payload).encode("utf-8")
                sent = False

                async def receive_wrapper():
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return {"type": "http.request", "body": b"", "more_body": False}

                scope = dict(scope)
                headers = [
                    (name, value)
                    for name, value in scope.get("headers", [])
                    if name not in {b"content-type", b"content-length"}
                ]
                headers.append((b"content-type", b"application/json"))
                headers.append((b"content-length", str(len(body)).encode("ascii")))
                scope["headers"] = headers
                scope["query_string"] = b""
                await self.app(scope, receive_wrapper, send)
                return

        await self.app(scope, receive, send)


app.add_middleware(ResetQueryToBodyMiddleware)

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

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

