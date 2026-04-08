"""
FastAPI server for LLMFleet-SRE.

Wraps the LLMFleetEnvironment and exposes it over HTTP and WebSockets
compatibly with openenv-core 0.2.x framework components.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Optional
from fastapi import FastAPI, Query, Request
from openenv.core.env_server import create_app
from urllib.parse import parse_qs

try:
    from ..models import LLMFleetAction, LLMFleetObservation, LLMFleetState
    from .environment import LLMFleetEnvironment
    from .tasks import TASKS, grade
except ImportError:
    from models import LLMFleetAction, LLMFleetObservation, LLMFleetState
    from server.environment import LLMFleetEnvironment
    from server.tasks import TASKS, grade

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
            {"name": "task_easy", "difficulty": "easy", "has_grader": True, "description": "Route 5 queued chat requests to a node that already has the model loaded."},
            {"name": "task_medium", "difficulty": "medium", "has_grader": True, "description": "Recover an OOM-crashed node and clear a backing-up request queue under latency pressure."},
            {"name": "task_hard", "difficulty": "hard", "has_grader": True, "description": "Evict chat models, load a code model, and serve a mixed premium/best-effort queue."},
            {"name": "task_longhaul", "difficulty": "hard", "has_grader": True, "description": "Sustain cluster performance across a 50-step episode with a quiet-to-spike-to-quiet traffic shift."},
        ]
    }


@app.post("/grade")
async def grade_episode(
    request: Request,
    task_name: Optional[str] = Query(default=None),
):
    """
    Score a completed episode.
    
    Args:
        task_name: Name of the task (task_easy, task_medium, task_hard, or task_longhaul)
        final_state: Final LLMFleetState as a dictionary
    
    Returns:
        {"score": float in [0.0, 1.0], "task_name": str}
    """
    try:
        # Support both evaluator payload styles:
        # 1) /grade?task_name=... with body=<state>
        # 2) /grade with body={"task_name": ..., "final_state": {...}}
        try:
            body = await request.json()
        except Exception:
            body = {}

        body = body if isinstance(body, dict) else {}

        resolved_task_name = task_name or body.get("task_name")

        if isinstance(body.get("final_state"), dict):
            resolved_final_state = body.get("final_state")
        else:
            # If task_name is in query, treat the whole body as raw final_state.
            # This preserves compatibility with /grade?task_name=... + raw-state-body.
            if task_name:
                resolved_final_state = body if body else None
            else:
                # Without query task_name, avoid interpreting wrapper-only payloads as state.
                resolved_final_state = body if body and "task_name" not in body else None

        if not resolved_task_name:
            return {"error": "task_name is required", "score": 0.0, "task_name": ""}
        if not isinstance(resolved_final_state, dict):
            return {"error": "final_state must be an object", "score": 0.0, "task_name": resolved_task_name}

        state = LLMFleetState(**resolved_final_state)
        score = grade(resolved_task_name, state)
        return {"score": score, "task_name": resolved_task_name}
    except Exception as e:
        return {"error": str(e), "score": 0.0, "task_name": task_name or ""}

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

