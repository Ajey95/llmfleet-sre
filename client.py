"""LLMFleet-SRE Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LLMFleetAction, LLMFleetObservation

class LLMFleetSreEnv(
    EnvClient[LLMFleetAction, LLMFleetObservation, State]
):
    """
    Client for the LLMFleet-SRE Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    """

    def _step_payload(self, action: LLMFleetAction) -> Dict:
        """Convert LLMFleetAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[LLMFleetObservation]:
        """Parse server response into StepResult[LLMFleetObservation]."""
        obs_data = payload.get("observation", {})
        # Rebuild nested models
        from .models import NodeState, IncomingRequest
        nodes = {
            k: NodeState(**v) if isinstance(v, dict) else v
            for k, v in obs_data.get("nodes", {}).items()
        }
        queue = [
            IncomingRequest(**r) if isinstance(r, dict) else r
            for r in obs_data.get("request_queue", [])
        ]
        observation = LLMFleetObservation(
            nodes=nodes,
            request_queue=queue,
            step=obs_data.get("step", 0),
            step_budget=obs_data.get("step_budget", 30),
            last_action_result=obs_data.get("last_action_result", ""),
            sla_violations=obs_data.get("sla_violations", 0),
            requests_served=obs_data.get("requests_served", 0),
            done=payload.get("done", False),
            reward=float(payload.get("reward", 0.0)),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

# Alias for backward compat with openenv scaffold expectations
LlmfleetSreEnv = LLMFleetSreEnv
