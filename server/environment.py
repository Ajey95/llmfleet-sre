"""
LLMFleet-SRE Environment Simulation Logic.

Simulates a 3-node GPU inference cluster where an LLM agent acts as an SRE:
- Load and evict models from nodes (VRAM-constrained)
- Route incoming requests to nodes that have the required model loaded
- Recover OOM-crashed nodes
- Manage SLA tiers (premium vs best_effort)
"""

from __future__ import annotations
import random
import uuid
from typing import Dict, List, Tuple, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        LLMFleetAction, LLMFleetObservation, LLMFleetState,
        NodeState, IncomingRequest, ModelSpec
    )
except ImportError:
    from models import (
        LLMFleetAction, LLMFleetObservation, LLMFleetState,
        NodeState, IncomingRequest, ModelSpec
    )


#  Static catalogue of available models 

MODEL_CATALOGUE: Dict[str, ModelSpec] = {
    "llama3-8b-chat":    ModelSpec(name="llama3-8b-chat",    vram_gb=18),
    "llama3-70b-chat":   ModelSpec(name="llama3-70b-chat",   vram_gb=45),
    "codellama-34b":     ModelSpec(name="codellama-34b",     vram_gb=35),
    "mistral-7b-sum":    ModelSpec(name="mistral-7b-sum",    vram_gb=16),
    "code-lora-adapter": ModelSpec(name="code-lora-adapter", vram_gb=8),
}

# Task type  required model mapping
TASK_MODEL_MAP: Dict[str, str] = {
    "chat":      "llama3-8b-chat",
    "code":      "codellama-34b",
    "summarize": "mistral-7b-sum",
}

SLA_THRESHOLD = 5     # steps before a premium request triggers a violation
MAX_QUEUE_SIZE = 20


class LLMFleetEnvironment(Environment):
    """
    Simulated LLM inference cluster SRE environment.

    The agent manages 3 nodes (80GB, 80GB, 40GB VRAM) and must:
    - Route requests to nodes with the required model loaded
    - Load/evict models to manage VRAM
    - Recover OOM-crashed nodes
    - Minimize latency and SLA violations to maximize reward
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "task_easy", step_budget: int = 30, seed: Optional[int] = None):
        self.task_name = task_name
        self.step_budget = step_budget
        self.rng = random.Random(seed)
        self._reset_internal()

    def _reset_internal(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._step = 0
        self._sla_violations = 0
        self._requests_served = 0
        self._requests_failed = 0
        self._cumulative_reward = 0.0
        self._request_counter = 0

        # Three nodes: A=80GB, B=80GB, C=40GB (asymmetry is intentional)
        self._nodes: Dict[str, NodeState] = {
            "node_a": NodeState(node_id="node_a", vram_total_gb=80),
            "node_b": NodeState(node_id="node_b", vram_total_gb=80),
            "node_c": NodeState(node_id="node_c", vram_total_gb=40),
        }
        self._request_queue: List[IncomingRequest] = []

    #  Public Environment API 

    def reset(self, task_name: Optional[str] = None, seed: Optional[int] = None) -> LLMFleetObservation:
        """Reset the environment for a new episode."""
        if task_name:
            self.task_name = task_name
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_internal()
        self._setup_for_task()
        return self._observe("Episode started.", reward=0.0, done=False)

    def step(self, action: LLMFleetAction) -> LLMFleetObservation:  # type: ignore[override]
        """Execute one environment step."""
        self._step += 1
        self._state.step_count = self._step
        reward = 0.0

        # Age all waiting requests
        for req in self._request_queue:
            req.age_steps += 1

        # Advance pending model loads (stochastic 13 step latency)
        for node in self._nodes.values():
            if node.load_latency_remaining > 0:
                node.load_latency_remaining -= 1
                if node.load_latency_remaining == 0:
                    node.status = "healthy"

        # Execute agent action
        result_msg, action_reward = self._execute_action(action)
        reward += action_reward

        # SLA violation check (after action so agent gets credit for fast routing)
        for req in self._request_queue:
            if req.sla_tier == "premium" and req.age_steps > SLA_THRESHOLD:
                self._sla_violations += 1
                reward -= 0.3

        # Latency penalty: -0.01 per request sitting in queue
        reward -= 0.01 * len(self._request_queue)

        # Idle penalty: -0.05 per healthy node that has models but can't serve any queued request
        if self._request_queue:
            idle_nodes = sum(
                1 for n in self._nodes.values()
                if n.status == "healthy" and len(n.loaded_models) > 0
                and not any(req.required_model in n.loaded_models for req in self._request_queue)
            )
            reward -= 0.05 * idle_nodes

        # Inject new requests (task-specific)
        self._inject_requests()

        self._cumulative_reward += reward
        done = self._step >= self.step_budget or len(self._request_queue) == 0

        return self._observe(result_msg, reward=reward, done=done)

    @property
    def state(self) -> State:
        """Returns openenv State for framework compatibility."""
        return self._state

    def get_full_state(self) -> LLMFleetState:
        """Returns full LLMFleetState for grading and the /state endpoint."""
        return LLMFleetState(
            nodes=self._nodes,
            request_queue=self._request_queue,
            step=self._step,
            step_budget=self.step_budget,
            sla_violations=self._sla_violations,
            requests_served=self._requests_served,
            requests_failed=self._requests_failed,
            cumulative_reward=self._cumulative_reward,
            task_name=self.task_name,
            episode_id=self._state.episode_id,
        )

    #  Action Execution 

    def _execute_action(self, action: LLMFleetAction) -> Tuple[str, float]:
        reward = 0.0
        at = action.action_type

        if at == "noop":
            return "No action taken.", reward

        node_id = action.target_node
        if node_id and node_id not in self._nodes:
            return f"Unknown node: {node_id}", reward - 0.1

        node = self._nodes.get(node_id) if node_id else None

        if at == "route_batch":
            return self._route_batch(node, node_id, action.request_ids)
        elif at == "load_model":
            return self._load_model(node, node_id, action.model_name)
        elif at == "evict_model":
            return self._evict_model(node, node_id, action.model_name)
        elif at == "restart_node":
            return self._restart_node(node, node_id)

        return "Unknown action.", reward

    def _route_batch(self, node, node_id, request_ids):
        if not node or node.status != "healthy":
            return f"Node {node_id} is not healthy.", -0.1
        if not request_ids:
            return "No request_ids specified.", -0.05

        served = 0
        for rid in request_ids:
            req = next((r for r in self._request_queue if r.request_id == rid), None)
            if not req:
                continue
            if req.required_model not in node.loaded_models:
                return f"Model {req.required_model} not loaded on {node_id}.", -0.1
            self._request_queue.remove(req)
            self._requests_served += 1
            served += 1

        reward = 0.2 * served
        return f"Routed {served} requests on {node_id}.", reward

    def _load_model(self, node, node_id, model_name):
        if not model_name or model_name not in MODEL_CATALOGUE:
            return f"Unknown model: {model_name}", -0.05
        if not node or node.status == "oom_crashed":
            return f"Node {node_id} is crashed.", -0.05
        if model_name in node.loaded_models:
            return f"{model_name} already loaded on {node_id}.", 0.0

        spec = MODEL_CATALOGUE[model_name]
        if node.vram_used_gb + spec.vram_gb > node.vram_total_gb:
            node.status = "oom_crashed"
            return f"OOM! {model_name} exceeds VRAM on {node_id}. Node crashed.", -0.5

        latency = self.rng.randint(1, 3)
        node.loaded_models.append(model_name)
        node.vram_used_gb += spec.vram_gb
        node.status = "loading"
        node.load_latency_remaining = latency
        return f"Loading {model_name} on {node_id} (ready in {latency} steps).", 0.0

    def _evict_model(self, node, node_id, model_name):
        if not model_name:
            return "No model_name specified.", -0.05
        if not node or node.status == "oom_crashed":
            return f"Node {node_id} is crashed.", -0.05
        if model_name not in node.loaded_models:
            return f"{model_name} not loaded on {node_id}.", -0.05

        spec = MODEL_CATALOGUE[model_name]
        node.loaded_models.remove(model_name)
        node.vram_used_gb = max(0, node.vram_used_gb - spec.vram_gb)
        return f"Evicted {model_name} from {node_id}. Freed {spec.vram_gb}GB.", 0.0

    def _restart_node(self, node, node_id):
        if not node:
            return "No node specified.", -0.05
        node.status = "loading"
        node.loaded_models = []
        node.vram_used_gb = 0
        node.load_latency_remaining = 2
        return f"Restarting {node_id}. Back online in 2 steps.", 0.0

    #  Task Setup 

    def _setup_for_task(self):
        """Pre-load models and seed queue based on task difficulty."""
        if self.task_name == "task_easy":
            # Node A has the chat model. 5 chat requests in queue.
            self._nodes["node_a"].loaded_models = ["llama3-8b-chat"]
            self._nodes["node_a"].vram_used_gb = 18
            for _ in range(5):
                self._add_request("chat", "best_effort")

        elif self.task_name == "task_medium":
            # Node B has OOM'd. Node A is healthy with chat model. Queue backing up.
            self._nodes["node_a"].loaded_models = ["llama3-8b-chat"]
            self._nodes["node_a"].vram_used_gb = 18
            self._nodes["node_b"].status = "oom_crashed"
            self._nodes["node_b"].loaded_models = []
            self._nodes["node_b"].vram_used_gb = 0
            for _ in range(3):
                self._add_request("chat", "premium")
            for _ in range(4):
                self._add_request("chat", "best_effort")

        elif self.task_name == "task_hard":
            # All nodes full of chat models. Influx of premium code requests.
            for node_id in ["node_a", "node_b"]:
                self._nodes[node_id].loaded_models = ["llama3-8b-chat", "mistral-7b-sum"]
                self._nodes[node_id].vram_used_gb = 34
            self._nodes["node_c"].loaded_models = ["llama3-8b-chat"]
            self._nodes["node_c"].vram_used_gb = 18
            for _ in range(4):
                self._add_request("code", "premium")
            for _ in range(3):
                self._add_request("chat", "best_effort")

    def _inject_requests(self):
        """Inject new requests each step based on task."""
        if len(self._request_queue) >= MAX_QUEUE_SIZE:
            return
        if self.task_name == "task_easy":
            pass  # Fixed queue  no new injections
        elif self.task_name == "task_medium":
            if self._step % 3 == 0:
                self._add_request("chat", "premium")
        elif self.task_name == "task_hard":
            if self._step % 2 == 0:
                tier = "premium" if self.rng.random() < 0.4 else "best_effort"
                task_type = self.rng.choice(["chat", "code"])
                self._add_request(task_type, tier)

    def _add_request(self, task_type: str, sla_tier: str):
        self._request_counter += 1
        self._request_queue.append(IncomingRequest(
            request_id=f"req_{self._request_counter:04d}",
            task_type=task_type,
            sla_tier=sla_tier,
            required_model=TASK_MODEL_MAP[task_type],
        ))

    def _observe(self, last_action_result: str, reward: float, done: bool) -> LLMFleetObservation:
        return LLMFleetObservation(
            nodes=dict(self._nodes),
            request_queue=list(self._request_queue),
            step=self._step,
            step_budget=self.step_budget,
            last_action_result=last_action_result,
            sla_violations=self._sla_violations,
            requests_served=self._requests_served,
            done=done,
            reward=reward,
        )

