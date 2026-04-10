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
    from ..tasks import normalize_task_name
except ImportError:
    from models import (
        LLMFleetAction, LLMFleetObservation, LLMFleetState,
        NodeState, IncomingRequest, ModelSpec
    )
    from tasks.definitions import normalize_task_name


#  Static catalogue of available models 

MODEL_CATALOGUE: Dict[str, ModelSpec] = {
    "llama3-8b-chat":    ModelSpec(name="llama3-8b-chat",    vram_gb=18, cost_per_step=0.10),
    "llama3-70b-chat":   ModelSpec(name="llama3-70b-chat",   vram_gb=45, cost_per_step=0.35),
    "codellama-34b":     ModelSpec(name="codellama-34b",     vram_gb=35, cost_per_step=0.25),
    "mistral-7b-sum":    ModelSpec(name="mistral-7b-sum",    vram_gb=16, cost_per_step=0.08),
    "code-lora-adapter": ModelSpec(name="code-lora-adapter", vram_gb=8,  cost_per_step=0.05),
}

# Task type -> required model mapping
TASK_MODEL_MAP: Dict[str, str] = {
    "chat":      "llama3-8b-chat",
    "code":      "codellama-34b",
    "summarize": "mistral-7b-sum",
}

# Step budgets per task — must match openenv.yaml
TASK_STEP_BUDGETS = {"easy": 5, "medium": 10, "hard": 30, "loghaul": 50}

SLA_THRESHOLD = 5     # steps before a premium request triggers a violation
MAX_QUEUE_SIZE = 20


def _format_nl_observation(
    nodes: Dict[str, NodeState],
    request_queue: List[IncomingRequest],
    step: int,
    step_budget: int,
    sla_violations: int,
    requests_served: int,
    last_action_result: str,
    reward: float,
    done: bool,
) -> str:
    """
    Format cluster state as a natural-language status report so that
    a general-purpose LLM agent can reason about it using language
    understanding rather than raw JSON parsing.
    """
    lines = []
    lines.append(f"=== CLUSTER STATUS REPORT — Step {step}/{step_budget} ===")
    lines.append("")

    # Node status
    lines.append("NODES:")
    for node_id, node in nodes.items():
        vram_free = node.vram_total_gb - node.vram_used_gb
        models_str = ", ".join(node.loaded_models) if node.loaded_models else "none"
        if node.status == "oom_crashed":
            status_str = "WARNING: OOM CRASHED — restart_node required before use"
        elif node.status == "loading":
            status_str = f"LOADING — ready in {node.load_latency_remaining} step(s), do NOT route yet"
        else:
            status_str = "healthy"
        lines.append(
            f"  {node_id}: {node.vram_used_gb}/{node.vram_total_gb} GB VRAM used "
            f"({vram_free} GB free) | models: [{models_str}] | {status_str}"
        )

    lines.append("")

    # Request queue
    premium = [r for r in request_queue if r.sla_tier == "premium"]
    best_effort = [r for r in request_queue if r.sla_tier == "best_effort"]
    lines.append(f"REQUEST QUEUE ({len(request_queue)} total, {len(premium)} premium, {len(best_effort)} best-effort):")

    if not request_queue:
        lines.append("  Queue is empty.")
    else:
        for req in sorted(premium, key=lambda r: -r.age_steps):
            steps_until_breach = max(0, SLA_THRESHOLD - req.age_steps)
            if steps_until_breach <= 2:
                breach_str = f" — SLA BREACH IN {steps_until_breach} STEP(S)!"
            else:
                breach_str = f" (breaches in {steps_until_breach} steps)"
            lines.append(
                f"  [PREMIUM] {req.request_id}: needs {req.required_model} | age={req.age_steps} steps{breach_str}"
            )
        for req in sorted(best_effort, key=lambda r: -r.age_steps):
            lines.append(
                f"  [best-effort] {req.request_id}: needs {req.required_model} | age={req.age_steps} steps"
            )

    lines.append("")
    lines.append(f"STATS: served={requests_served} | sla_violations={sla_violations} | last_reward={reward:+.2f}")
    lines.append(f"LAST ACTION RESULT: {last_action_result}")

    if done:
        lines.append("")
        lines.append("Episode complete.")
    else:
        lines.append("")
        lines.append("ACTION REQUIRED: Choose one of: route_batch, load_model, evict_model, restart_node, noop.")
        lines.append(
            'Respond with JSON: {"action_type": "...", "target_node": "...", "model_name": "...", "request_ids": [...]}'
        )

    return "\n".join(lines)


class LLMFleetEnvironment(Environment):
    """
    Simulated LLM inference cluster SRE environment.

    The agent manages 3 nodes (80GB, 80GB, 40GB VRAM) and must:
    - Route requests to nodes with the required model loaded
    - Load/evict models to manage VRAM
    - Recover OOM-crashed nodes
    - Minimize latency and SLA violations to maximize reward

    Observations are returned as natural-language status reports so that
    general-purpose LLM agents can reason using language understanding.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "easy", step_budget: int = 5, seed: Optional[int] = None):
        self.task_name = normalize_task_name(task_name)
        self.step_budget = TASK_STEP_BUDGETS.get(self.task_name, step_budget)
        self.rng = random.Random(seed)
        self._last_reward = 0.0
        self._reset_internal()

    def _reset_internal(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._step = 0
        self._sla_violations = 0
        self._requests_served = 0
        self._requests_failed = 0
        self._cumulative_reward = 0.0
        self._request_counter = 0
        self._last_reward = 0.0

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
            self.task_name = normalize_task_name(task_name)
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_internal()
        # FIX 6: per-task step budgets matching openenv.yaml
        self.step_budget = TASK_STEP_BUDGETS.get(self.task_name, 30)
        self._setup_for_task()
        return self._observe("Episode started. Analyze the cluster and decide your first action.", reward=0.0, done=False)

    def step(self, action: LLMFleetAction) -> LLMFleetObservation:  # type: ignore[override]
        """Execute one environment step."""
        self._step += 1
        self._state.step_count = self._step
        reward = 0.0

        # Age all waiting requests
        for req in self._request_queue:
            req.age_steps += 1

        # Advance pending model loads (stochastic 1-3 step latency)
        for node in self._nodes.values():
            if node.load_latency_remaining > 0:
                node.load_latency_remaining -= 1
                if node.load_latency_remaining == 0:
                    node.status = "healthy"

        # Execute agent action
        result_msg, action_reward = self._execute_action(action)
        reward += action_reward

        # Cost penalty: penalize expensive loaded models when cheaper ones could serve
        for node in self._nodes.values():
            if node.status == "healthy":
                for model_name in node.loaded_models:
                    spec = MODEL_CATALOGUE[model_name]
                    can_serve_with_cheaper = any(
                        req.required_model != model_name
                        and MODEL_CATALOGUE.get(req.required_model, spec).cost_per_step < spec.cost_per_step
                        for req in self._request_queue
                    )
                    if can_serve_with_cheaper:
                        reward -= 0.02 * spec.cost_per_step

        # SLA violation check
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

        # Inject new requests before done check
        self._inject_requests()

        self._cumulative_reward += reward
        self._last_reward = reward

        done = self._step >= self.step_budget
        if self.task_name != "loghaul" and len(self._request_queue) == 0:
            done = True

        return self._observe(result_msg, reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    def get_full_state(self) -> LLMFleetState:
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
            return "No action taken this step.", reward

        node_id = action.target_node
        if node_id and node_id not in self._nodes:
            return f"Unknown node: {node_id}. Valid nodes: node_a, node_b, node_c.", reward - 0.1

        node = self._nodes.get(node_id) if node_id else None

        if at == "route_batch":
            return self._route_batch(node, node_id, action.request_ids)
        elif at == "load_model":
            return self._load_model(node, node_id, action.model_name)
        elif at == "evict_model":
            return self._evict_model(node, node_id, action.model_name)
        elif at == "restart_node":
            return self._restart_node(node, node_id)

        return "Unknown action type.", reward

    def _route_batch(self, node, node_id, request_ids):
        if not node or node.status != "healthy":
            return f"Cannot route to {node_id}: node is not healthy (status={node.status if node else 'unknown'}).", -0.1
        if node.load_latency_remaining > 0:
            return f"Cannot route to {node_id}: still loading, ready in {node.load_latency_remaining} step(s).", -0.1
        if not request_ids:
            return "No request_ids specified for route_batch.", -0.05

        served = 0
        failed = []
        for rid in request_ids:
            req = next((r for r in self._request_queue if r.request_id == rid), None)
            if not req:
                continue
            if req.required_model not in node.loaded_models:
                failed.append(f"{rid} needs {req.required_model} (not loaded on {node_id})")
                continue
            self._request_queue.remove(req)
            self._requests_served += 1
            served += 1

        reward = 0.2 * served
        msg = f"Routed {served} request(s) via {node_id}."
        if failed:
            msg += f" Failed: {'; '.join(failed)}."
        return msg, reward

    def _load_model(self, node, node_id, model_name):
        if not model_name or model_name not in MODEL_CATALOGUE:
            available = ", ".join(MODEL_CATALOGUE.keys())
            return f"Unknown model: {model_name}. Available: {available}.", -0.05
        if not node or node.status == "oom_crashed":
            return f"Node {node_id} is OOM-crashed. Use restart_node first.", -0.05
        if model_name in node.loaded_models:
            return f"{model_name} is already loaded on {node_id}.", 0.0

        spec = MODEL_CATALOGUE[model_name]
        vram_free = node.vram_total_gb - node.vram_used_gb
        if node.vram_used_gb + spec.vram_gb > node.vram_total_gb:
            node.status = "oom_crashed"
            return (
                f"OOM CRASH on {node_id}! Tried to load {model_name} ({spec.vram_gb} GB) "
                f"but only {vram_free} GB free. Node crashed — use restart_node to recover."
            ), -0.5

        latency = self.rng.randint(1, 3)
        node.loaded_models.append(model_name)
        node.vram_used_gb += spec.vram_gb
        node.status = "loading"
        node.load_latency_remaining = latency
        return (
            f"Loading {model_name} ({spec.vram_gb} GB) onto {node_id}. "
            f"Will be ready in {latency} step(s). Do not route to it yet."
        ), 0.0

    def _evict_model(self, node, node_id, model_name):
        if not model_name:
            return "No model_name specified for evict_model.", -0.05
        if not node or node.status == "oom_crashed":
            return f"Node {node_id} is OOM-crashed. Restart it first.", -0.05
        if model_name not in node.loaded_models:
            return f"{model_name} is not loaded on {node_id}. Nothing to evict.", -0.05

        spec = MODEL_CATALOGUE[model_name]
        node.loaded_models.remove(model_name)
        node.vram_used_gb = max(0, node.vram_used_gb - spec.vram_gb)
        vram_free = node.vram_total_gb - node.vram_used_gb
        return (
            f"Evicted {model_name} from {node_id}. Freed {spec.vram_gb} GB. "
            f"Now {vram_free} GB free on {node_id}."
        ), 0.0

    def _restart_node(self, node, node_id):
        if not node:
            return "No target_node specified for restart_node.", -0.05
        old_status = node.status
        node.status = "loading"
        node.loaded_models = []
        node.vram_used_gb = 0
        node.load_latency_remaining = 2
        return (
            f"Restarting {node_id} (was: {old_status}). All models cleared. "
            f"Node will be healthy in 2 steps — do not route to it until then."
        ), 0.0

    #  Task Setup 

    def _setup_for_task(self):
        """Pre-load models and seed queue based on task difficulty."""
        if self.task_name == "easy":
            self._nodes["node_a"].loaded_models = ["llama3-8b-chat"]
            self._nodes["node_a"].vram_used_gb = 18
            for _ in range(5):
                self._add_request("chat", "best_effort")

        elif self.task_name == "medium":
            self._nodes["node_a"].loaded_models = ["llama3-8b-chat"]
            self._nodes["node_a"].vram_used_gb = 18
            self._nodes["node_b"].status = "oom_crashed"
            self._nodes["node_b"].loaded_models = []
            self._nodes["node_b"].vram_used_gb = 0
            for _ in range(3):
                self._add_request("chat", "premium")
            for _ in range(4):
                self._add_request("chat", "best_effort")

        elif self.task_name == "hard":
            chat_models = ["llama3-8b-chat", "llama3-70b-chat"]
            big_model = self.rng.choice(chat_models)
            self._nodes["node_a"].loaded_models = [big_model]
            self._nodes["node_a"].vram_used_gb = MODEL_CATALOGUE[big_model].vram_gb
            self._nodes["node_b"].loaded_models = ["llama3-8b-chat", "mistral-7b-sum"]
            self._nodes["node_b"].vram_used_gb = 34
            if self.rng.random() < 0.5:
                self._nodes["node_c"].loaded_models = ["mistral-7b-sum"]
                self._nodes["node_c"].vram_used_gb = 16
            else:
                self._nodes["node_c"].loaded_models = []
                self._nodes["node_c"].vram_used_gb = 0
            code_count = self.rng.randint(3, 6)
            chat_count = self.rng.randint(2, 5)
            for _ in range(code_count):
                tier = "premium" if self.rng.random() < 0.7 else "best_effort"
                self._add_request("code", tier)
            for _ in range(chat_count):
                tier = "premium" if self.rng.random() < 0.5 else "best_effort"
                self._add_request("chat", tier)

        elif self.task_name == "loghaul":
            self._nodes["node_a"].loaded_models = ["llama3-8b-chat"]
            self._nodes["node_a"].vram_used_gb = 18
            self._nodes["node_b"].loaded_models = ["mistral-7b-sum"]
            self._nodes["node_b"].vram_used_gb = 16
            self._nodes["node_c"].loaded_models = []
            self._nodes["node_c"].vram_used_gb = 0
            for _ in range(2):
                self._add_request("code", "premium")
            for _ in range(2):
                self._add_request("chat", "premium")
            for _ in range(2):
                self._add_request("summarize", "best_effort")

    def _inject_requests(self):
        """Inject new requests each step based on task. FIX 1: canonical task names."""
        if len(self._request_queue) >= MAX_QUEUE_SIZE:
            return

        if self._step < 10:
            arrival_rate = 0.3
        elif self._step < 25:
            arrival_rate = 0.8
        else:
            arrival_rate = 0.3

        # FIX 1: was "task_easy", "task_medium", "task_hard", "task_longhaul"
        if self.task_name == "easy":
            return  # fixed queue, no new arrivals

        elif self.task_name == "medium":
            if self.rng.random() < arrival_rate:
                tier = "premium" if self.rng.random() < 0.5 else "best_effort"
                self._add_request("chat", tier)

        elif self.task_name == "hard":
            if self.rng.random() < arrival_rate:
                if self.rng.random() < 0.5:
                    loaded_models = set()
                    for node in self._nodes.values():
                        loaded_models.update(node.loaded_models)
                    all_task_models = {"llama3-8b-chat", "codellama-34b", "mistral-7b-sum"}
                    not_loaded = list(all_task_models - loaded_models)
                    if not_loaded:
                        needed_model = self.rng.choice(not_loaded)
                        task_matches = [t for t, m in TASK_MODEL_MAP.items() if m == needed_model]
                        if task_matches:
                            tier = "premium" if self.rng.random() < 0.6 else "best_effort"
                            self._add_request(task_matches[0], tier)
                            return
                tier = "premium" if self.rng.random() < 0.5 else "best_effort"
                task_type = self.rng.choice(["chat", "code"])
                self._add_request(task_type, tier)

        elif self.task_name == "loghaul":
            if self._step < 15:
                rate = 0.35
                premium_prob = 0.2
            elif self._step < 35:
                rate = 0.95
                premium_prob = 0.7
            else:
                rate = 0.3
                premium_prob = 0.2

            if self.rng.random() < rate:
                arrivals = 1
                if 15 <= self._step < 35:
                    arrivals = 2
                    if self.rng.random() < 0.55:
                        arrivals += 1
                for _ in range(arrivals):
                    tier = "premium" if self.rng.random() < premium_prob else "best_effort"
                    if 15 <= self._step < 35:
                        roll = self.rng.random()
                        if roll < 0.55:
                            task_type = "code"
                        elif roll < 0.85:
                            task_type = "chat"
                        else:
                            task_type = "summarize"
                    else:
                        task_type = self.rng.choice(["chat", "code", "summarize"])
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
        nl_report = _format_nl_observation(
            nodes=self._nodes,
            request_queue=self._request_queue,
            step=self._step,
            step_budget=self.step_budget,
            sla_violations=self._sla_violations,
            requests_served=self._requests_served,
            last_action_result=last_action_result,
            reward=reward,
            done=done,
        )
        return LLMFleetObservation(
            nodes=dict(self._nodes),
            request_queue=list(self._request_queue),
            step=self._step,
            step_budget=self.step_budget,
            last_action_result=nl_report,
            sla_violations=self._sla_violations,
            requests_served=self._requests_served,
            done=done,
            reward=reward,
        )