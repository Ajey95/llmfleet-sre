"""
Data models for the LLMFleet-SRE Environment.

An LLM agent acts as an SRE for a simulated GPU inference cluster:
- 3 GPU nodes with VRAM budgets
- Incoming request queue with SLA tiers
- Agent must route, load, evict, and recover nodes
"""

from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


#  Literals 

NodeStatus = Literal["healthy", "oom_crashed", "loading"]
ActionType = Literal["route_batch", "load_model", "evict_model", "restart_node", "noop"]
TaskType   = Literal["chat", "code", "summarize"]
SlaTier    = Literal["premium", "best_effort"]


#  Sub-models 

class ModelSpec(BaseModel):
    """Static definition of an available model."""
    name: str
    vram_gb: int
    cost_per_step: float  # relative cost while loaded


class NodeState(BaseModel):
    """Runtime state of a single GPU node."""
    node_id: str
    vram_total_gb: int = 80
    vram_used_gb: int = 0
    loaded_models: List[str] = Field(default_factory=list)
    status: NodeStatus = "healthy"
    load_latency_remaining: int = 0


class IncomingRequest(BaseModel):
    """A single request waiting in the queue."""
    request_id: str
    task_type: TaskType
    sla_tier: SlaTier
    required_model: str
    age_steps: int = 0


#  Action 

class LLMFleetAction(Action):
    """
    One action the agent can take per step.

    action_type   What to do
    target_node   Which node to act on (required for all actions except noop)
    model_name    Which model to load/evict (required for load_model, evict_model)
    request_ids   Which requests to route (required for route_batch)
    """
    action_type: ActionType = Field(..., description="Type of action to take")
    target_node: Optional[str] = Field(None, description="Target node id (node_a | node_b | node_c)")
    model_name: Optional[str] = Field(None, description="Model name for load/evict actions")
    request_ids: List[str] = Field(default_factory=list, description="Request IDs to route")


#  Observation 

class LLMFleetObservation(Observation):
    """
    Full cluster snapshot returned after every step.
    This is what the agent sees.
    """
    nodes: Dict[str, NodeState] = Field(default_factory=dict, description="Current state of all nodes")
    request_queue: List[IncomingRequest] = Field(default_factory=list, description="Pending requests")
    step: int = Field(0, description="Current step number")
    step_budget: int = Field(30, description="Max steps in this episode")
    last_action_result: str = Field("", description="Human-readable result of the last action")
    sla_violations: int = Field(0, description="Cumulative premium requests that waited > 5 steps")
    requests_served: int = Field(0, description="Cumulative successful serves")
    done: bool = Field(False, description="Whether the episode is over")
    reward: float = Field(0.0, description="Reward from the last step")


#  State (for state() endpoint) 

class LLMFleetState(BaseModel):
    """Full serializable state for the state() endpoint."""
    nodes: Dict[str, NodeState]
    request_queue: List[IncomingRequest]
    step: int
    step_budget: int
    sla_violations: int
    requests_served: int
    requests_failed: int
    cumulative_reward: float
    task_name: str
    episode_id: str


#  Reward 

class LLMFleetReward(BaseModel):
    """Decomposed reward for transparency and debugging."""
    total: float
    throughput: float       # +0.2 per request served
    latency_penalty: float  # -0.01 per step per queued request
    idle_penalty: float     # -0.05 per step per idle healthy node with queue > 0
    oom_penalty: float      # -0.5 for triggering an OOM crash
    sla_penalty: float      # -0.3 per premium request waiting > 5 steps
    cost_penalty: float     # additional penalty for expensive loaded models

