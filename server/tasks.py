"""
Task graders for LLMFleet-SRE.

Each task has a distinct grader that returns a float in [0.0, 1.0].
"""

from __future__ import annotations

try:
    from ..models import LLMFleetState
except ImportError:
    from models import LLMFleetState

TASKS = ["task_easy", "task_medium", "task_hard"]


def grade(task_name: str, final_state: LLMFleetState) -> float:
    """
    Score the agent's performance on a completed episode.
    Returns a float in [0.0, 1.0].
    """
    if task_name == "task_easy":
        return _grade_easy(final_state)
    elif task_name == "task_medium":
        return _grade_medium(final_state)
    elif task_name == "task_hard":
        return _grade_hard(final_state)
    return 0.0


def _grade_easy(state: LLMFleetState) -> float:
    """
    Task: 5 chat requests queued. Node A has model loaded.
    Score: 1.0 if all served with no OOM. 0.0 if any OOM.
    Partial: fraction of requests served.
    """
    any_crashed = any(n.status == "oom_crashed" for n in state.nodes.values())
    if any_crashed:
        return 0.0
    served = state.requests_served
    total = 5
    return min(served / total, 1.0)


def _grade_medium(state: LLMFleetState) -> float:
    """
    Task: Node B crashed. Recover and clear queue.
    Score: Decays based on latency. 1.0 if cleared fast. Partial for partial progress.
    """
    queue_cleared = len(state.request_queue) == 0
    if not queue_cleared:
        return max(0.0, state.requests_served / 7)  # partial credit
    # Bonus for speed
    speed_score = 1.0 - (state.step / state.step_budget) * 0.5
    sla_penalty = min(state.sla_violations * 0.1, 0.4)
    return max(0.0, speed_score - sla_penalty)


def _grade_hard(state: LLMFleetState) -> float:
    """
    Task: Evict chat models, load code model, serve mixed premium/best_effort queue.
    Full score only if both code AND chat queues cleared with <= 2 SLA violations.
    """
    total_initial = 7  # 4 code + 3 chat
    fraction_served = state.requests_served / total_initial
    sla_penalty = min(state.sla_violations * 0.15, 0.5)
    base = fraction_served - sla_penalty
    return max(0.0, min(base, 1.0))
