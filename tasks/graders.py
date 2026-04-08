"""Task graders for LLMFleet-SRE."""

from __future__ import annotations

try:
    from ..models import LLMFleetState
except ImportError:
    from models import LLMFleetState

STRICT_MIN_SCORE = 0.01
STRICT_MAX_SCORE = 0.99


def _strict_open_unit(score: float) -> float:
    """Clamp score to strict open interval (0, 1)."""
    return min(STRICT_MAX_SCORE, max(STRICT_MIN_SCORE, float(score)))


def grade(task_name: str, final_state: LLMFleetState) -> float:
    """Score the agent's performance on a completed episode."""
    if task_name == "task_easy":
        raw = _grade_easy(final_state)
    elif task_name == "task_medium":
        raw = _grade_medium(final_state)
    elif task_name == "task_hard":
        raw = _grade_hard(final_state)
    elif task_name == "task_longhaul":
        raw = _grade_longhaul(final_state)
    else:
        raw = 0.0
    return _strict_open_unit(raw)


def _grade_easy(state: LLMFleetState) -> float:
    any_crashed = any(n.status == "oom_crashed" for n in state.nodes.values())
    if any_crashed:
        return 0.0
    total = 5
    return min(state.requests_served / total, 1.0)


def _grade_medium(state: LLMFleetState) -> float:
    queue_cleared = len(state.request_queue) == 0
    if not queue_cleared:
        return max(0.0, state.requests_served / 7)
    speed_score = 1.0 - (state.step / state.step_budget) * 0.5
    sla_penalty = min(state.sla_violations * 0.1, 0.4)
    return max(0.0, speed_score - sla_penalty)


def _grade_hard(state: LLMFleetState) -> float:
    total_seen = state.requests_served + len(state.request_queue) + state.requests_failed
    if total_seen == 0:
        return 0.0

    fraction_served = state.requests_served / total_seen
    sla_penalty = min(state.sla_violations * 0.18, 0.55)
    oom_penalty = 0.25 if any(n.status == "oom_crashed" for n in state.nodes.values()) else 0.0

    return max(0.0, min(fraction_served - sla_penalty - oom_penalty, 1.0))


def _grade_longhaul(state: LLMFleetState) -> float:
    any_crashed = any(n.status == "oom_crashed" for n in state.nodes.values())
    total_seen = state.requests_served + len(state.request_queue) + state.requests_failed
    if total_seen == 0:
        return 0.0

    throughput_score = state.requests_served / total_seen
    sla_penalty = min(state.sla_violations * 0.07, 0.6)
    backlog_penalty = min((len(state.request_queue) / 20.0) * 0.4, 0.4)
    oom_penalty = 0.3 if any_crashed else 0.0

    return max(0.0, min(throughput_score - sla_penalty - backlog_penalty - oom_penalty, 1.0))
