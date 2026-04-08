"""Evaluator-facing grader entrypoints for task-level reflection checks."""

from __future__ import annotations

from typing import Any, Optional

try:
    from ..models import LLMFleetState
    from ..tasks.definitions import normalize_task_name
    from ..tasks.graders import grade as grade_task
except ImportError:
    from models import LLMFleetState
    from tasks.definitions import normalize_task_name
    from tasks.graders import grade as grade_task


def _safe_default_score() -> float:
    return 0.5


def _coerce_state(obj: Any) -> Optional[LLMFleetState]:
    if isinstance(obj, LLMFleetState):
        return obj
    if isinstance(obj, dict):
        try:
            return LLMFleetState(**obj)
        except Exception:
            return None
    return None


def _extract_state(trajectory: Any = None, final_state: Any = None, **kwargs: Any) -> Optional[LLMFleetState]:
    state = _coerce_state(final_state)
    if state is not None:
        return state

    if isinstance(trajectory, dict):
        for key in ("final_state", "state", "observation", "last_observation"):
            state = _coerce_state(trajectory.get(key))
            if state is not None:
                return state

    for key in ("state", "observation", "last_observation"):
        state = _coerce_state(kwargs.get(key))
        if state is not None:
            return state

    return None


def _grade_task_name(task_name: str, trajectory: Any = None, final_state: Any = None, **kwargs: Any) -> float:
    state = _extract_state(trajectory=trajectory, final_state=final_state, **kwargs)
    if state is None:
        return _safe_default_score()
    try:
        return float(grade_task(normalize_task_name(task_name), state))
    except Exception:
        return _safe_default_score()


def easy_grader(trajectory: Any = None, final_state: Any = None, **kwargs: Any) -> float:
    return _grade_task_name("easy", trajectory=trajectory, final_state=final_state, **kwargs)


def medium_grader(trajectory: Any = None, final_state: Any = None, **kwargs: Any) -> float:
    return _grade_task_name("medium", trajectory=trajectory, final_state=final_state, **kwargs)


def hard_grader(trajectory: Any = None, final_state: Any = None, **kwargs: Any) -> float:
    return _grade_task_name("hard", trajectory=trajectory, final_state=final_state, **kwargs)


def loghaul_grader(trajectory: Any = None, final_state: Any = None, **kwargs: Any) -> float:
    return _grade_task_name("loghaul", trajectory=trajectory, final_state=final_state, **kwargs)

