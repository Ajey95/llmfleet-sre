"""Compatibility shim — re-exports task definitions and graders from the tasks package."""

from __future__ import annotations

try:
    from ..tasks import TASKS, TASK_METADATA, grade
except ImportError:
    from tasks import TASKS, TASK_METADATA, grade

__all__ = ["TASKS", "TASK_METADATA", "grade"]
