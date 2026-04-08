"""Compatibility shim — re-exports task definitions and graders from the tasks package."""

from __future__ import annotations

try:
    from ..tasks import TASKS, TASK_METADATA, grade
except ImportError:
    from tasks.definitions import TASKS, TASK_METADATA
    from tasks.graders import grade

__all__ = ["TASKS", "TASK_METADATA", "grade"]
