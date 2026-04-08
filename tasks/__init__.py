"""Canonical task definitions and graders for LLMFleet-SRE."""

from .definitions import TASKS, TASK_METADATA, normalize_task_name
from .graders import grade

__all__ = ["TASKS", "TASK_METADATA", "normalize_task_name", "grade"]
