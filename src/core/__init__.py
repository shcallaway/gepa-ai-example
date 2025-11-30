# Core module for GEPA multi-agent optimization
from .base_task import Task, Evaluator, Example, MetricFn
from .registry import get_task, TASK_REGISTRY
from .runner import run_gepa_for_task

__all__ = [
    "Task",
    "Evaluator",
    "Example",
    "MetricFn",
    "get_task",
    "TASK_REGISTRY",
    "run_gepa_for_task",
]
