# Core module for GEPA multi-agent optimization
from .base_task import Task, SimpleTask, Evaluator, Example, MetricFn
from .dataset import load_dataset
from .registry import get_task, TASK_REGISTRY
from .runner import run_gepa_for_task

__all__ = [
    "Task",
    "SimpleTask",
    "Evaluator",
    "Example",
    "MetricFn",
    "load_dataset",
    "get_task",
    "TASK_REGISTRY",
    "run_gepa_for_task",
]
