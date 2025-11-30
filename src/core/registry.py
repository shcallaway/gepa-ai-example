from typing import Dict, Callable
from src.core.base_task import Task

from src.tasks.math_aime.task import TaskImpl as MathAimeTask
from src.tasks.qa_multistep.task import TaskImpl as MultiStepQATask

TASK_REGISTRY: Dict[str, Callable[[], Task]] = {
    "math_aime": MathAimeTask,
    "qa_multistep": MultiStepQATask,
}


def get_task(task_name: str) -> Task:
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        available = ", ".join(TASK_REGISTRY.keys())
        raise SystemExit(f"Unknown task '{task_name}'. Available: {available}")
