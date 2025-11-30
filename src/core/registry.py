from typing import Dict
from src.core.base_task import Task

from src.tasks.math_aime.task import TaskImpl as MathAimeTask
from src.tasks.qa_multistep.task import TaskImpl as MultiStepQATask
from src.tasks.qa_langgraph.task import TaskImpl as QALangGraphTask
from src.tasks.open_qa.task import TaskImpl as OpenQATask

TASK_REGISTRY: Dict[str, Task] = {
    "math_aime": MathAimeTask,
    "qa_multistep": MultiStepQATask,
    "qa_langgraph": QALangGraphTask,
    "open_qa": OpenQATask,
}


def get_task(task_name: str) -> Task:
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        available = ", ".join(TASK_REGISTRY.keys())
        raise SystemExit(f"Unknown task '{task_name}'. Available: {available}")
