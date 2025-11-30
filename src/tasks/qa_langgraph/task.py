"""Task definition for qa_langgraph.

This task demonstrates using a remote LangGraph agent as the executor.
The agent runs on http://localhost:2024 and is invoked via HTTP calls.
"""

from pathlib import Path

from src.core.base_task import SimpleTask
from .evaluators import EVALUATORS, PRIMARY_METRIC
from .executor import executor

_TASK_DIR = Path(__file__).parent
_PROMPT = (_TASK_DIR / "prompt.txt").read_text().strip()

TaskImpl = SimpleTask(
    name="qa_langgraph",
    data_path=str(_TASK_DIR / "data.jsonl"),
    system_prompt=_PROMPT,
    evaluators=EVALUATORS,
    primary_metric=PRIMARY_METRIC,
    executor=executor,
)
