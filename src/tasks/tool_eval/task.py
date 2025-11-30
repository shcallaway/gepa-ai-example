"""Tool-calling evaluation task configuration."""

from pathlib import Path

from src.core.base_task import SimpleTask

from .evaluators import EVALUATORS, PRIMARY_METRIC
from .executor import executor

_TASK_DIR = Path(__file__).parent

TaskImpl = SimpleTask(
    name="tool_eval",
    data_path=str(_TASK_DIR / "data.jsonl"),
    system_prompt=(_TASK_DIR / "prompt.txt").read_text(),
    evaluators=EVALUATORS,
    primary_metric=PRIMARY_METRIC,
    executor=executor,
    train_frac=0.6,
    val_frac=0.2,
    seed=42,
)
