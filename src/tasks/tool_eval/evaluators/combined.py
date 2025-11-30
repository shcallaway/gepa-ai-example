from __future__ import annotations

from src.core.base_task import Example
from .answer_f1 import answer_f1
from .tool_recall import tool_selection_recall
from .tool_precision import tool_selection_precision
from .efficiency import tool_call_efficiency


def combined_score(example: Example, model_output: str) -> float:
    """Weighted combination of answer quality and tool usage metrics."""
    answer_weight = 0.4
    tool_recall_weight = 0.3
    tool_precision_weight = 0.2
    efficiency_weight = 0.1

    return (
        answer_weight * answer_f1(example, model_output)
        + tool_recall_weight * tool_selection_recall(example, model_output)
        + tool_precision_weight * tool_selection_precision(example, model_output)
        + efficiency_weight * tool_call_efficiency(example, model_output)
    )
