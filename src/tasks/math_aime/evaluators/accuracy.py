from __future__ import annotations

from src.core.base_task import Example
from ._utils import extract_answer


def accuracy_metric(example: Example, model_output: str) -> float:
    gold = str(example["answer"]).strip()
    pred = extract_answer(model_output)
    return 1.0 if pred == gold else 0.0
