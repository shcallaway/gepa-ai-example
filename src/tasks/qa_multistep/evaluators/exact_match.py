from __future__ import annotations

from src.core.base_task import Example
from ._utils import normalize, extract_answer


def exact_match(example: Example, model_output: str) -> float:
    gold = normalize(example["answer"])
    pred = normalize(extract_answer(model_output))
    return 1.0 if gold == pred else 0.0
