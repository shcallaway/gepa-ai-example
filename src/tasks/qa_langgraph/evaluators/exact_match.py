from __future__ import annotations

from src.core.base_task import Example
from ._utils import normalize, extract_answer


def exact_match(example: Example, model_output: str) -> float:
    """Return 1.0 if normalized answer matches exactly, 0.0 otherwise."""
    gold = normalize(example["answer"])
    pred = normalize(extract_answer(model_output))
    return 1.0 if gold == pred else 0.0
