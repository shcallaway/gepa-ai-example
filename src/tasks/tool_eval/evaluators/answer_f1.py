from __future__ import annotations
from collections import Counter

from src.core.base_task import Example
from ._utils import parse_output, normalize


def answer_f1(example: Example, model_output: str) -> float:
    """Token-level F1 between expected and actual answer."""
    data = parse_output(model_output)
    gold_tokens = normalize(example["answer"]).split()
    pred_tokens = normalize(data.get("answer", "")).split()

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    common = sum((gold_counts & pred_counts).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
