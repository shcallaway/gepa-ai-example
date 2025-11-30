from __future__ import annotations
import re

from src.core.base_task import Evaluator, Example

ANSWER_REGEX = re.compile(r"###\s*([0-9\-\.]+)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_REGEX.search(text)
    return m.group(1).strip() if m else None


def accuracy_metric(example: Example, model_output: str) -> float:
    gold = str(example["expected_output"]).strip()
    pred = extract_answer(model_output)
    return 1.0 if pred == gold else 0.0


EVALUATORS = [
    Evaluator(name="accuracy", metric_fn=accuracy_metric, weight=1.0),
]

PRIMARY_METRIC = accuracy_metric
