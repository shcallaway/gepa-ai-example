from __future__ import annotations
from typing import Mapping, Any
from collections import Counter
import re

from src.core.base_task import Evaluator, Example


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_answer(text: str) -> str:
    """Extract text after 'Answer:' prefix, or return full text as fallback."""
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()


def exact_match(example: Example, model_output: str) -> float:
    gold = normalize(example["expected_output"])
    pred = normalize(extract_answer(model_output))
    return 1.0 if gold == pred else 0.0


def f1_score(example: Example, model_output: str) -> float:
    gold_tokens = normalize(example["expected_output"]).split()
    pred_tokens = normalize(extract_answer(model_output)).split()

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


# Canonical export name for the primary metric used during GEPA optimization
primary_metric = exact_match


def get_evaluators() -> list[Evaluator]:
    return [
        Evaluator(name="exact_match", metric_fn=exact_match, weight=1.0),
        Evaluator(name="f1", metric_fn=f1_score, weight=1.0),
    ]
