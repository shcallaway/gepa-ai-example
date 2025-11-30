"""Evaluators for tool-calling behavior.

Metrics for assessing both final answer quality and tool usage patterns.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from src.core.base_task import Evaluator, Example


def parse_output(model_output: str) -> dict:
    """Parse the structured JSON output from the executor."""
    try:
        return json.loads(model_output)
    except json.JSONDecodeError:
        return {"answer": model_output, "tool_calls": []}


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


# --- Answer Quality Metrics ---


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


# --- Tool Selection Metrics ---


def tool_selection_recall(example: Example, model_output: str) -> float:
    """Fraction of expected tools that were actually used."""
    data = parse_output(model_output)
    expected_tools = set(example.get("expected_tools", []))

    if not expected_tools:
        return 1.0  # No expected tools = automatic pass

    used_tools = {tc["name"] for tc in data.get("tool_calls", [])}
    matched = len(expected_tools & used_tools)
    return matched / len(expected_tools)


def tool_selection_precision(example: Example, model_output: str) -> float:
    """Fraction of used tools that were expected (penalizes unnecessary tools)."""
    data = parse_output(model_output)
    expected_tools = set(example.get("expected_tools", []))
    used_tools = {tc["name"] for tc in data.get("tool_calls", [])}

    if not used_tools:
        return 1.0 if not expected_tools else 0.0

    if not expected_tools:
        return 0.0  # Used tools when none expected

    matched = len(expected_tools & used_tools)
    return matched / len(used_tools)


def no_forbidden_tools(example: Example, model_output: str) -> float:
    """Return 1.0 if no forbidden tools were used, 0.0 otherwise."""
    data = parse_output(model_output)
    forbidden = set(example.get("forbidden_tools", []))

    if not forbidden:
        return 1.0

    used_tools = {tc["name"] for tc in data.get("tool_calls", [])}
    if used_tools & forbidden:
        return 0.0
    return 1.0


# --- Tool Efficiency Metrics ---


def tool_call_efficiency(example: Example, model_output: str) -> float:
    """Score based on staying within expected tool call budget."""
    data = parse_output(model_output)
    num_calls = len(data.get("tool_calls", []))
    max_calls = example.get("max_tool_calls", 3)

    if num_calls <= max_calls:
        return 1.0
    # Penalize 0.2 per extra call
    penalty = (num_calls - max_calls) * 0.2
    return max(0.0, 1.0 - penalty)


def used_any_tools(example: Example, model_output: str) -> float:
    """Return 1.0 if at least one tool was used when tools were expected."""
    data = parse_output(model_output)
    expected_tools = example.get("expected_tools", [])

    if not expected_tools:
        return 1.0  # No tools expected

    tool_calls = data.get("tool_calls", [])
    return 1.0 if tool_calls else 0.0


# --- Combined Metric ---


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


EVALUATORS = [
    Evaluator(name="answer_f1", metric_fn=answer_f1, weight=1.0),
    Evaluator(name="tool_recall", metric_fn=tool_selection_recall, weight=1.0),
    Evaluator(name="tool_precision", metric_fn=tool_selection_precision, weight=1.0),
    Evaluator(name="no_forbidden", metric_fn=no_forbidden_tools, weight=1.0),
    Evaluator(name="efficiency", metric_fn=tool_call_efficiency, weight=1.0),
    Evaluator(name="combined", metric_fn=combined_score, weight=1.0),
]

PRIMARY_METRIC = combined_score
