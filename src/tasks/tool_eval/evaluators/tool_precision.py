from __future__ import annotations

from src.core.base_task import Example
from ._utils import parse_output


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
