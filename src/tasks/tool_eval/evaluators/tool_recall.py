from __future__ import annotations

from src.core.base_task import Example
from ._utils import parse_output


def tool_selection_recall(example: Example, model_output: str) -> float:
    """Fraction of expected tools that were actually used."""
    data = parse_output(model_output)
    expected_tools = set(example.get("expected_tools", []))

    if not expected_tools:
        return 1.0  # No expected tools = automatic pass

    used_tools = {tc["name"] for tc in data.get("tool_calls", [])}
    matched = len(expected_tools & used_tools)
    return matched / len(expected_tools)
