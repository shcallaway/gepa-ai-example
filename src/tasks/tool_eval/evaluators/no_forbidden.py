from __future__ import annotations

from src.core.base_task import Example
from ._utils import parse_output


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
