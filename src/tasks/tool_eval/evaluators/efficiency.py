from __future__ import annotations

from src.core.base_task import Example
from ._utils import parse_output


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
