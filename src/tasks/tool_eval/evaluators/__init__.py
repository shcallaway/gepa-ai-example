"""Evaluators for tool-calling behavior.

Metrics for assessing both final answer quality and tool usage patterns.
"""

from src.core.base_task import Evaluator
from .answer_f1 import answer_f1
from .tool_recall import tool_selection_recall
from .tool_precision import tool_selection_precision
from .no_forbidden import no_forbidden_tools
from .efficiency import tool_call_efficiency
from .combined import combined_score

EVALUATORS = [
    Evaluator(name="answer_f1", metric_fn=answer_f1, weight=1.0),
    Evaluator(name="tool_recall", metric_fn=tool_selection_recall, weight=1.0),
    Evaluator(name="tool_precision", metric_fn=tool_selection_precision, weight=1.0),
    Evaluator(name="no_forbidden", metric_fn=no_forbidden_tools, weight=1.0),
    Evaluator(name="efficiency", metric_fn=tool_call_efficiency, weight=1.0),
    Evaluator(name="combined", metric_fn=combined_score, weight=1.0),
]

PRIMARY_METRIC = combined_score
