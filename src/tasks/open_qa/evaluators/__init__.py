"""Evaluators for the open_qa task.

Demonstrates LLM-as-judge evaluation for open-ended questions where
exact matching is insufficient.
"""

from src.core.base_task import Evaluator
from .correctness import llm_judge_correctness
from .helpfulness import llm_judge_helpfulness

EVALUATORS = [
    Evaluator(name="correctness", metric_fn=llm_judge_correctness, weight=1.0),
    Evaluator(name="helpfulness", metric_fn=llm_judge_helpfulness, weight=0.5),
]

PRIMARY_METRIC = llm_judge_correctness
