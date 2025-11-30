from src.core.base_task import Evaluator
from .exact_match import exact_match
from .f1 import f1_score

EVALUATORS = [
    Evaluator(name="exact_match", metric_fn=exact_match, weight=1.0),
    Evaluator(name="f1", metric_fn=f1_score, weight=1.0),
]

PRIMARY_METRIC = exact_match
