from src.core.base_task import Evaluator
from .accuracy import accuracy_metric

EVALUATORS = [
    Evaluator(name="accuracy", metric_fn=accuracy_metric, weight=1.0),
]

PRIMARY_METRIC = accuracy_metric
