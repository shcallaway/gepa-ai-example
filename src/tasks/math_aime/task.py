from src.core.base_task import SimpleTask
from .evaluators import EVALUATORS, PRIMARY_METRIC
from .executor import executor

TaskImpl = SimpleTask(
    name="math_aime",
    data_path="data/math_aime.jsonl",
    system_prompt=(
        "You are a helpful assistant that solves AIME-style math problems. "
        "Given a question, think step-by-step and compute the answer. "
        "At the end of your response, output the final numerical answer "
        "in exactly the format '### <final answer>'."
    ),
    evaluators=EVALUATORS,
    primary_metric=PRIMARY_METRIC,
    executor=executor,
)
