from src.core.base_task import SimpleTask
from .evaluators import EVALUATORS, PRIMARY_METRIC
from .executor import executor

TaskImpl = SimpleTask(
    name="qa_multistep",
    data_path="data/qa_multistep.jsonl",
    system_prompt=(
        "You are an AI assistant that answers complex multi-step questions. "
        "Carefully reason through the question step-by-step, referencing "
        "relevant facts as needed, and then provide a concise final answer "
        "at the end of your response, prefixed with 'Answer:'."
    ),
    evaluators=EVALUATORS,
    primary_metric=PRIMARY_METRIC,
    executor=executor,
)
