"""Task definition for qa_langgraph.

This task demonstrates using a remote LangGraph agent as the executor.
The agent runs on http://localhost:2024 and is invoked via HTTP calls.
"""

from src.core.base_task import SimpleTask
from .evaluators import EVALUATORS, PRIMARY_METRIC
from .executor import executor


TaskImpl = SimpleTask(
    name="qa_langgraph",
    data_path="data/qa_langgraph.jsonl",
    system_prompt="""You are a helpful assistant that answers questions accurately and concisely.

When answering, always format your final response as:
Answer: <your answer>

For example:
Question: What is the capital of France?
Answer: Paris""",
    evaluators=EVALUATORS,
    primary_metric=PRIMARY_METRIC,
    executor=executor,
)
