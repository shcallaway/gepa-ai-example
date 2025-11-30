from __future__ import annotations
from typing import Sequence, Any
from src.core.base_task import Task, Evaluator, Example
from .dataset import load_dataset
from .evaluators import get_evaluators, primary_metric


class MultiStepQATask(Task):
    name = "qa_multistep"

    def load_datasets(
        self,
    ) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
        return load_dataset()

    def make_seed_candidate(self) -> dict:
        return {
            "system_prompt": (
                "You are an AI assistant that answers complex multi-step questions. "
                "Carefully reason through the question step-by-step, referencing "
                "relevant facts as needed, and then provide a concise final answer "
                "at the end of your response, prefixed with 'Answer:'."
            )
        }

    def get_evaluators(self) -> list[Evaluator]:
        return get_evaluators()

    def get_gepa_kwargs(self) -> dict[str, Any]:
        # Use our answer-extracting metric during optimization
        return {"metric": primary_metric}


# Canonical export name for the task class
TaskImpl = MultiStepQATask
