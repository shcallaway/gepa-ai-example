from __future__ import annotations
from typing import Sequence, Any
from src.core.base_task import Task, Evaluator, Example
from .dataset import load_dataset
from .evaluators import get_evaluators, primary_metric


class MathAimeTask(Task):
    name = "math_aime"

    def load_datasets(
        self,
    ) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
        return load_dataset()

    def make_seed_candidate(self) -> dict:
        return {
            "system_prompt": (
                "You are a helpful assistant that solves AIME-style math problems. "
                "Given a question, think step-by-step and compute the answer. "
                "At the end of your response, output the final numerical answer "
                "in exactly the format '### <final answer>'."
            )
        }

    def get_evaluators(self) -> list[Evaluator]:
        return get_evaluators()

    def get_gepa_kwargs(self) -> dict[str, Any]:
        # Use our parsing-aware metric during optimization
        return {"metric": primary_metric}


# Canonical export name for the task class
TaskImpl = MathAimeTask
