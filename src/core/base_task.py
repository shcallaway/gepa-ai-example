from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence, Mapping, Any, Callable

from .dataset import load_dataset

Example = Mapping[str, Any]

# Metric fn: one example + model output -> score (0.0-1.0 recommended)
MetricFn = Callable[[Example, str], float]


@dataclass
class Evaluator:
    name: str
    metric_fn: MetricFn
    weight: float = 1.0  # optional, for combining metrics


class Task(Protocol):
    """Protocol defining the interface for optimization tasks."""

    name: str

    def load_datasets(
        self,
    ) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
        """Return (trainset, valset, testset)."""
        ...

    def make_seed_candidate(self) -> dict:
        """Return the initial GEPA candidate, e.g. {"system_prompt": "..."}."""
        ...

    def get_evaluators(self) -> list[Evaluator]:
        """Return the evaluators used for post-hoc test evaluation."""
        ...

    def get_gepa_kwargs(self) -> dict[str, Any]:
        """Optional extra kwargs for gepa.optimize."""
        ...


@dataclass
class SimpleTask:
    """
    Declarative task configuration that implements the Task protocol.

    Example usage:
        TaskImpl = SimpleTask(
            name="math_aime",
            data_path="data/math_aime.jsonl",
            system_prompt="You are a helpful assistant...",
            evaluators=[Evaluator(name="accuracy", metric_fn=accuracy_metric)],
            primary_metric=accuracy_metric,
        )
    """

    name: str
    data_path: str
    system_prompt: str
    evaluators: list[Evaluator]
    primary_metric: MetricFn
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42
    extra_gepa_kwargs: dict[str, Any] = field(default_factory=dict)

    def load_datasets(
        self,
    ) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
        return load_dataset(
            self.data_path,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            seed=self.seed,
        )

    def make_seed_candidate(self) -> dict:
        return {"system_prompt": self.system_prompt}

    def get_evaluators(self) -> list[Evaluator]:
        return self.evaluators

    def get_gepa_kwargs(self) -> dict[str, Any]:
        return {**self.extra_gepa_kwargs}
