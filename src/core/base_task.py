from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, Mapping, Any, Callable

Example = Mapping[str, Any]

# Metric fn: one example + model output -> score (0.0-1.0 recommended)
MetricFn = Callable[[Example, str], float]


@dataclass
class Evaluator:
    name: str
    metric_fn: MetricFn
    weight: float = 1.0  # optional, for combining metrics


class Task(Protocol):
    # Human-readable name and registry key
    name: str

    def load_datasets(
        self,
    ) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
        """
        Return (trainset, valset, testset).
        Each element should be a mapping; for GEPA's DefaultAdapter,
        examples must have at least 'input' and 'expected_output' keys.
        """
        ...

    def make_seed_candidate(self) -> dict:
        """
        Return the initial GEPA candidate, e.g. {"system_prompt": "..."}.
        GEPA will mutate this candidate during optimization.
        """
        ...

    def get_evaluators(self) -> list[Evaluator]:
        """
        Return the evaluators used for post-hoc test evaluation
        (and optionally for primary GEPA metric if we hook that in later).
        """
        ...

    def get_gepa_kwargs(self) -> dict[str, Any]:
        """
        Optional extra kwargs for gepa.optimize, e.g. custom adapter/metric.
        Can return {} if defaults are fine.
        """
        ...
