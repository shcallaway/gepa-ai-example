from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json

import gepa

from src.core.base_task import Task, Evaluator, Example


def evaluate_candidate_on_testset(
    candidate: dict,
    testset: list[Example],
    evaluators: list[Evaluator],
    executor: Executor,
) -> dict[str, float]:
    """
    Run the optimized candidate on the test set and compute metrics.

    Args:
        candidate: The optimized candidate containing 'system_prompt'.
        testset: List of test examples with 'input' and 'answer' fields.
        evaluators: List of evaluators to compute metrics.
        executor: The executor to use for LLM calls.

    Returns:
        Dictionary mapping evaluator names to average scores.
    """
    system_prompt = candidate.get("system_prompt", "")

    metrics_sum: dict[str, float] = {e.name: 0.0 for e in evaluators}

    for example in testset:
        user_input = example["input"]

        # Call the model via executor
        model_output = executor([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ])

        # Apply each evaluator
        for evaluator in evaluators:
            score = evaluator.metric_fn(example, model_output)
            metrics_sum[evaluator.name] += score

    # Compute averages
    n = len(testset)
    if n == 0:
        return {e.name: 0.0 for e in evaluators}

    return {name: total / n for name, total in metrics_sum.items()}


def run_gepa_for_task(
    task: Task,
    reflection_lm: str = "openai/gpt-4o",
    max_metric_calls: int = 150,
    artifacts_root: str = "artifacts",
) -> Path:
    """
    Run GEPA optimization for a given task.

    Args:
        task: The task to optimize (includes its own executor).
        reflection_lm: Model string for GEPA's internal reflection (passed to GEPA).
        max_metric_calls: Maximum number of metric evaluations.
        artifacts_root: Root directory for saving artifacts.

    Returns:
        Path to the run directory containing artifacts.
    """
    executor = task.get_executor()
    task_lm = executor
    eval_executor = executor

    # 1. Data
    trainset, valset, testset = task.load_datasets()
    print(f"Loaded datasets: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    # 2. Seed + extra GEPA config
    seed_candidate = task.make_seed_candidate()
    gepa_kwargs = task.get_gepa_kwargs()
    print(f"Seed candidate: {seed_candidate}")

    # 3. Run directory
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(artifacts_root) / task.name / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # 4. GEPA optimization
    print(f"Starting GEPA optimization with max_metric_calls={max_metric_calls}...")
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=list(trainset),
        valset=list(valset),
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        **gepa_kwargs,
    )

    best_candidate = result.best_candidate
    (run_dir / "optimized_prompt.json").write_text(
        json.dumps(best_candidate, indent=2)
    )
    print(f"Best candidate saved to: {run_dir / 'optimized_prompt.json'}")

    # 5. Post-hoc evaluation on test set
    evaluators = task.get_evaluators()
    if evaluators and testset:
        print("Running post-hoc evaluation on test set...")
        metrics = evaluate_candidate_on_testset(
            candidate=best_candidate,
            testset=list(testset),
            evaluators=evaluators,
            executor=eval_executor,
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"Test metrics: {metrics}")

    return run_dir
