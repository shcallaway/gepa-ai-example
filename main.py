import argparse
import importlib
import sys

from src.core.registry import get_task
from src.core.runner import run_gepa_for_task
from src.core.executor import Executor


def load_executor_from_module(module_path: str) -> Executor:
    """Load an executor from a Python module.

    The module must export either:
    - `executor`: An Executor instance ready to use
    - `get_executor()`: A factory function returning an Executor

    Args:
        module_path: Dotted Python module path (e.g., 'examples.custom_executor')

    Returns:
        The loaded Executor instance.

    Raises:
        ValueError: If the module doesn't export 'executor' or 'get_executor()'.
    """
    module = importlib.import_module(module_path)

    if hasattr(module, "get_executor"):
        return module.get_executor()
    elif hasattr(module, "executor"):
        return module.executor
    else:
        raise ValueError(
            f"Module '{module_path}' must export 'executor' or 'get_executor()'"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GEPA prompt optimization for a specific task"
    )
    parser.add_argument("--task", required=True, help="Task name, e.g. math_aime")
    parser.add_argument("--max-metric-calls", type=int, default=150)

    # Executor options (mutually exclusive)
    executor_group = parser.add_mutually_exclusive_group()
    executor_group.add_argument(
        "--task-lm",
        default="openai/gpt-4o-mini",
        help="LiteLLM model string for task execution (default: openai/gpt-4o-mini)",
    )
    executor_group.add_argument(
        "--executor-module",
        help=(
            "Python module path to load a custom executor from. "
            "Module must export 'executor' or 'get_executor()'. "
            "WARNING: Only use trusted modules as this executes arbitrary code."
        ),
    )

    parser.add_argument("--reflection-lm", default="openai/gpt-4o")
    return parser.parse_args()


def main():
    args = parse_args()

    task = get_task(args.task)

    # Determine executor
    if args.executor_module:
        print(f"Loading custom executor from module: {args.executor_module}")
        print("WARNING: Custom executors execute arbitrary code. Only use trusted modules.")
        executor = load_executor_from_module(args.executor_module)
    else:
        executor = args.task_lm  # String will be wrapped by run_gepa_for_task

    run_dir = run_gepa_for_task(
        task=task,
        executor=executor,
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_metric_calls,
    )
    print(f"Finished optimization for task '{task.name}'. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
