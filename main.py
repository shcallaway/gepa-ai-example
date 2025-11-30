import argparse
from src.core.registry import get_task
from src.core.runner import run_gepa_for_task


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GEPA prompt optimization for a specific task"
    )
    parser.add_argument("--task", required=True, help="Task name, e.g. math_aime")
    parser.add_argument("--max-metric-calls", type=int, default=150)
    parser.add_argument("--task-lm", default="openai/gpt-4o-mini")
    parser.add_argument("--reflection-lm", default="openai/gpt-4o")
    return parser.parse_args()


def main():
    args = parse_args()

    task = get_task(args.task)
    run_dir = run_gepa_for_task(
        task=task,
        task_lm=args.task_lm,
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_metric_calls,
    )
    print(f"Finished optimization for task '{task.name}'. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
