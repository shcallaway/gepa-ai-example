import argparse

from dotenv import load_dotenv

from src.core.registry import get_task

load_dotenv()
from src.core.runner import run_gepa_for_task


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GEPA prompt optimization for a specific task"
    )
    parser.add_argument("--task", required=True, help="Task name, e.g. math_aime")
    parser.add_argument("--max-metric-calls", type=int, default=5)
    parser.add_argument("--reflection-lm", default="openai/gpt-5")
    return parser.parse_args()


def main():
    args = parse_args()

    task = get_task(args.task)

    run_dir = run_gepa_for_task(
        task=task,
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_metric_calls,
    )
    print(f"Finished optimization for task '{task.name}'. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
