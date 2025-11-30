# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent prompt optimization lab using the GEPA (Generative Evolutionary Prompt Architecture) library. It provides a framework for running automated prompt optimization experiments on various LLM tasks.

## Commands

### Run optimization
```bash
uv run python3 main.py --task <task_name> [--max-metric-calls N] [--reflection-lm MODEL]
```
Note: Use `uv run` to execute within the virtual environment.
- `--task`: Required. Task name (e.g., `math_aime`, `qa_multistep`)
- `--max-metric-calls`: Optimization budget (default: 150)
- `--reflection-lm`: Model for reflection/optimization (default: `openai/gpt-4o`)

### Install dependencies
```bash
uv pip install -e .
```

### Environment
Requires `OPENAI_API_KEY` environment variable to be set.

## Architecture

### Core Framework (`src/core/`)
- **`base_task.py`**: Defines the `Task` protocol and `SimpleTask` dataclass for declarative task configuration. Tasks provide datasets, seed prompts, evaluators, primary metric, executor, and optional extra GEPA kwargs.
- **`registry.py`**: Maps task names to task implementations. New tasks must be registered here.
- **`runner.py`**: Orchestrates GEPA optimization: loads data, runs `gepa.optimize()`, evaluates on test set, saves artifacts.
- **`dataset.py`**: Shared dataset loader that reads JSONL files, transforms `expected_output` to `answer` for GEPA compatibility, and splits into train/val/test sets.

### Tasks (`src/tasks/`)
Each task is a self-contained subdirectory with all task-specific resources:
- **`task.py`**: Creates a `TaskImpl` using `SimpleTask` with task-specific config
- **`evaluators.py`**: Defines metric functions and exports `EVALUATORS` list and `PRIMARY_METRIC`
- **`executor.py`**: Defines the executor (LLM or agent) for the task
- **`prompt.txt`**: Seed system prompt for optimization
- **`data.jsonl`**: Training/validation/test data

### Data Format
Tasks use JSONL files with examples containing `input` and `expected_output` fields. The dataset loader transforms `expected_output` to `answer` for GEPA compatibility. Evaluator metric functions receive examples with `input` and `answer` fields.

### Artifacts
Optimization runs save outputs to `artifacts/<task_name>/run-<timestamp>/`:
- `optimized_prompt.txt`: Best system prompt found (plain text)
- `optimized_prompt.json`: Best candidate found (full structure)
- `metrics.json`: Test set evaluation results

## Adding a New Task

1. Create `src/tasks/<task_name>/` directory with `__init__.py`
2. Create `evaluators.py` with metric functions, `EVALUATORS` list, and `PRIMARY_METRIC`
3. Create `executor.py` with your custom executor function
4. Create `prompt.txt` with the seed system prompt
5. Create `data.jsonl` with training/validation/test examples
6. Create `task.py` with `TaskImpl = SimpleTask(...)` configuration
7. Register in `src/core/registry.py`
