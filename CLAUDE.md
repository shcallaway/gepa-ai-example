# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent prompt optimization lab using the GEPA (Generative Evolutionary Prompt Architecture) library. It provides a framework for running automated prompt optimization experiments on various LLM tasks.

## Commands

### Run optimization
```bash
uv run python3 main.py --task <task_name> [--max-metric-calls N] [--task-lm MODEL] [--reflection-lm MODEL]
```
Note: Use `uv run` to execute within the virtual environment.
- `--task`: Required. Task name (e.g., `math_aime`, `qa_multistep`)
- `--max-metric-calls`: Optimization budget (default: 150)
- `--task-lm`: Model for task execution (default: `openai/gpt-4o-mini`)
- `--reflection-lm`: Model for reflection/optimization (default: `openai/gpt-4o`)

### Install dependencies
```bash
uv pip install -e .
```

### Environment
Requires `OPENAI_API_KEY` environment variable to be set.

## Architecture

### Core Framework (`src/core/`)
- **`base_task.py`**: Defines the `Task` protocol and `SimpleTask` dataclass for declarative task configuration. Tasks provide datasets, seed prompts, evaluators, primary metric, and optional extra GEPA kwargs.
- **`registry.py`**: Maps task names to task implementations. New tasks must be registered here.
- **`runner.py`**: Orchestrates GEPA optimization: loads data, runs `gepa.optimize()`, evaluates on test set, saves artifacts.
- **`dataset.py`**: Shared dataset loader that reads JSONL files, transforms `expected_output` to `answer` for GEPA compatibility, and splits into train/val/test sets.

### Tasks (`src/tasks/`)
Each task is a subdirectory containing:
- **`task.py`**: Creates a `TaskImpl` using `SimpleTask` with task-specific config (data path, system prompt, evaluators)
- **`evaluators.py`**: Defines metric functions and exports `EVALUATORS` list and `PRIMARY_METRIC`

### Data Format
Tasks use JSONL files in `data/` with examples containing `input` and `expected_output` fields. The dataset loader transforms `expected_output` to `answer` for GEPA compatibility. Evaluator metric functions receive examples with `input` and `answer` fields.

### Artifacts
Optimization runs save outputs to `artifacts/<task_name>/run-<timestamp>/`:
- `optimized_prompt.json`: Best candidate found
- `metrics.json`: Test set evaluation results

## Adding a New Task

1. Create `src/tasks/<task_name>/` directory with `__init__.py`
2. Create `evaluators.py` with metric functions, `EVALUATORS` list, and `PRIMARY_METRIC`
3. Create `task.py` with `TaskImpl = SimpleTask(...)` configuration
4. Add data file at `data/<task_name>.jsonl`
5. Register in `src/core/registry.py`
