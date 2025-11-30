# Project: GEPA Multi-Agent Prompt Optimization Lab

## 1. Project Overview

**Goal:**
Build a small but extensible Python project that uses [GEPA](https://github.com/gepa-ai/gepa) to **optimize prompts** for multiple “agents” (tasks). Each agent:

* Has its **own dataset** (train/val/test).
* Has its **own seed prompt** (GEPA candidate).
* Can define its **own evaluators/metrics**.
* Can be optimized independently via:

```bash
python main.py --task <task-name>
```

**Initial agents:**

1. `math_aime` – solves AIME-style math problems (using GEPA’s built-in AIME dataset).
2. `qa_multistep` – answers complex multi-step Q&A from a custom dataset.

The design should make it easy to add more agents without touching core orchestration logic.

---

## 2. Technical Stack

* **Language:** Python 3.10+
* **Core libraries:**

  * `gepa[full]`

    * Provides `gepa.optimize`, example datasets, adapters, and LLM access via LiteLLM.
  * `litellm` (pulled in via `gepa[full]`) for model calls.
* **Environment:**

  * `OPENAI_API_KEY` (or other provider key supported by LiteLLM) in environment variables.

**Example dependencies (pyproject.toml style):**

```toml
[project]
name = "gepa-multi-agent-demo"
version = "0.1.0"
dependencies = [
  "gepa[full]",
]
```

A `requirements.txt` equivalent is fine if preferred.

---

## 3. Directory Structure

Suggested layout:

```text
gepa-multi-agent-demo/
  pyproject.toml          # or requirements.txt
  main.py                 # CLI entrypoint

  src/
    core/
      base_task.py        # Task interface / protocol
      registry.py         # maps task name -> Task class
      runner.py           # runs GEPA optimization for a task

    tasks/
      math_aime/
        task.py           # MathAimeTask implementation
        dataset.py        # AIME dataset load function
        evaluators.py     # accuracy metric, parsing helpers

      qa_multistep/
        task.py           # MultiStepQATask implementation
        dataset.py        # QA dataset load/split
        evaluators.py     # EM/F1 metrics

  data/
    qa_multistep/
      qa_multistep.jsonl  # Custom QA dataset (input/expected_output per line)

  artifacts/
    math_aime/
      run-YYYYMMDD-HHMMSS/
        optimized_prompt.json
        metrics.json      # optional: test-set evaluation
        logs.json         # optional
    qa_multistep/
      run-YYYYMMDD-HHMMSS/
        optimized_prompt.json
        metrics.json
```

The `artifacts` tree is per-task, per-run to keep results organized.

---

## 4. Core Abstractions

### 4.1 Example & Evaluator types

In `src/core/base_task.py`:

* Shared typing and evaluator structure.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, Mapping, Any, Callable

Example = Mapping[str, Any]

# Metric fn: one example + model output -> score (0.0–1.0 recommended)
MetricFn = Callable[[Example, str], float]

@dataclass
class Evaluator:
    name: str
    metric_fn: MetricFn
    weight: float = 1.0  # optional, for combining metrics
```

### 4.2 Task interface

Each agent implements this protocol:

```python
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
```

This interface is intentionally small; new agents just implement these methods.

---

## 5. Task Registry

Central mapping from CLI `--task` name to Task class.

`src/core/registry.py`:

```python
from typing import Dict, Callable
from src.core.base_task import Task

from src.tasks.math_aime.task import MathAimeTask
from src.tasks.qa_multistep.task import MultiStepQATask

TASK_REGISTRY: Dict[str, Callable[[], Task]] = {
    "math_aime": MathAimeTask,
    "qa_multistep": MultiStepQATask,
}

def get_task(task_name: str) -> Task:
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        available = ", ".join(TASK_REGISTRY.keys())
        raise SystemExit(f"Unknown task '{task_name}'. Available: {available}")
```

To add a new agent later, you just:

1. Implement a `Task` subclass under `src/tasks/<new_task>/task.py`.
2. Register it here.

---

## 6. GEPA Runner

The runner is responsible for:

* Loading datasets via `task.load_datasets`.
* Calling `gepa.optimize`.
* Saving the best candidate (prompt) into `artifacts`.
* Optionally running post-hoc evaluation on the test set.

`src/core/runner.py` (outline):

```python
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import gepa

from src.core.base_task import Task

def run_gepa_for_task(
    task: Task,
    task_lm: str = "openai/gpt-4o-mini",
    reflection_lm: str = "openai/gpt-4o",
    max_metric_calls: int = 150,
    artifacts_root: str = "artifacts",
) -> Path:
    # 1. Data
    trainset, valset, testset = task.load_datasets()

    # 2. Seed + extra GEPA config
    seed_candidate = task.make_seed_candidate()
    gepa_kwargs = task.get_gepa_kwargs()

    # 3. Run directory
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(artifacts_root) / task.name / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4. GEPA optimization
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
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

    # 5. Post-hoc evaluation on test set (optional but recommended)
    evaluators = task.get_evaluators()
    if evaluators and testset:
        metrics = evaluate_candidate_on_testset(
            candidate=best_candidate,
            testset=testset,
            evaluators=evaluators,
            model=task_lm,
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return run_dir
```

### 6.1 Evaluation helper (conceptual)

Implement `evaluate_candidate_on_testset` in `runner.py` or a separate module:

* Loads `system_prompt` from the candidate.
* For each example in `testset`:

  * Calls the same `task_lm` (via LiteLLM or direct OpenAI client).
  * Gets model output.
  * Applies each evaluator’s `metric_fn(example, output)`.
* Aggregates metrics (mean across test set) and returns a dict like:

```json
{
  "exact_match": 0.53,
  "f1": 0.71
}
```

This makes metrics visible for each run.

---

## 7. CLI Entrypoint (main.py)

The CLI simply:

* Parses args.
* Looks up the task from the registry.
* Calls `run_gepa_for_task`.

`main.py`:

```python
import argparse
from src.core.registry import get_task
from src.core.runner import run_gepa_for_task

def parse_args():
    parser = argparse.ArgumentParser()
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
```

**Usage:**

```bash
python main.py --task math_aime
python main.py --task qa_multistep
```

---

## 8. GEPA Default Metric

GEPA's default metric evaluates correctness by comparing model output to `expected_output`. For v1, we rely on this default for optimization. If task-specific evaluation metrics (e.g., F1, custom parsing) diverge significantly from simple correctness matching, customize the metric via `get_gepa_kwargs()` by passing a custom `metric` function to `gepa.optimize`.

---

## 9. Agent 1: Math AIME Task

### 9.1 Dataset loader

Use GEPA’s built-in AIME dataset.

`src/tasks/math_aime/dataset.py`:

* Calls `gepa.examples.aime.init_dataset()`.
* Returns `(trainset, valset, testset)`.
* Optional subsampling for speed during development.

```python
from typing import Sequence, Mapping, Any
import gepa

Example = Mapping[str, Any]

def load_aime_dataset(
    sample_size: int | None = None,
) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
    trainset, valset, testset = gepa.examples.aime.init_dataset()

    if sample_size is not None:
        trainset = trainset[:sample_size]
        valset = valset[:sample_size]
        testset = testset[:sample_size]

    return trainset, valset, testset
```

Dataset should already match GEPA’s expected format for the default adapter (`"input"` and `"expected_output"` keys).

### 9.2 Evaluators

`src/tasks/math_aime/evaluators.py`:

* Metric: accuracy (1.0 if predicted final answer equals gold, else 0.0).
* Extract final answer with regex from model output, e.g. from format `"### <answer>"`.

```python
from __future__ import annotations
from typing import Mapping, Any
import re
from src.core.base_task import Evaluator, Example

ANSWER_REGEX = re.compile(r"###\s*([0-9\-\.]+)")

def extract_answer(text: str) -> str | None:
    m = ANSWER_REGEX.search(text)
    return m.group(1).strip() if m else None

def accuracy_metric(example: Example, model_output: str) -> float:
    gold = str(example["expected_output"]).strip()
    pred = extract_answer(model_output)
    return 1.0 if pred == gold else 0.0

def get_evaluators() -> list[Evaluator]:
    return [
        Evaluator(name="accuracy", metric_fn=accuracy_metric, weight=1.0),
    ]
```

> **Note:** Verify that GEPA's built-in AIME dataset uses `expected_output` values that match the parsed output from the `"### <answer>"` format. If GEPA's AIME example expects structured JSON output instead, consider either (a) adapting the seed prompt to match, or (b) passing `accuracy_metric` as a custom metric via `get_gepa_kwargs()`.

### 9.3 Task implementation

`src/tasks/math_aime/task.py`:

* Defines the seed system prompt tailored to AIME problems and the required output format.
* Hooks up loader and evaluators.

```python
from __future__ import annotations
from typing import Sequence, Any
from src.core.base_task import Task, Evaluator, Example
from .dataset import load_aime_dataset
from .evaluators import get_evaluators, accuracy_metric

class MathAimeTask(Task):
    name = "math_aime"

    def load_datasets(
        self,
    ) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
        return load_aime_dataset(sample_size=60)  # adjust or remove sampling

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
        return {"metric": accuracy_metric}
```

---

## 10. Agent 2: Multi-Step QA Task

### 10.1 Dataset

`src/tasks/qa_multistep/dataset.py`:

* Expects a `qa_multistep.jsonl` file under `data/qa_multistep`.
* Each line: JSON with at least `"input"` (question/context) and `"expected_output"` (target answer).
* Splits into train/val/test.

```python
from __future__ import annotations
from typing import Sequence, Mapping, Any
from pathlib import Path
import json
import random

Example = Mapping[str, Any]

def _load_jsonl(path: Path) -> list[Example]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def load_qa_dataset(
    root: str = "data/qa_multistep",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
    path = Path(root) / "qa_multistep.jsonl"
    data = _load_jsonl(path)
    random.Random(seed).shuffle(data)

    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    trainset = data[:n_train]
    valset = data[n_train:n_train + n_val]
    testset = data[n_train + n_val:]

    return trainset, valset, testset
```

### 10.2 Evaluators

`src/tasks/qa_multistep/evaluators.py`:

* Metrics:

  * `exact_match` on normalized strings.
  * `f1` over token overlap.

```python
from __future__ import annotations
from typing import Mapping, Any
from collections import Counter
import re

from src.core.base_task import Evaluator, Example

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def extract_answer(text: str) -> str:
    """Extract text after 'Answer:' prefix, or return full text as fallback."""
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()

def exact_match(example: Example, model_output: str) -> float:
    gold = normalize(example["expected_output"])
    pred = normalize(extract_answer(model_output))
    return 1.0 if gold == pred else 0.0

def f1_score(example: Example, model_output: str) -> float:
    gold_tokens = normalize(example["expected_output"]).split()
    pred_tokens = normalize(extract_answer(model_output)).split()

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    common = sum((gold_counts & pred_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def get_evaluators() -> list[Evaluator]:
    return [
        Evaluator(name="exact_match", metric_fn=exact_match, weight=1.0),
        Evaluator(name="f1", metric_fn=f1_score, weight=1.0),
    ]
```

### 10.3 Task implementation

`src/tasks/qa_multistep/task.py`:

* Seed prompt describes multi-step reasoning and final answer formatting (e.g. “Answer:” prefix).
* Uses QA dataset and evaluators.

```python
from __future__ import annotations
from typing import Sequence, Any
from src.core.base_task import Task, Evaluator, Example
from .dataset import load_qa_dataset
from .evaluators import get_evaluators, exact_match

class MultiStepQATask(Task):
    name = "qa_multistep"

    def load_datasets(
        self,
    ) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
        return load_qa_dataset()

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
        return {"metric": exact_match}
```

---

## 11. Future Extensions

The design is intentionally modular:

* **Add new agents** by creating `src/tasks/<name>/` with:

  * `task.py` implementing the `Task` protocol,
  * `dataset.py` and `evaluators.py`,
  * and registering the Task in `registry.py`.
* **Custom metrics inside GEPA**:

  * `get_gepa_kwargs()` can later pass custom scoring functions/adapters to `gepa.optimize` if you want GEPA to optimize directly for a more bespoke metric.
* **Interactive agents**:

  * You can add `agent.py` per task that loads `artifacts/<task>/latest/optimized_prompt.json` and exposes a simple CLI for ad-hoc queries.
* **Experiment tracking**:

  * Integrate MLflow or Weights & Biases via GEPA’s `run_dir` outputs for better telemetry.

---

## 12. Implementation Order (for the developer)

Recommended sequence:

1. **Environment & deps**

   * Create repo, add `pyproject.toml` / `requirements.txt`.
   * Install and verify `gepa` import works.

2. **Core skeleton**

   * Implement `base_task.py`, `registry.py`, `runner.py`.
   * Implement `main.py` CLI with `--task`.

3. **Math AIME Task**

   * Implement `math_aime/dataset.py` using GEPA’s AIME dataset.
   * Implement `math_aime/evaluators.py` (accuracy via “### answer”).
   * Implement `math_aime/task.py`.
   * Register `MathAimeTask` in `registry.py`.
   * Run: `python main.py --task math_aime` to confirm full GEPA loop works (start with small `max_metric_calls`).

4. **QA Multi-step Task**

   * Prepare `data/qa_multistep/qa_multistep.jsonl`.
   * Implement `qa_multistep/dataset.py`, `evaluators.py`, `task.py`.
   * Register `MultiStepQATask`.
   * Run: `python main.py --task qa_multistep`.

5. **Post-hoc evaluation**

   * Implement `evaluate_candidate_on_testset` and `metrics.json` export in `runner.py`.
   * Confirm metrics show up in `artifacts/<task>/run-.../metrics.json`.

6. **Polish**

   * Add `README.md` explaining:

     * Setup, environment variables, how to run each task.
     * Where optimized prompts and metrics are stored.
   * Add simple logging/prints to show progress and selected best candidates.


