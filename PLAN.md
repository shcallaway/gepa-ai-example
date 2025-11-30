# Plan: Custom Executor Support

## Goal

Enable custom LLM executors (LangChain agents, direct API calls, etc.) for prompt optimization, not just LiteLLM model strings.

## Key Insight

GEPA's `task_lm` parameter already accepts callables:
```python
task_lm: str | ChatCompletionCallable | None
```

We just need to surface this capability to users.

## Implementation

### 1. Create `src/core/executor.py`

```python
"""Executor types for custom LLM/agent support."""

from collections.abc import Sequence
from gepa.adapters.default_adapter.default_adapter import (
    ChatCompletionCallable as Executor,
    ChatMessage,
)

__all__ = ["Executor", "ChatMessage", "LiteLLMExecutor"]


class LiteLLMExecutor:
    """Built-in executor using LiteLLM."""

    def __init__(self, model: str):
        self.model = model

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        import litellm
        response = litellm.completion(model=self.model, messages=list(messages))
        return response.choices[0].message.content or ""
```

### 2. Update `src/core/runner.py`

- Change `task_lm: str` to `task_lm: Executor` in both functions
- Replace `litellm.completion()` in `evaluate_candidate_on_testset()` with `executor(messages)`

### 3. Update `main.py`

- Keep `--task-lm` as shorthand (creates `LiteLLMExecutor` internally)
- Add `--executor-module` for custom executors (mutually exclusive with `--task-lm`)

```python
def load_executor_from_module(module_path: str) -> Executor:
    import importlib
    module = importlib.import_module(module_path)
    if hasattr(module, "get_executor"):
        return module.get_executor()
    elif hasattr(module, "executor"):
        return module.executor
    raise ValueError(f"Module {module_path} must export 'executor' or 'get_executor()'")
```

### 4. Create `examples/custom_executor.py`

Example implementations: `AnthropicExecutor`, `MockExecutor`.

### 5. Update docs

Update README.md with custom executor usage examples.

## File Changes

| File | Action |
|------|--------|
| `src/core/executor.py` | Create |
| `src/core/runner.py` | Modify |
| `main.py` | Modify |
| `examples/custom_executor.py` | Create |
| `README.md` | Modify |

## Usage

```bash
# Standard (creates LiteLLMExecutor internally)
uv run python3 main.py --task math_aime --task-lm openai/gpt-4o-mini

# Custom executor
uv run python3 main.py --task math_aime --executor-module examples.custom_executor
```

## Executor Module Contract

Export either:
- `executor`: An instance ready to use
- `get_executor()`: A factory function returning an executor

## Notes

- `reflection_lm` remains a string (GEPA internal use)
- `--executor-module` dynamically imports Python - users should only use trusted modules
