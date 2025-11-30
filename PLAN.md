# Plan: Adding Custom Agent/Executor Support

## Overview

Add support for custom agents (LangChain, custom Python, etc.) to the GEPA prompt optimization framework, enabling users to optimize prompts for any LLM-based system, not just LiteLLM-supported models.

## Current State

- `runner.py` accepts `task_lm` and `reflection_lm` as model strings (e.g., `"openai/gpt-4o-mini"`)
- `gepa.optimize()` is called with these strings, which internally use LiteLLM
- `evaluate_candidate_on_testset()` directly calls `litellm.completion()` for post-hoc evaluation
- No way to inject custom execution logic

## Key Discovery

**GEPA already supports custom executors!** The `task_lm` parameter accepts:
```python
task_lm: str | ChatCompletionCallable | None
```

Where `ChatCompletionCallable` is a protocol:
```python
class ChatCompletionCallable(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...
```

This means we can leverage GEPA's existing types rather than defining our own.

## Target State

- All execution goes through `Executor` callables (no raw model strings in the core API)
- `LiteLLMExecutor` provided as the built-in executor for standard models
- CLI provides `--task-lm` as a convenience shorthand that creates `LiteLLMExecutor` internally
- Custom executors supported via `--executor-module` for advanced users
- Clean, simple API that's easy for users to implement

---

## Implementation Plan

### Step 1: Define Executor Module (`src/core/executor.py`)

Create a new module that re-exports GEPA's types and provides `LiteLLMExecutor`:

```python
"""Executor types and built-in implementations for custom LLM/agent support."""

from collections.abc import Sequence

# Re-export GEPA's types for user convenience
from gepa.adapters.default_adapter.default_adapter import (
    ChatCompletionCallable as Executor,
    ChatMessage,
)

__all__ = ["Executor", "ChatMessage", "LiteLLMExecutor"]


class LiteLLMExecutor:
    """Built-in executor using LiteLLM for standard model access."""

    def __init__(self, model: str):
        self.model = model

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        import litellm
        response = litellm.completion(model=self.model, messages=list(messages))
        return response.choices[0].message.content or ""
```

**Key changes from original plan:**
- Re-export `ChatCompletionCallable` as `Executor` from GEPA (same interface, better compatibility)
- Re-export `ChatMessage` from GEPA
- Use `Sequence[ChatMessage]` to match GEPA's signature exactly
- Add `or ""` guard for cases where content is `None`

### Step 2: Update `runner.py`

Modify `run_gepa_for_task()`:
- Change `task_lm: str` parameter to `task_lm: Executor`
- Pass the executor directly to `gepa.optimize(task_lm=...)` (GEPA already accepts callables)
- Update the call to `evaluate_candidate_on_testset()` to pass the executor

Modify `evaluate_candidate_on_testset()`:
- Change `model: str` parameter to `executor: Executor`
- Replace `litellm.completion()` call with `executor(messages)`
- Remove direct LiteLLM import from this function

```python
# Before
def evaluate_candidate_on_testset(
    candidate: dict,
    testset: list[Example],
    evaluators: list[Evaluator],
    model: str,
) -> dict[str, float]:
    ...
    response = litellm.completion(model=model, messages=[...])
    model_output = response.choices[0].message.content or ""

# After
def evaluate_candidate_on_testset(
    candidate: dict,
    testset: list[Example],
    evaluators: list[Evaluator],
    executor: Executor,
) -> dict[str, float]:
    ...
    model_output = executor([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ])
```

### Step 3: Update `main.py` CLI

- Keep `--task-lm` argument as a convenience shorthand
  - Internally creates `LiteLLMExecutor(model_string)`
- Add `--executor-module` argument for custom executors
  - Loads a Python module and expects it to export `executor` (an instance) or `get_executor()` (a factory function)
  - Mutually exclusive with `--task-lm`

```python
# Module loading logic
def load_executor_from_module(module_path: str) -> Executor:
    """Load an executor from a Python module path."""
    import importlib
    module = importlib.import_module(module_path)

    # Try get_executor() factory first, then executor instance
    if hasattr(module, "get_executor"):
        return module.get_executor()
    elif hasattr(module, "executor"):
        return module.executor
    else:
        raise ValueError(
            f"Module {module_path} must export 'executor' or 'get_executor()'"
        )
```

Example CLI usage:
```bash
# Convenience shorthand (creates LiteLLMExecutor internally)
uv run python3 main.py --task math_aime --task-lm openai/gpt-4o-mini

# Custom executor module
uv run python3 main.py --task math_aime --executor-module my_agents.claude_executor
```

**Note:** `--executor-module` dynamically imports Python code. Users should only use modules they trust.

### Step 4: Add Example Custom Executor

Create `examples/custom_executor.py` demonstrating:
- A simple custom executor wrapping the Anthropic API directly
- A mock executor for testing
- How to structure the module for `--executor-module` usage

```python
"""Example custom executors for use with --executor-module."""

from collections.abc import Sequence
from src.core.executor import ChatMessage


class AnthropicExecutor:
    """Custom executor using the Anthropic API directly."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=chat_messages,
        )
        return response.content[0].text


class MockExecutor:
    """Mock executor for testing - returns fixed responses."""

    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.call_count = 0
        self.last_messages: list[ChatMessage] = []

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        self.call_count += 1
        self.last_messages = list(messages)
        return self.response


# Export for --executor-module (uses Anthropic by default)
executor = AnthropicExecutor()

# Or use a factory for lazy initialization
def get_executor() -> AnthropicExecutor:
    return AnthropicExecutor()
```

### Step 5: Update Documentation

- Update `README.md` with custom executor usage
- Update `CLAUDE.md` with new architecture details
- Add docstrings to new modules

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/core/executor.py` | New | Re-exports GEPA types, provides `LiteLLMExecutor` |
| `src/core/__init__.py` | Modify | Export new executor types |
| `src/core/runner.py` | Modify | Accept `Executor`, update `evaluate_candidate_on_testset()` |
| `main.py` | Modify | Add `--executor-module`, wire up `LiteLLMExecutor` for `--task-lm` |
| `examples/custom_executor.py` | New | `AnthropicExecutor`, `MockExecutor` examples |
| `README.md` | Modify | Document custom executor usage |
| `CLAUDE.md` | Modify | Update architecture section |

---

## API Design

### CLI API

```bash
# Option 1: Model string shorthand (creates LiteLLMExecutor internally)
uv run python3 main.py --task math_aime --task-lm openai/gpt-4o-mini

# Option 2: Custom executor module
uv run python3 main.py --task math_aime --executor-module examples.custom_executor
```

### Programmatic API

```python
from src.core.executor import Executor, LiteLLMExecutor
from src.core.runner import run_gepa_for_task

# Using built-in LiteLLM executor
run_gepa_for_task(
    task,
    task_lm=LiteLLMExecutor("openai/gpt-4o-mini"),
    reflection_lm="openai/gpt-4o"
)

# Using custom executor
from collections.abc import Sequence
from src.core.executor import ChatMessage

class MyAgent:
    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        # Custom logic here
        return response

run_gepa_for_task(task, task_lm=MyAgent(), reflection_lm="openai/gpt-4o")
```

**Note on `reflection_lm`:** This parameter remains a model string (not an executor). GEPA uses it internally for the reflection/mutation process. Supporting custom reflection executors is a potential future enhancement.

### Executor Module Contract

For `--executor-module`, the module must export one of:
- `executor`: An `Executor` instance ready to use
- `get_executor()`: A factory function that returns an `Executor`

The factory is checked first, allowing for lazy initialization or configuration.

---

## Testing Strategy

1. **Unit tests**: Test `LiteLLMExecutor` wraps LiteLLM correctly
2. **Integration test**: Run optimization with `MockExecutor` that returns fixed responses
3. **Manual test**: Run full optimization with `AnthropicExecutor` or another custom executor

---

## Design Decisions

### Why re-export GEPA's types instead of defining our own?

1. **Compatibility**: Executors are passed directly to `gepa.optimize()`, so using GEPA's exact types avoids any type mismatches
2. **Maintenance**: If GEPA's interface changes, we only need to update our re-export
3. **Simplicity**: Users can reference one place (`src/core/executor.py`) for all executor-related types

### Why keep `reflection_lm` as a string?

1. **Scope**: The reflection LM is internal to GEPA's optimization loop; users rarely need to customize it
2. **Complexity**: Supporting custom reflection executors would require deeper GEPA integration
3. **Value**: The primary use case is custom task execution, not custom reflection

---

## Future Enhancements (Out of Scope)

- Custom reflection executors (for now, `reflection_lm` stays as model string)
- Async executor support
- Batch execution optimization for custom executors
- Executor configuration via YAML/JSON files
- Custom `GEPAAdapter` support for advanced users who need full control over evaluation, traces, and reflection
