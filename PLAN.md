# Plan: Adding Custom Agent/Executor Support

## Overview

Add support for custom agents (LangChain, custom Python, etc.) to the GEPA prompt optimization framework, enabling users to optimize prompts for any LLM-based system, not just LiteLLM-supported models.

## Current State

- `runner.py` accepts `task_lm` and `reflection_lm` as model strings (e.g., `"openai/gpt-4o-mini"`)
- `gepa.optimize()` is called with these strings, which internally use LiteLLM
- `evaluate_candidate_on_testset()` directly calls `litellm.completion()` for post-hoc evaluation
- No way to inject custom execution logic

## Target State

- Support passing callable executors instead of model strings
- Custom executors work for both GEPA optimization and post-hoc test evaluation
- Backward compatible: string model names continue to work
- Clean API that's easy for users to implement

---

## Implementation Plan

### Step 1: Define Executor Protocol (`src/core/executor.py`)

Create a new module defining the executor interface:

```python
from typing import Protocol, TypedDict

class ChatMessage(TypedDict):
    role: str  # "system", "user", "assistant"
    content: str

class Executor(Protocol):
    """Protocol for custom LLM/agent executors."""
    def __call__(self, messages: list[ChatMessage]) -> str:
        """Execute a chat completion and return the response text."""
        ...
```

Also provide:
- `LiteLLMExecutor` class: wraps a model string for backward compatibility
- `make_executor()` factory: converts `str | Executor` to `Executor`

### Step 2: Update `runner.py`

Modify `run_gepa_for_task()`:
- Change `task_lm: str` parameter to `task_lm: str | Executor`
- Use `make_executor()` to normalize the input
- Pass the executor directly to `gepa.optimize(task_lm=...)`

Modify `evaluate_candidate_on_testset()`:
- Add `executor: Executor` parameter
- Replace `litellm.completion()` call with `executor(messages)`
- Remove direct LiteLLM dependency from this function

### Step 3: Update `main.py` CLI

- Keep existing `--task-lm` argument for model strings
- Add optional `--executor-module` argument for advanced users who want to load a custom executor from a Python module
- Default behavior unchanged: model strings work as before

### Step 4: Update `base_task.py` (Optional Enhancement)

Add optional `executor` field to `SimpleTask`:
- Allows tasks to specify a default custom executor
- Useful for tasks that require specific agent architectures

### Step 5: Add Example Custom Executor

Create `examples/custom_executor.py` demonstrating:
- A simple custom executor wrapping the Anthropic API directly
- A LangChain agent executor example (commented/pseudocode)
- How to pass the executor to the runner

### Step 6: Update Documentation

- Update `README.md` with custom executor usage
- Update `CLAUDE.md` with new architecture details
- Add docstrings to new modules

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/core/executor.py` | New | Executor protocol, LiteLLMExecutor, make_executor() |
| `src/core/__init__.py` | Modify | Export new executor types |
| `src/core/runner.py` | Modify | Accept Executor, update evaluate function |
| `main.py` | Modify | Add --executor-module option |
| `examples/custom_executor.py` | New | Example implementations |
| `README.md` | Modify | Document custom executor usage |
| `CLAUDE.md` | Modify | Update architecture section |

---

## API Design

### User-Facing API

```python
# Option 1: Model string (existing behavior)
uv run python3 main.py --task math_aime --task-lm openai/gpt-4o-mini

# Option 2: Custom executor module
uv run python3 main.py --task math_aime --executor-module my_agents.claude_executor

# Option 3: Programmatic usage
from src.core.executor import Executor
from src.core.runner import run_gepa_for_task

class MyAgent:
    def __call__(self, messages: list[dict]) -> str:
        # Custom logic here
        return response

run_gepa_for_task(task, task_lm=MyAgent(), reflection_lm="openai/gpt-4o")
```

### Executor Implementation Example

```python
from src.core.executor import Executor, ChatMessage

class ClaudeExecutor:
    """Example executor using Anthropic's Claude API directly."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def __call__(self, messages: list[ChatMessage]) -> str:
        # Extract system message if present
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
```

---

## Testing Strategy

1. **Unit tests**: Test `make_executor()` with both strings and callables
2. **Integration test**: Run optimization with a mock executor that returns fixed responses
3. **Manual test**: Run full optimization with a real custom executor

---

## Backward Compatibility

- All existing CLI commands continue to work unchanged
- `--task-lm "openai/gpt-4o-mini"` works exactly as before
- No changes required to existing task definitions
- New functionality is opt-in

---

## Future Enhancements (Out of Scope)

- Custom reflection executors (for now, reflection_lm stays as model string)
- Async executor support
- Batch execution optimization for custom executors
- Executor configuration via YAML/JSON files
