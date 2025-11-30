"""Example custom executors for prompt optimization.

This module demonstrates how to create custom executors for use with
the --executor-module flag. Custom executors allow you to use any LLM
provider, custom agents, or mock implementations.

Usage:
    uv run python3 main.py --task math_aime --executor-module examples.custom_executor

The module must export either:
    - `executor`: An Executor instance ready to use
    - `get_executor()`: A factory function returning an Executor
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from src.core.executor import ChatMessage


class AnthropicExecutor:
    """Executor using the Anthropic API directly.

    Requires ANTHROPIC_API_KEY environment variable.

    Example:
        executor = AnthropicExecutor("claude-sonnet-4-20250514")
        response = executor([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: uv pip install anthropic"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        # Anthropic API uses system as a separate parameter
        system_content = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_content,
            messages=user_messages,
        )

        content = response.content[0].text if response.content else None
        if content is None:
            raise ValueError("Anthropic returned empty response")
        return content


class MockExecutor:
    """Mock executor for testing without API calls.

    Returns a configurable response for all inputs. Useful for:
    - Testing the optimization pipeline
    - Debugging evaluators
    - CI/CD environments without API access

    Example:
        executor = MockExecutor("42")  # Always returns "42"
    """

    def __init__(self, response: str = "This is a mock response.") -> None:
        self.response = response
        self.call_count = 0

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        self.call_count += 1
        return self.response


def get_executor():
    """Factory function to create an executor based on environment.

    This demonstrates the `get_executor()` pattern which allows dynamic
    configuration based on environment variables or other runtime factors.

    Returns:
        AnthropicExecutor if ANTHROPIC_API_KEY is set, otherwise MockExecutor.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Using Anthropic executor (claude-sonnet-4-20250514)")
        return AnthropicExecutor()
    else:
        print("ANTHROPIC_API_KEY not set, using MockExecutor")
        return MockExecutor()


# Alternative: Export a pre-configured executor instance
# Uncomment to use the `executor` export pattern instead of `get_executor()`
#
# executor = AnthropicExecutor("claude-sonnet-4-20250514")
