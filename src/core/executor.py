"""Executor types for custom LLM/agent support.

This module provides the Executor protocol for custom LLM implementations.
GEPA's task_lm parameter accepts callables, so users can provide their own
LLM integrations (LangChain agents, direct API calls, etc.) instead of
just LiteLLM model strings.
"""

from __future__ import annotations

from collections.abc import Sequence

from gepa.adapters.default_adapter.default_adapter import (
    ChatCompletionCallable as Executor,
    ChatMessage,
)

__all__ = ["Executor", "ChatMessage", "LiteLLMExecutor"]


class LiteLLMExecutor:
    """Built-in executor using LiteLLM.

    This is the default executor that wraps LiteLLM's completion API.
    Use this when you want to use any model supported by LiteLLM.

    Example:
        executor = LiteLLMExecutor("openai/gpt-4o-mini")
        response = executor([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model: str) -> None:
        """Initialize with a LiteLLM model string.

        Args:
            model: LiteLLM model identifier (e.g., "openai/gpt-4o-mini")
        """
        self.model = model

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        """Execute the LLM call.

        Args:
            messages: Sequence of chat messages with 'role' and 'content' keys.

        Returns:
            The model's response content as a string.

        Raises:
            ValueError: If the model returns an empty response.
        """
        import litellm

        response = litellm.completion(model=self.model, messages=list(messages))
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty response")
        return content
