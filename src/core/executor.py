"""Executor types for custom LLM/agent support.

This module provides the Executor protocol for custom LLM implementations.
Users must provide a custom executor module via --executor-module.
"""

from gepa.adapters.default_adapter.default_adapter import (
    ChatCompletionCallable as Executor,
    ChatMessage,
)

__all__ = ["Executor", "ChatMessage"]
