"""Shared utilities for tool_eval evaluators."""

from __future__ import annotations

import json
import re


def parse_output(model_output: str) -> dict:
    """Parse the structured JSON output from the executor."""
    try:
        return json.loads(model_output)
    except json.JSONDecodeError:
        return {"answer": model_output, "tool_calls": []}


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())
