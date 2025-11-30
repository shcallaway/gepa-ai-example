"""Executor for the math_aime task."""

import litellm


def executor(messages):
    """Execute LLM calls for math problem solving.

    Uses GPT-4o-mini for cost-effective math reasoning.
    Expects model output to end with ### followed by the numeric answer.
    """
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content
