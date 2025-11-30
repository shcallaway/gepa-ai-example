"""Executor for the open_qa task."""

import litellm


def executor(messages):
    """Execute LLM calls for open-ended question answering.

    Uses GPT-4o-mini for generating responses to open-ended questions.
    """
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content
