"""Executor for the qa_multistep task."""

import litellm


def executor(messages):
    """Execute LLM calls for multi-step question answering.

    Uses GPT-4o-mini for cost-effective reasoning.
    Expects model output to include 'Answer:' followed by the final answer.
    """
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content
