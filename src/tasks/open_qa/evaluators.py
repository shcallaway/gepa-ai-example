"""Evaluators for the open_qa task.

Demonstrates LLM-as-judge evaluation for open-ended questions where
exact matching is insufficient.
"""

from __future__ import annotations

import litellm

from src.core.base_task import Evaluator, Example

# Model used for judging responses
JUDGE_MODEL = "openai/gpt-4o-mini"


def llm_judge_correctness(example: Example, model_output: str) -> float:
    """Use an LLM to judge whether the response correctly answers the question.

    Returns a score from 0.0 to 1.0 based on the judge's assessment.
    """
    judge_prompt = f"""You are evaluating an AI assistant's response to a question.

Question: {example["input"]}

Reference answer (for context, not the only valid answer): {example["answer"]}

Assistant's response: {model_output}

Rate the correctness of the assistant's response on a scale from 0 to 10:
- 0: Completely wrong or irrelevant
- 5: Partially correct but missing key information
- 10: Fully correct and complete

Respond with ONLY a single integer from 0 to 10."""

    try:
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        score = int(score_text)
        return max(0.0, min(1.0, score / 10.0))
    except (ValueError, AttributeError):
        return 0.0


def llm_judge_helpfulness(example: Example, model_output: str) -> float:
    """Use an LLM to judge how helpful and well-explained the response is.

    Returns a score from 0.0 to 1.0 based on clarity and helpfulness.
    """
    judge_prompt = f"""You are evaluating an AI assistant's response for helpfulness.

Question: {example["input"]}

Assistant's response: {model_output}

Rate the helpfulness of this response on a scale from 0 to 10:
- 0: Unhelpful, confusing, or harmful
- 5: Somewhat helpful but could be clearer or more complete
- 10: Extremely helpful, clear, and well-explained

Respond with ONLY a single integer from 0 to 10."""

    try:
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        score = int(score_text)
        return max(0.0, min(1.0, score / 10.0))
    except (ValueError, AttributeError):
        return 0.0


EVALUATORS = [
    Evaluator(name="correctness", metric_fn=llm_judge_correctness, weight=1.0),
    Evaluator(name="helpfulness", metric_fn=llm_judge_helpfulness, weight=0.5),
]

PRIMARY_METRIC = llm_judge_correctness
