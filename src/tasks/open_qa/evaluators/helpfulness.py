from __future__ import annotations

import litellm

from src.core.base_task import Example
from ._utils import JUDGE_MODEL


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
