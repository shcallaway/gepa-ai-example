from __future__ import annotations
import re

ANSWER_REGEX = re.compile(r"###\s*([0-9\-\.]+)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_REGEX.search(text)
    return m.group(1).strip() if m else None
