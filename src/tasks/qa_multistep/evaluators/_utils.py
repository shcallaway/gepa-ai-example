from __future__ import annotations
import re


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_answer(text: str) -> str:
    """Extract text after 'Answer:' prefix, or return full text as fallback."""
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()
