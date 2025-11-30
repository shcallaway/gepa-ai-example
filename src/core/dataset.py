from __future__ import annotations
from typing import Sequence, Mapping, Any
from pathlib import Path
import json
import random

Example = Mapping[str, Any]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries.

    Transforms 'expected_output' to 'answer' for GEPA compatibility.
    """
    data = []
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            # GEPA expects 'answer' field, not 'expected_output'
            if "expected_output" in item and "answer" not in item:
                item["answer"] = item.pop("expected_output")
            data.append(item)
    return data


def load_dataset(
    path: str | Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
    """
    Load a JSONL dataset and split into train/val/test sets.

    Args:
        path: Path to the JSONL file
        train_frac: Fraction of data for training (default 0.8)
        val_frac: Fraction of data for validation (default 0.1)
        seed: Random seed for shuffling (default 42)

    Returns:
        Tuple of (trainset, valset, testset)
    """
    data = load_jsonl(Path(path))
    random.Random(seed).shuffle(data)

    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    trainset = data[:n_train]
    valset = data[n_train : n_train + n_val]
    testset = data[n_train + n_val :]

    return trainset, valset, testset
