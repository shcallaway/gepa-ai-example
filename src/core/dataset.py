from __future__ import annotations
from typing import Sequence, Mapping, Any
from pathlib import Path
import json
import random

Example = Mapping[str, Any]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
