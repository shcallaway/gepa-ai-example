from __future__ import annotations
from typing import Sequence, Mapping, Any
from pathlib import Path
import json
import random

Example = Mapping[str, Any]


def _load_jsonl(path: Path) -> list[Example]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_dataset(
    root: str = "data/math_aime",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Sequence[Example], Sequence[Example], Sequence[Example]]:
    path = Path(root) / "math_aime.jsonl"
    data = _load_jsonl(path)
    random.Random(seed).shuffle(data)

    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    trainset = data[:n_train]
    valset = data[n_train : n_train + n_val]
    testset = data[n_train + n_val :]

    return trainset, valset, testset
