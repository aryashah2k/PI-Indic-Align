from typing import Dict, List, Tuple
import numpy as np

from .data_loader import Record


def random_split(records: List[Record], train_frac: float, dev_frac: float, test_frac: float, seed: int) -> Dict[str, List[Record]]:
    assert abs(train_frac + dev_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(records)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]
    return {
        "train": [records[i] for i in train_idx],
        "dev": [records[i] for i in dev_idx],
        "test": [records[i] for i in test_idx],
    }


def leave_one_language_out(by_lang: Dict[str, List[Record]], held_out_lang: str) -> Dict[str, List[Record]]:
    train: List[Record] = []
    for lang, recs in by_lang.items():
        if lang == held_out_lang:
            continue
        train.extend(recs)
    test = by_lang.get(held_out_lang, [])
    return {"train": train, "test": test}


def filter_by_shot_types(records: List[Record], shot_types: List[str]) -> List[Record]:
    if not shot_types:
        return records
    allowed = set([s.lower() for s in shot_types])
    out: List[Record] = []
    for r in records:
        desc = (r.description or "").lower()
        if any(tag in desc for tag in allowed):
            out.append(r)
    return out
