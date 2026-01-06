from typing import Dict, List

import os
import numpy as np
from sklearn import metrics as skm

# Disable numba JIT in ranx to avoid Numba warnings when PYTHONWARNINGS=error
os.environ.setdefault("RANX_DISABLE_NUMBA", "1")
from ranx import Qrels, Run, evaluate


def retrieval_metrics(qrels: Dict[str, Dict[str, float]], run: Dict[str, Dict[str, float]], ks: List[int]):
    try:
        q = Qrels(qrels)
        r = Run(run)
        res = evaluate(q, r, metrics=["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"])
        # Filter only requested ks for recall
        out = {
            "recall": {k: float(res.get(f"recall@{k}")) for k in ks if f"recall@{k}" in res},
            "mrr@10": float(res.get("mrr@10")),
            "ndcg@10": float(res.get("ndcg@10")),
        }
        return out
    except Exception:
        # Fallback (single relevant doc per query assumed):
        # recall@k: 1 if relevant doc in top-k, else 0; averaged
        # mrr@10: reciprocal rank if <=10, else 0; averaged
        # ndcg@10: 1/log2(1+rank) if rank<=10, else 0; averaged (IDCG=1)
        import math

        qids = list(qrels.keys())
        # For each qid, get the relevant docid (there is exactly one in our setup)
        rel_doc = {q: next(iter(qrels[q].keys())) for q in qids}

        recalls = {k: [] for k in ks}
        mrr_list = []
        ndcg_list = []
        for q in qids:
            ranking = sorted(run[q].items(), key=lambda x: -x[1])
            # Find rank (1-based)
            rank = None
            for i, (did, _) in enumerate(ranking, start=1):
                if did == rel_doc[q]:
                    rank = i
                    break
            for k in ks:
                recalls[k].append(1.0 if (rank is not None and rank <= k) else 0.0)
            # MRR@10
            if rank is not None and rank <= 10:
                mrr_list.append(1.0 / rank)
            else:
                mrr_list.append(0.0)
            # nDCG@10
            if rank is not None and rank <= 10:
                ndcg_list.append(1.0 / math.log2(1 + rank))
            else:
                ndcg_list.append(0.0)

        out = {
            "recall": {k: float(np.mean(recalls[k])) for k in ks},
            "mrr@10": float(np.mean(mrr_list)),
            "ndcg@10": float(np.mean(ndcg_list)),
        }
        return out


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    auroc = skm.roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = skm.average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    acc = skm.accuracy_score(y_true, y_pred)
    return {"auroc": float(auroc), "auprc": float(auprc), "accuracy": float(acc)}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    # ECE using bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)
