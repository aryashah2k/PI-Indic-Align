from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from .embeddings import EmbeddingBackbone
from .utils import write_jsonl, write_json
from .metrics import retrieval_metrics


def build_qrels(query_ids: List[str], doc_ids: List[str]) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    for q, d in zip(query_ids, doc_ids):
        qrels.setdefault(q, {})[d] = 1.0
    return qrels


def run_dense_retrieval(
    backbone: EmbeddingBackbone,
    query_ids: List[str],
    queries: List[str],
    doc_ids: List[str],
    docs: List[str],
    batch_size: int,
    top_k_values: List[int],
    run_dir: str,
):
    # Encode docs once
    d_emb = backbone.encode(docs, batch_size=batch_size, is_query=False)
    # Prepare run dict for ranx: query -> {doc: score} (store only top-N per query)
    run: Dict[str, Dict[str, float]] = {}
    top_n = max(100, max(top_k_values))  # keep at least 100 per query for auditing

    # Process queries in batches to limit memory
    num_queries = len(queries)
    for start in tqdm(range(0, num_queries, batch_size), desc="Ranking (batched)"):
        end = min(start + batch_size, num_queries)
        q_batch = queries[start:end]
        q_ids_batch = query_ids[start:end]
        q_emb = backbone.encode(q_batch, batch_size=batch_size, is_query=True)
        # Compute scores for this batch: (B, D)
        scores_batch = np.matmul(q_emb, d_emb.T)
        # For each query in batch, keep only top_n
        for i, qid in enumerate(q_ids_batch):
            scores = scores_batch[i]
            if top_n < len(doc_ids):
                idx = np.argpartition(-scores, top_n)[:top_n]
            else:
                idx = np.arange(len(doc_ids))
            # Order the selected indices by score desc
            order = idx[np.argsort(-scores[idx])]
            run[qid] = {doc_ids[j]: float(scores[j]) for j in order}

    qrels = build_qrels(query_ids, doc_ids)
    metrics = retrieval_metrics(qrels, run, ks=top_k_values)

    # Save artifacts
    write_jsonl(f"{run_dir}/queries.jsonl", [{"id": qid, "text": q} for qid, q in zip(query_ids, queries)])
    write_jsonl(f"{run_dir}/candidates.jsonl", [{"id": did, "text": d} for did, d in zip(doc_ids, docs)])

    # Save top-100 per query for auditing
    scores_rows = []
    for qid in tqdm(query_ids, desc="Saving top-100"):
        ranked = list(run[qid].items())[:100]
        for did, s in ranked:
            scores_rows.append({"qid": qid, "docid": did, "score": s})
    write_jsonl(f"{run_dir}/scores.jsonl", scores_rows)
    write_json(f"{run_dir}/metrics.json", metrics)

    return metrics
