from typing import Any, Dict, List, Tuple, Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz
from tqdm.auto import tqdm

from .data_loader import Record


@dataclass
class Pair:
    q_id: str
    d_id: str
    q_text: str
    d_text: str
    label: int  # 1 for positive, 0 for negative
    lang: str


def build_positive_pairs(records: Iterable[Record]) -> List[Pair]:
    pairs: List[Pair] = []
    # Wrap with tqdm if length is known
    try:
        iterator = tqdm(records, total=len(records), desc="Building positive pairs")
    except Exception:
        iterator = tqdm(records, desc="Building positive pairs")
    for r in iterator:
        pairs.append(Pair(q_id=r.id, d_id=r.id, q_text=r.persona, d_text=r.instruction, label=1, lang=r.lang))
    return pairs


def mine_hard_negatives(records: List[Record], threshold: int = 70, max_negatives_per_query: int = 10) -> List[Pair]:
    pairs: List[Pair] = []
    # Within-language negatives only
    by_lang: Dict[str, List[Record]] = {}
    try:
        iterator = tqdm(records, total=len(records), desc="Indexing records by language")
    except Exception:
        iterator = tqdm(records, desc="Indexing records by language")
    for r in iterator:
        by_lang.setdefault(r.lang, []).append(r)

    # Mine per-language with visible progress
    for lang, recs in tqdm(by_lang.items(), desc="Mining hard negatives (langs)"):
        try:
            outer_iter = tqdm(enumerate(recs), total=len(recs), desc=f"HN {lang} queries")
        except Exception:
            outer_iter = tqdm(enumerate(recs), desc=f"HN {lang} queries")
        for i, r in outer_iter:
            cands = []
            # inner loop across candidates
            for other in recs:
                if r.id == other.id:
                    continue
                score = fuzz.token_set_ratio(r.instruction, other.instruction)
                if score >= threshold:
                    cands.append((score, other))
            # Sort by decreasing similarity and pick top-k
            cands.sort(key=lambda x: -x[0])
            for _, neg in cands[:max_negatives_per_query]:
                pairs.append(Pair(q_id=r.id, d_id=neg.id, q_text=r.persona, d_text=neg.instruction, label=0, lang=lang))
    return pairs


def build_retrieval_corpus(records: Iterable[Record], mode: str = "persona->instruction") -> Tuple[List[str], List[str], List[str], List[str]]:
    # Returns: query_ids, queries, doc_ids, docs
    query_ids: List[str] = []
    queries: List[str] = []
    doc_ids: List[str] = []
    docs: List[str] = []
    if mode == "persona->instruction":
        for r in records:
            query_ids.append(r.id)
            queries.append(r.persona)
            doc_ids.append(r.id)
            docs.append(r.instruction)
    elif mode == "instruction->persona":
        for r in records:
            query_ids.append(r.id)
            queries.append(r.instruction)
            doc_ids.append(r.id)
            docs.append(r.persona)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return query_ids, queries, doc_ids, docs
