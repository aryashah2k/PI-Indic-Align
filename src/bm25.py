from typing import List, Tuple, Dict

from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


def whitespace_tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, documents: List[str]):
        self.docs = documents
        self.tokenized = [whitespace_tokenize(d) for d in documents]
        self.model = BM25Okapi(self.tokenized)

    def search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[int, float]]]:
        results: List[List[Tuple[int, float]]] = []
        for q in tqdm(queries, desc="BM25 searching"):
            toks = whitespace_tokenize(q)
            scores = self.model.get_scores(toks)
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
            results.append(ranked)
        return results
