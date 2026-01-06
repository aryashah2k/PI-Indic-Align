from typing import Dict, List, Tuple

import numpy as np

from sentence_transformers import SentenceTransformer

import torch


MODEL_MAP = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "labse": "sentence-transformers/LaBSE",
    "indic-sbert": "l3cube-pune/indic-sentence-similarity-sbert",
    "indic-sbert-nli": "l3cube-pune/indic-sentence-bert-nli",
    # optional per-language models can be passed directly as names as well
}


class EmbeddingBackbone:
    def __init__(self, backbone: str, device: str = "cpu", prompts: Dict[str, Dict[str, str]] = None):
        self.backbone_name = backbone
        model_name = MODEL_MAP.get(backbone, backbone)
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.prompts = prompts or {}
        self.model.eval()

    def _format(self, texts: List[str], is_query: bool) -> List[str]:
        cfg = self.prompts.get(self.backbone_name, {"query_prefix": "", "doc_prefix": ""})
        prefix = cfg["query_prefix" if is_query else "doc_prefix"]
        if prefix:
            return [prefix + t for t in texts]
        return texts

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 64, is_query: bool = False) -> np.ndarray:
        texts_fmt = self._format(texts, is_query=is_query)
        emb = self.model.encode(
            texts_fmt,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=True,  # Use SentenceTransformers' built-in tqdm progress bar
        )
        assert emb.ndim == 2 and emb.shape[0] == len(texts)
        return emb

    def encode_queries_docs(
        self, queries: List[str], docs: List[str], batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_emb = self.encode(queries, batch_size=batch_size, is_query=True)
        d_emb = self.encode(docs, batch_size=batch_size, is_query=False)
        return q_emb, d_emb
