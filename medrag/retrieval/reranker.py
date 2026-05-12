from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks hybrid retrieval candidates using a cross-encoder model.
    Cross-encoders jointly encode (query, document) pairs, producing
    more accurate relevance scores than bi-encoder retrieval alone.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Loading cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
