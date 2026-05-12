from __future__ import annotations

from typing import Any

from medrag.indexing.bm25_index import BM25Index
from medrag.indexing.vector_index import VectorIndex


class HybridRetriever:
    """
    Fuses BM25 and vector retrieval scores using weighted combination,
    then returns top-K candidates for downstream reranking.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def retrieve(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        fetch_k = top_k * 5

        bm25_results = self.bm25_index.search(query, top_k=fetch_k)
        vector_results = self.vector_index.search(query, top_k=fetch_k)

        bm25_scores = self._normalize({r["chunk_id"]: r["bm25_score"] for r in bm25_results})
        vector_scores = self._normalize({r["chunk_id"]: r["vector_score"] for r in vector_results})

        combined: dict[str, dict[str, Any]] = {}
        for r in bm25_results:
            combined[r["chunk_id"]] = r.copy()

        for r in vector_results:
            cid = r["chunk_id"]
            if cid not in combined:
                combined[cid] = r.copy()

        for cid, entry in combined.items():
            bm25 = bm25_scores.get(cid, 0.0)
            vec = vector_scores.get(cid, 0.0)
            entry["bm25_score_norm"] = bm25
            entry["vector_score_norm"] = vec
            entry["hybrid_score"] = self.bm25_weight * bm25 + self.vector_weight * vec

        ranked = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        min_s = min(scores.values())
        max_s = max(scores.values())
        span = max_s - min_s
        if span == 0:
            return {k: 1.0 for k in scores}
        return {k: (v - min_s) / span for k, v in scores.items()}
