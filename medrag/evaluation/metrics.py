from __future__ import annotations

import numpy as np


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / k


def hallucination_rate(verification_results: list[dict]) -> float:
    if not verification_results:
        return 0.0
    unsupported = sum(1 for r in verification_results if r.get("unsupported", False))
    return unsupported / len(verification_results)


def mean_faithfulness_score(verification_results: list[dict]) -> float:
    if not verification_results:
        return 0.0
    scores = [r.get("faithfulness_score", 0.0) for r in verification_results]
    return float(np.mean(scores))


def aggregate_retrieval_metrics(
    all_retrieved: list[list[str]],
    all_relevant: list[set[str]],
    k_values: list[int] = [5, 10, 20],
) -> dict[str, float]:
    results = {}
    for k in k_values:
        ndcg_scores = [ndcg_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        recall_scores = [recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        results[f"ndcg@{k}"] = float(np.mean(ndcg_scores))
        results[f"recall@{k}"] = float(np.mean(recall_scores))
    return results
