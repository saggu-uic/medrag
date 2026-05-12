"""
Retrieval evaluation — 10 configs compared on held-out eval set.

Configs:
  1.  bm25_only
  2.  bm25_reranked              (BM25 + cross-encoder)
  3.  dense_minilm
  4.  dense_minilm_reranked      (MiniLM + cross-encoder)
  5.  dense_pubmedbert
  6.  dense_pubmedbert_reranked  (PubMedBERT + cross-encoder)
  7.  hybrid_minilm              (BM25 + MiniLM)
  8.  hybrid_pubmedbert          (BM25 + PubMedBERT)
  9.  hybrid_reranked            (BM25 + MiniLM + cross-encoder)
  10. hybrid_pubmed_reranked     (BM25 + PubMedBERT + cross-encoder)

Usage:
    python scripts/evaluate_retrieval.py \
        --index-dir data/index \
        --eval-set data/processed/eval_queries.json \
        --max-samples 150
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np

from medrag.indexing.bm25_index import BM25Index
from medrag.indexing.vector_index import VectorIndex
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import CrossEncoderReranker
from medrag.evaluation.metrics import ndcg_at_k, recall_at_k


def free():
    gc.collect()


def load_eval_set(path: str, max_samples: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)[:max_samples]


def dedupe_by_article(results: list[dict]) -> list[str]:
    seen, ids = set(), []
    for r in results:
        aid = r["article_id"]
        if aid not in seen:
            seen.add(aid)
            ids.append(aid)
    return ids


def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a list of per-query scores."""
    rng = random.Random(42)
    n = len(scores)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lower = (1 - ci) / 2
    upper = 1 - lower
    return round(boot_means[int(lower * n_bootstrap)], 4), round(boot_means[int(upper * n_bootstrap)], 4)


def run_config(name: str, fn, samples: list[dict], k_values: list[int]) -> dict:
    all_retrieved, all_relevant = [], []
    for sample in samples:
        try:
            res = fn(sample["question"], max(k_values))
            all_retrieved.append(dedupe_by_article(res))
        except Exception:
            all_retrieved.append([])
        all_relevant.append(set(sample["relevant_doc_ids"]))

    n = len(samples)
    metrics = {}
    for k in k_values:
        ndcg_scores   = [ndcg_at_k(r, rel, k)   for r, rel in zip(all_retrieved, all_relevant)]
        recall_scores = [recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]

        mean_ndcg   = round(sum(ndcg_scores) / n, 4)
        mean_recall = round(sum(recall_scores) / n, 4)

        ci_low, ci_high = bootstrap_ci(ndcg_scores)

        metrics[f"ndcg@{k}"]            = mean_ndcg
        metrics[f"ndcg@{k}_ci"]         = [ci_low, ci_high]
        metrics[f"recall@{k}"]          = mean_recall

    print(f"  {name:<30} NDCG@5={metrics['ndcg@5']:.3f} 95%CI=[{metrics['ndcg@5_ci'][0]:.3f},{metrics['ndcg@5_ci'][1]:.3f}]  Recall@10={metrics['recall@10']:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir",   default="data/index")
    parser.add_argument("--eval-set",    default="data/processed/eval_queries.json")
    parser.add_argument("--output",      default="results/retrieval_results.json")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--skip-pubmedbert", action="store_true")
    args = parser.parse_args()

    samples = load_eval_set(args.eval_set, args.max_samples)
    print(f"Evaluating on {len(samples)} samples\n")

    k_values = [5, 10, 20]
    results  = {}

    print("=== Retrieval Ablation ===")
    print(f"  {'Config':<28} {'NDCG@5':>8} {'NDCG@10':>9} {'Recall@10':>10}")
    print("  " + "-" * 58)

    # ── 1. BM25 ───────────────────────────────────────────────────────────────
    print("Loading BM25...")
    bm25 = BM25Index()
    bm25.load(f"{args.index_dir}/bm25")
    results["bm25_only"] = run_config(
        "bm25_only", lambda q, k: bm25.search(q, top_k=k), samples, k_values)
    free()

    # ── Load reranker once (used across multiple configs) ─────────────────────
    print("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    # ── 2. BM25 + Reranker ────────────────────────────────────────────────────
    results["bm25_reranked"] = run_config(
        "bm25_reranked",
        lambda q, k: reranker.rerank(q, bm25.search(q, top_k=20), top_k=k),
        samples, k_values)
    free()

    # ── 3. Dense MiniLM ───────────────────────────────────────────────────────
    print("Loading MiniLM vector index...")
    vec_minilm = VectorIndex(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="medrag_minilm",
        persist_dir=f"{args.index_dir}/chroma_minilm",
    )
    results["dense_minilm"] = run_config(
        "dense_minilm", lambda q, k: vec_minilm.search(q, top_k=k), samples, k_values)
    free()

    # ── 4. Dense MiniLM + Reranker ────────────────────────────────────────────
    results["dense_minilm_reranked"] = run_config(
        "dense_minilm_reranked",
        lambda q, k: reranker.rerank(q, vec_minilm.search(q, top_k=20), top_k=k),
        samples, k_values)
    free()

    # ── 5. Dense PubMedBERT ───────────────────────────────────────────────────
    if not args.skip_pubmedbert:
        print("Loading PubMedBERT vector index...")
        vec_pubmed = VectorIndex(
            model_name="NeuML/pubmedbert-base-embeddings",
            collection_name="medrag_pubmedbert",
            persist_dir=f"{args.index_dir}/chroma_pubmedbert",
        )
        results["dense_pubmedbert"] = run_config(
            "dense_pubmedbert", lambda q, k: vec_pubmed.search(q, top_k=k), samples, k_values)
        free()

        # ── 6. Dense PubMedBERT + Reranker ────────────────────────────────────
        results["dense_pubmedbert_reranked"] = run_config(
            "dense_pubmedbert_reranked",
            lambda q, k: reranker.rerank(q, vec_pubmed.search(q, top_k=20), top_k=k),
            samples, k_values)
        del vec_pubmed; free()

    # ── 7. Hybrid MiniLM ──────────────────────────────────────────────────────
    print("Loading hybrid (BM25 + MiniLM)...")
    hybrid_m = HybridRetriever(bm25, vec_minilm)
    results["hybrid_minilm"] = run_config(
        "hybrid_minilm", lambda q, k: hybrid_m.retrieve(q, top_k=k), samples, k_values)
    free()

    # ── 8. Hybrid PubMedBERT ──────────────────────────────────────────────────
    if not args.skip_pubmedbert:
        print("Loading hybrid (BM25 + PubMedBERT)...")
        vec_p = VectorIndex(
            model_name="NeuML/pubmedbert-base-embeddings",
            collection_name="medrag_pubmedbert",
            persist_dir=f"{args.index_dir}/chroma_pubmedbert",
        )
        hybrid_p = HybridRetriever(bm25, vec_p)
        results["hybrid_pubmedbert"] = run_config(
            "hybrid_pubmedbert", lambda q, k: hybrid_p.retrieve(q, top_k=k), samples, k_values)
        del vec_p, hybrid_p; free()

    # ── 9. Hybrid MiniLM + Reranker ───────────────────────────────────────────
    print("Loading hybrid MiniLM + reranker...")
    hybrid_r = HybridRetriever(bm25, vec_minilm)
    results["hybrid_reranked"] = run_config(
        "hybrid_reranked",
        lambda q, k: reranker.rerank(q, hybrid_r.retrieve(q, top_k=20), top_k=k),
        samples, k_values)
    del hybrid_r, vec_minilm; free()

    # ── 10. Hybrid PubMedBERT + Reranker ──────────────────────────────────────
    if not args.skip_pubmedbert:
        print("Loading hybrid PubMedBERT + reranker...")
        vec_p2 = VectorIndex(
            model_name="NeuML/pubmedbert-base-embeddings",
            collection_name="medrag_pubmedbert",
            persist_dir=f"{args.index_dir}/chroma_pubmedbert",
        )
        hybrid_p2 = HybridRetriever(bm25, vec_p2)
        results["hybrid_pubmed_reranked"] = run_config(
            "hybrid_pubmed_reranked",
            lambda q, k: reranker.rerank(q, hybrid_p2.retrieve(q, top_k=20), top_k=k),
            samples, k_values)
        del vec_p2, hybrid_p2; free()

    del reranker; free()

    # ── Summary ───────────────────────────────────────────────────────────────
    base = results["bm25_only"]["ndcg@5"]
    best_name = max(results, key=lambda k: results[k]["ndcg@5"])
    best = results[best_name]["ndcg@5"]
    improvement = (best - base) / base * 100 if base > 0 else 0

    print(f"\n  Best config    : {best_name} (NDCG@5={best:.3f})")
    print(f"  Improvement    : +{improvement:.1f}% over BM25 baseline")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
