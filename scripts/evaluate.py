"""
Runs the full evaluation pipeline on PubMedQA.
Produces ablation table across 4 retrieval configurations.

Usage:
    python scripts/evaluate.py --index-dir data/index --output results/ablation.json --max-samples 100
"""

import argparse
import json
import traceback
from pathlib import Path

from medrag.indexing.bm25_index import BM25Index
from medrag.indexing.vector_index import VectorIndex
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import CrossEncoderReranker
from medrag.generation.pipeline import RAGPipeline
from medrag.verification.faithfulness import FaithfulnessVerifier
from medrag.evaluation.bioasq import BioASQEvaluator
from medrag.evaluation.metrics import (
    ndcg_at_k, recall_at_k,
    hallucination_rate, mean_faithfulness_score,
)


def run_retrieval_config(name, retrieve_fn, samples, k_values):
    all_retrieved, all_relevant = [], []
    for sample in samples:
        try:
            results = retrieve_fn(sample.question, max(k_values))
            retrieved_ids = [r["article_id"] for r in results]
        except Exception:
            retrieved_ids = []
        all_retrieved.append(retrieved_ids)
        all_relevant.append(sample.relevant_doc_ids)

    metrics = {}
    for k in k_values:
        ndcgs = [ndcg_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        recalls = [recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        metrics[f"ndcg@{k}"] = round(sum(ndcgs) / len(ndcgs), 4)
        metrics[f"recall@{k}"] = round(sum(recalls) / len(recalls), 4)

    print(f"  {name}: NDCG@5={metrics.get('ndcg@5', 0):.3f}  Recall@10={metrics.get('recall@10', 0):.3f}")
    return metrics


def run_hallucination_eval(samples, retrieve_fn, generator, verifier, top_k=5):
    ver_results = []
    for sample in samples:
        try:
            chunks = retrieve_fn(sample.question, top_k)
            gen = generator.generate(sample.question, chunks)
            ver = verifier.verify(gen["answer"], chunks)
            ver_results.append(ver.to_dict())
        except Exception:
            traceback.print_exc()
            continue

    return {
        "n_samples": len(ver_results),
        "hallucination_rate": round(hallucination_rate(ver_results), 4),
        "mean_faithfulness_score": round(mean_faithfulness_score(ver_results), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--output", default="results/ablation.json")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    print("Loading indexes and models...")
    bm25 = BM25Index()
    bm25.load(f"{args.index_dir}/bm25")
    vector = VectorIndex(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_dir=f"{args.index_dir}/chroma",
    )
    hybrid = HybridRetriever(bm25, vector)
    reranker = CrossEncoderReranker()
    generator = RAGPipeline()
    verifier = FaithfulnessVerifier()

    print(f"Loading {args.max_samples} PubMedQA samples...")
    evaluator = BioASQEvaluator(max_samples=args.max_samples)
    samples = evaluator.load()

    results = {}

    # --- Ablation: 4 retrieval configs ---
    print("\n=== Retrieval Ablation ===")
    configs = {
        "bm25_only":       lambda q, k: bm25.search(q, top_k=k),
        "vector_only":     lambda q, k: vector.search(q, top_k=k),
        "hybrid":          lambda q, k: hybrid.retrieve(q, top_k=k),
        "hybrid_reranked": lambda q, k: reranker.rerank(q, hybrid.retrieve(q, top_k=20), top_k=k),
    }
    for name, fn in configs.items():
        results[name] = run_retrieval_config(name, fn, samples, args.k_values)

    # --- K sensitivity ---
    print("\n=== K Sensitivity (hybrid_reranked) ===")
    k_sensitivity = {}
    for k in args.k_values:
        fn = lambda q, _k=k: reranker.rerank(q, hybrid.retrieve(q, top_k=20), top_k=_k)
        metrics = run_retrieval_config(f"k={k}", lambda q, top_k, _k=k: reranker.rerank(q, hybrid.retrieve(q, top_k=20), top_k=_k), samples, [k])
        k_sensitivity[f"k={k}"] = metrics
    results["k_sensitivity"] = k_sensitivity

    # --- Hallucination eval ---
    print("\n=== Hallucination Evaluation ===")

    def baseline_retrieve(q, k):
        return bm25.search(q, top_k=k)

    def rag_retrieve(q, k):
        return reranker.rerank(q, hybrid.retrieve(q, top_k=20), top_k=k)

    print("  Running baseline (BM25, no reranking)...")
    results["hallucination_baseline"] = run_hallucination_eval(
        samples, baseline_retrieve, generator, verifier, top_k=5
    )
    print("  Running hybrid_reranked...")
    results["hallucination_reranked"] = run_hallucination_eval(
        samples, rag_retrieve, generator, verifier, top_k=5
    )

    # --- Save ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    _print_summary(results)


def _print_summary(results):
    print("\n" + "=" * 55)
    print("ABLATION TABLE")
    print("=" * 55)
    print(f"{'Config':<22} {'NDCG@5':>8} {'Recall@10':>10}")
    print("-" * 55)
    for name in ["bm25_only", "vector_only", "hybrid", "hybrid_reranked"]:
        if name in results:
            m = results[name]
            print(f"{name:<22} {m.get('ndcg@5',0):>8.3f} {m.get('recall@10',0):>10.3f}")

    print("\nHALLUCINATION EVALUATION")
    print("-" * 55)
    for label, key in [("Baseline (BM25)", "hallucination_baseline"),
                       ("Hybrid + Reranked", "hallucination_reranked")]:
        if key in results:
            h = results[key]
            print(f"{label:<22} hallucination={h['hallucination_rate']:.3f}  "
                  f"faithfulness={h['mean_faithfulness_score']:.3f}")


if __name__ == "__main__":
    main()
