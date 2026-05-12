"""
End-to-end RAG evaluation using Fireworks LLM + NLI faithfulness check.

Pipeline:
  Question → hybrid retrieval (top-20) → cross-encoder rerank (top-5)
           → Fireworks LLM generates answer
           → DeBERTa NLI checks faithfulness of generated answer against retrieved chunks

Questions are real medical questions covering topics in the NFCorpus corpus
(nutrition, cancer, cardiovascular disease, diabetes, diet, lifestyle).

Usage:
    python scripts/evaluate_generation.py \
        --index-dir data/index \
        --max-samples 20
"""

import argparse
import json
from pathlib import Path

import numpy as np

from medrag.indexing.bm25_index import BM25Index
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import CrossEncoderReranker
from medrag.generation.fireworks import FireworksGenerator
from medrag.verification.faithfulness import FaithfulnessVerifier


# ── 20 medical questions covering NFCorpus topics ─────────────────────────────
MEDICAL_QUESTIONS = [
    "Does dietary fiber intake reduce the risk of colorectal cancer?",
    "Is there a link between red meat consumption and cardiovascular disease?",
    "Does vitamin D deficiency increase the risk of type 2 diabetes?",
    "Can omega-3 fatty acids reduce inflammation in rheumatoid arthritis?",
    "Does obesity increase the risk of developing type 2 diabetes?",
    "Is physical activity associated with reduced risk of breast cancer?",
    "Does high sodium intake increase blood pressure?",
    "Can antioxidants in fruits and vegetables reduce cancer risk?",
    "Does alcohol consumption increase the risk of liver disease?",
    "Is there a relationship between gut microbiome composition and obesity?",
    "Does smoking increase the risk of cardiovascular disease?",
    "Can a Mediterranean diet reduce the risk of heart disease?",
    "Does sleep deprivation negatively affect metabolic health?",
    "Is there evidence that green tea consumption reduces cancer risk?",
    "Does high sugar intake contribute to insulin resistance?",
    "Can probiotics improve gut health and reduce systemic inflammation?",
    "Does air pollution exposure increase the risk of lung cancer?",
    "Is there a link between vitamin C intake and immune function?",
    "Does regular aerobic exercise lower blood pressure?",
    "Can caloric restriction improve longevity-related health markers?",
]


def dedupe(results: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in results:
        if r["article_id"] not in seen:
            seen.add(r["article_id"])
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir",   default="data/index")
    parser.add_argument("--output",      default="results/generation_results.json")
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    questions = MEDICAL_QUESTIONS[: args.max_samples]

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading BM25 index...")
    bm25 = BM25Index()
    bm25.load(f"{args.index_dir}/bm25")

    print("Loading MiniLM vector index...")
    from medrag.indexing.vector_index import VectorIndex
    vector = VectorIndex(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="medrag_minilm",
        persist_dir=f"{args.index_dir}/chroma_minilm",
    )

    hybrid   = HybridRetriever(bm25, vector)
    reranker = CrossEncoderReranker()

    print("Loading Fireworks generator...")
    generator = FireworksGenerator()

    print("Loading NLI faithfulness verifier (DeBERTa)...")
    verifier = FaithfulnessVerifier()
    print("All models loaded.\n")

    outputs     = []
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    faith_scores = []

    print("=== End-to-End RAG Evaluation ===\n")

    for i, question in enumerate(questions):
        try:
            # Retrieve → rerank
            candidates = dedupe(hybrid.retrieve(question, top_k=20))
            top_chunks  = reranker.rerank(question, candidates, top_k=5)

            # Generate answer
            answer = generator.generate(question, top_chunks)

            # Verify faithfulness of generated answer against retrieved chunks
            result = verifier.verify(answer, top_chunks)

            risk_counts[result.risk.value] += 1
            faith_scores.append(result.faithfulness_score)

            print(f"[{i+1}/{len(questions)}] Q: {question}")
            print(f"  Answer : {answer[:120]}...")
            print(f"  Faith  : {result.faithfulness_score:.3f}  Risk: {result.risk.value}\n")

            outputs.append({
                "question":           question,
                "answer":             answer,
                "faithfulness_score": result.faithfulness_score,
                "hallucination_risk": result.risk.value,
                "supporting_chunks":  result.supporting_chunks,
            })

        except Exception as e:
            print(f"  [{i+1}] Failed: {e}")
            outputs.append({"question": question, "error": str(e)})

    # ── Summary ───────────────────────────────────────────────────────────────
    n              = len(questions)
    high_risk_rate = round(risk_counts["HIGH"] / n, 4) if n > 0 else 0
    mean_faith     = round(float(np.mean(faith_scores)), 4) if faith_scores else 0

    print("=== Summary ===")
    print(f"  Samples evaluated : {n}")
    print(f"  LOW  risk answers : {risk_counts['LOW']}")
    print(f"  MED  risk answers : {risk_counts['MEDIUM']}")
    print(f"  HIGH risk answers : {risk_counts['HIGH']}")
    print(f"  High-risk rate    : {high_risk_rate:.1%}")
    print(f"  Mean faithfulness : {mean_faith:.3f}")

    summary = {
        "n_samples":         n,
        "risk_counts":       risk_counts,
        "high_risk_rate":    high_risk_rate,
        "mean_faithfulness": mean_faith,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": outputs}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
