"""
End-to-end RAG evaluation using Fireworks LLM + NLI faithfulness check.

Pipeline:
  Question → retrieval (config) → cross-encoder rerank (top-5)
           → Fireworks LLM generates answer
           → DeBERTa NLI checks faithfulness of generated answer against retrieved chunks

Run with best retrieval config:
    python scripts/evaluate_generation.py --config hybrid_reranked

Run with worst retrieval config (for comparison):
    python scripts/evaluate_generation.py --config dense_pubmedbert

Questions include 20 standard medical questions + 4 trick questions
(half-supported by corpus) to stress-test the NLI faithfulness check.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from medrag.indexing.bm25_index import BM25Index
from medrag.indexing.vector_index import VectorIndex
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import CrossEncoderReranker
from medrag.generation.fireworks import FireworksGenerator
from medrag.verification.faithfulness import FaithfulnessVerifier


# ── Standard medical questions (well-covered by NFCorpus) ─────────────────────
STANDARD_QUESTIONS = [
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

# ── Trick questions (half-supported — first half is in corpus, second is not) ──
# These stress-test the NLI check. LLM answering the unsupported half = hallucination.
TRICK_QUESTIONS = [
    # Cardiovascular from red meat = supported. Improves memory = NOT in corpus.
    "Does red meat consumption increase cardiovascular disease risk and also improve memory and cognitive function?",
    # Obesity + diabetes = supported. Obesity improves immune response = medically wrong.
    "Is obesity associated with type 2 diabetes and also linked to a stronger immune response?",
    # Green tea + cancer = partially supported. Green tea reverses liver cirrhosis = NOT supported.
    "Does green tea consumption reduce cancer risk and also reverse existing liver cirrhosis?",
    # Alcohol + liver disease = supported. Alcohol enhances cognitive performance = wrong.
    "Does alcohol consumption increase liver disease risk and also enhance long-term cognitive performance?",
]

ALL_QUESTIONS = STANDARD_QUESTIONS + TRICK_QUESTIONS


def dedupe(results: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in results:
        if r["article_id"] not in seen:
            seen.add(r["article_id"])
            out.append(r)
    return out


def build_retriever(config: str, index_dir: str):
    """Load the retrieval components for the given config."""
    print(f"Loading BM25 index...")
    bm25 = BM25Index()
    bm25.load(f"{index_dir}/bm25")

    reranker = CrossEncoderReranker()

    if config == "dense_pubmedbert":
        print("Loading PubMedBERT vector index...")
        vec = VectorIndex(
            model_name="NeuML/pubmedbert-base-embeddings",
            collection_name="medrag_pubmedbert",
            persist_dir=f"{index_dir}/chroma_pubmedbert",
        )
        def retrieve_fn(question):
            candidates = dedupe(vec.search(question, top_k=20))
            return reranker.rerank(question, candidates, top_k=5)

    elif config == "hybrid_reranked":
        print("Loading MiniLM vector index...")
        vec = VectorIndex(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            collection_name="medrag_minilm",
            persist_dir=f"{index_dir}/chroma_minilm",
        )
        hybrid = HybridRetriever(bm25, vec)
        def retrieve_fn(question):
            candidates = dedupe(hybrid.retrieve(question, top_k=20))
            return reranker.rerank(question, candidates, top_k=5)

    else:
        raise ValueError(f"Unknown config: {config}. Use 'hybrid_reranked' or 'dense_pubmedbert'")

    return retrieve_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir",   default="data/index")
    parser.add_argument("--output-dir",  default="results")
    parser.add_argument("--config",      default="hybrid_reranked",
                        choices=["hybrid_reranked", "dense_pubmedbert"],
                        help="Retrieval config to use for generation eval")
    parser.add_argument("--max-samples", type=int, default=24,
                        help="Number of questions (max 24 = 20 standard + 4 trick)")
    args = parser.parse_args()

    questions = ALL_QUESTIONS[: args.max_samples]
    output_path = f"{args.output_dir}/generation_{args.config}.json"

    print(f"Config       : {args.config}")
    print(f"Questions    : {len(questions)} ({len(STANDARD_QUESTIONS)} standard + {len(TRICK_QUESTIONS)} trick)")
    print(f"Output       : {output_path}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    retrieve_fn = build_retriever(args.config, args.index_dir)

    print("Loading Fireworks generator...")
    generator = FireworksGenerator()

    print("Loading NLI faithfulness verifier (DeBERTa)...")
    verifier = FaithfulnessVerifier()
    print("All models loaded.\n")

    outputs      = []
    risk_counts  = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    faith_scores = []

    print(f"=== Generation Eval — {args.config} ===\n")

    for i, question in enumerate(questions):
        q_type = "TRICK" if question in TRICK_QUESTIONS else "standard"
        try:
            top_chunks = retrieve_fn(question)
            answer     = generator.generate(question, top_chunks)
            result     = verifier.verify(answer, top_chunks)

            risk_counts[result.risk.value] += 1
            faith_scores.append(result.faithfulness_score)

            print(f"[{i+1}/{len(questions)}] [{q_type}] {question}")
            print(f"  Answer : {answer[:120]}...")
            print(f"  Faith  : {result.faithfulness_score:.3f}  Risk: {result.risk.value}\n")

            outputs.append({
                "question":           question,
                "question_type":      q_type,
                "answer":             answer,
                "faithfulness_score": result.faithfulness_score,
                "hallucination_risk": result.risk.value,
                "supporting_chunks":  result.supporting_chunks,
            })

        except Exception as e:
            print(f"  [{i+1}] Failed: {e}")
            outputs.append({"question": question, "question_type": q_type, "error": str(e)})

    # ── Summary ───────────────────────────────────────────────────────────────
    n              = len(questions)
    high_risk_rate = round(risk_counts["HIGH"] / n, 4) if n > 0 else 0
    mean_faith     = round(float(np.mean(faith_scores)), 4) if faith_scores else 0

    # Separate trick question stats
    trick_outputs   = [o for o in outputs if o.get("question_type") == "TRICK" and "error" not in o]
    trick_high_risk = sum(1 for o in trick_outputs if o.get("hallucination_risk") == "HIGH")
    trick_medium    = sum(1 for o in trick_outputs if o.get("hallucination_risk") == "MEDIUM")

    print(f"=== Summary — {args.config} ===")
    print(f"  Samples evaluated  : {n}")
    print(f"  Mean faithfulness  : {mean_faith:.3f}")
    print(f"  Risk counts        : {risk_counts}")
    print(f"  High-risk rate     : {high_risk_rate:.1%}")
    print(f"  Trick Q — HIGH risk: {trick_high_risk}/{len(trick_outputs)}")
    print(f"  Trick Q — MEDIUM   : {trick_medium}/{len(trick_outputs)}")

    summary = {
        "config":            args.config,
        "n_samples":         n,
        "risk_counts":       risk_counts,
        "high_risk_rate":    high_risk_rate,
        "mean_faithfulness": mean_faith,
        "trick_high_risk":   trick_high_risk,
        "trick_medium_risk": trick_medium,
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": outputs}, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
