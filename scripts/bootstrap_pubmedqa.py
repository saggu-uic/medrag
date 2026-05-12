"""
Downloads PubMedQA (pqa_labeled) from HuggingFace.
Produces articles.jsonl for build_index.py AND eval_queries.json for evaluation.

PubMedQA is a biomedical QA benchmark with real medical yes/no questions
paired directly with PubMed abstracts that answer them.

Questions like:
  "Does metformin improve glycemic control in type 2 diabetes?"
  "Is low-dose aspirin effective for primary prevention of cardiovascular events?"

Usage:
    python scripts/bootstrap_pubmedqa.py
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Downloading PubMedQA (pqa_labeled) from HuggingFace...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"Loaded {len(ds)} labeled QA pairs")

    articles = []
    eval_rows = []

    print("Processing articles and queries...")
    for row in tqdm(ds):
        pubmed_id = str(row["pubid"])

        # ── Article ────────────────────────────────────────────────────────────
        # Contexts = list of sentences from the abstract
        contexts = row.get("context", {})
        sentences = contexts.get("contexts", [])
        body_text = " ".join(sentences) if sentences else ""
        abstract = row.get("long_answer", "")

        article = {
            "pmc_id":           pubmed_id,
            "title":            row.get("question", ""),
            "abstract":         abstract,
            "body":             body_text,
            "journal":          "",
            "publication_year": 2020,
            "author_count":     3,
            "citation_count":   0,
        }
        articles.append(article)

        # ── Eval query ─────────────────────────────────────────────────────────
        question   = row.get("question", "")
        long_answer = row.get("long_answer", "")  # detailed answer from abstract
        final_answer = row.get("final_decision", "")  # yes / no / maybe

        if not question:
            continue

        eval_rows.append({
            "question_id":      pubmed_id,
            "question":         question,
            "ideal_answer":     long_answer,
            "final_decision":   final_answer,
            "relevant_doc_ids": [pubmed_id],
        })

    # ── Save articles ──────────────────────────────────────────────────────────
    with open("data/processed/articles.jsonl", "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")
    print(f"Saved {len(articles)} articles → data/processed/articles.jsonl")

    # ── Save eval queries ──────────────────────────────────────────────────────
    with open("data/processed/eval_queries.json", "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2)
    print(f"Saved {len(eval_rows)} eval queries → data/processed/eval_queries.json")

    yes = sum(1 for r in eval_rows if r["final_decision"] == "yes")
    no  = sum(1 for r in eval_rows if r["final_decision"] == "no")
    maybe = sum(1 for r in eval_rows if r["final_decision"] == "maybe")
    print(f"\nDataset: {len(articles)} docs | {len(eval_rows)} queries")
    print(f"Answers: yes={yes}  no={no}  maybe={maybe}")
    print("\nNext steps:")
    print("  python scripts/build_index.py")
    print("  python scripts/evaluate_retrieval.py --max-samples 100")
    print("  python scripts/evaluate_generation.py --max-samples 20")


if __name__ == "__main__":
    main()
