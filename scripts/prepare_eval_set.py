"""
Creates an evaluation set from PubMedQA articles that ARE in the index.

Design:
  - The index contains body text (evidence passages) for articles 0..index_size-1.
  - Queries are the questions associated with those articles.
  - The question is NOT in the indexed text (only body/context passages are indexed),
    so retrieval must genuinely match question semantics against evidence passages.
  - relevant_doc_ids = [pmc_id of the source article]

Usage:
    python scripts/prepare_eval_set.py --output data/processed/eval_queries.json
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/eval_queries.json")
    parser.add_argument("--index-size", type=int, default=700,
                        help="Number of articles in the index (eval queries come from these)")
    parser.add_argument("--eval-size", type=int, default=200,
                        help="Number of eval queries to sample from indexed articles")
    args = parser.parse_args()

    print("Downloading PubMedQA...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    # Sample eval queries from the SAME articles that are in the index.
    # The question is the title; the indexed text is the body (evidence passages).
    # Retrieval must match question against evidence — a real semantic challenge.
    eval_rows = []
    for i, row in enumerate(dataset):
        if i >= args.index_size:
            break
        if len(eval_rows) >= args.eval_size:
            break

        contexts = row.get("context", {})
        contexts_list = contexts.get("contexts", []) if isinstance(contexts, dict) else []

        # Skip articles with no body text (nothing was indexed for them)
        if not contexts_list:
            continue

        eval_rows.append({
            "question_id": str(row.get("pubid", i)),
            "question": row["question"],
            "ideal_answer": row.get("long_answer", ""),
            "relevant_doc_ids": [str(row.get("pubid", i))],
            "context_snippets": contexts_list[:2],
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2)

    print(f"Saved {len(eval_rows)} eval queries to {args.output}")
    print(f"Queries come from indexed articles → retrieval is non-trivial (question vs. evidence body text)")


if __name__ == "__main__":
    main()
