"""
Downloads PubMedQA from HuggingFace and converts it to medrag's Article format.
Saves articles.jsonl ready for build_index.py.

Usage:
    python scripts/bootstrap_data.py --output data/processed/articles.jsonl --max-articles 1000
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/articles.jsonl")
    parser.add_argument("--max-articles", type=int, default=1000)
    args = parser.parse_args()

    print("Downloading PubMedQA (pqa_labeled) from HuggingFace...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Use first max_articles as the index corpus (train split)
    # Remaining articles are reserved for evaluation (test split)
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc="Converting articles"):
            if count >= args.max_articles:
                break

            contexts = row.get("context", {})
            if isinstance(contexts, dict):
                contexts_list = contexts.get("contexts", [])
                mesh_terms = contexts.get("meshes", [])
            else:
                contexts_list = []
                mesh_terms = []

            body = " ".join(contexts_list) if contexts_list else ""
            journal = mesh_terms[0] if mesh_terms else ""

            article = {
                "pmc_id": str(row.get("pubid", count)),
                "title": row["question"],
                "abstract": row.get("long_answer", ""),
                "body": body,
                "journal": journal,
                "publication_year": 2020,
                "author_count": 3,
                "citation_count": len(contexts_list),
            }

            f.write(json.dumps(article) + "\n")
            count += 1

    print(f"Saved {count} articles to {args.output}")


if __name__ == "__main__":
    main()
