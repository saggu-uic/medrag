"""
Downloads NFCorpus (BeIR biomedical retrieval benchmark) from HuggingFace.
Produces articles.jsonl for build_index.py AND eval_queries.json for evaluation.

NFCorpus is the standard benchmark for biomedical IR — designed so that
retrieval is genuinely hard and realistic NDCG scores are 0.28-0.40.

Usage:
    python scripts/bootstrap_nfcorpus.py
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ── 1. Corpus ──────────────────────────────────────────────────────────────
    print("Downloading NFCorpus corpus (BeIR/nfcorpus)...")
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")

    count = 0
    with open("data/processed/articles.jsonl", "w", encoding="utf-8") as f:
        for row in tqdm(corpus_ds, desc="Converting corpus"):
            article = {
                "pmc_id":           str(row["_id"]),
                "title":            row.get("title", ""),
                "abstract":         "",
                "body":             row.get("text", ""),
                "journal":          "",
                "publication_year": 2020,
                "author_count":     3,
                "citation_count":   0,
            }
            f.write(json.dumps(article) + "\n")
            count += 1

    print(f"Saved {count} corpus articles → data/processed/articles.jsonl")

    # Build a fast lookup: doc_id → body text (for ideal_answer in NLI eval)
    print("Building corpus text lookup...")
    corpus_text: dict[str, str] = {}
    for row in corpus_ds:
        corpus_text[str(row["_id"])] = row.get("text", "")

    # ── 2. Queries + qrels (test split) ────────────────────────────────────────
    print("\nDownloading NFCorpus queries...")
    queries_ds = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    queries_map = {str(row["_id"]): row["text"] for row in queries_ds}

    print("Downloading NFCorpus qrels (test)...")
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels", split="test")

    # Group relevant doc IDs per query (only score > 0)
    qrels: dict[str, list[str]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if int(row["score"]) > 0:
            qrels.setdefault(qid, []).append(cid)

    eval_rows = []
    for qid, relevant_ids in qrels.items():
        query_text = queries_map.get(qid, "")
        if not query_text or not relevant_ids:
            continue

        # ideal_answer = body text of first relevant doc (used for NLI faithfulness eval).
        # Checks: does our retrieved context entail the content of the known-relevant doc?
        first_rel_text = corpus_text.get(relevant_ids[0], "")
        ideal_answer = first_rel_text[:600] if first_rel_text else query_text

        eval_rows.append({
            "question_id":      qid,
            "question":         query_text,
            "ideal_answer":     ideal_answer,
            "relevant_doc_ids": relevant_ids,
            "context_snippets": [],
        })

    with open("data/processed/eval_queries.json", "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2)

    print(f"Saved {len(eval_rows)} eval queries → data/processed/eval_queries.json")
    print(
        f"\nCorpus: {count} docs | Eval queries: {len(eval_rows)} | "
        f"Avg relevant per query: {sum(len(r['relevant_doc_ids']) for r in eval_rows)/max(len(eval_rows),1):.1f}"
    )
    print("\nNext steps:")
    print("  python scripts/build_index.py --articles data/processed/articles.jsonl --index-dir data/index")
    print("  python scripts/evaluate_retrieval.py --index-dir data/index --eval-set data/processed/eval_queries.json --max-samples 150")


if __name__ == "__main__":
    main()
