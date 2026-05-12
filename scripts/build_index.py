"""
Builds BM25, MiniLM vector, and PubMedBERT vector indexes from processed articles.
Deletes existing indexes before rebuilding.

Usage:
    python scripts/build_index.py --articles data/processed/articles.jsonl --index-dir data/index
"""

import argparse
import shutil
from pathlib import Path

from medrag.ingestion.pmc_parser import PMCParser
from medrag.indexing.bm25_index import BM25Index
from medrag.indexing.vector_index import VectorIndex


def clear_chroma(chroma_path: str, collection_name: str) -> None:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            client.delete_collection(collection_name)
            print(f"Cleared ChromaDB collection: {collection_name}")
        except Exception:
            pass
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--articles", default="data/processed/articles.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--skip-pubmedbert", action="store_true",
                        help="Skip PubMedBERT index (saves ~15 min)")
    args = parser.parse_args()

    index_path = Path(args.index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # ── Clear old indexes ─────────────────────────────────────────────────────
    bm25_path = index_path / "bm25"
    if bm25_path.exists():
        shutil.rmtree(bm25_path, ignore_errors=True)
        print("Cleared old BM25 index")

    clear_chroma(str(index_path / "chroma_minilm"), "medrag_minilm")
    clear_chroma(str(index_path / "chroma_pubmedbert"), "medrag_pubmedbert")

    # ── Load articles ─────────────────────────────────────────────────────────
    print(f"\nLoading articles from {args.articles}")
    articles = PMCParser.load_jsonl(args.articles)
    print(f"Loaded {len(articles)} articles")

    # ── BM25 ──────────────────────────────────────────────────────────────────
    print("\n[1/3] Building BM25 index...")
    bm25 = BM25Index(chunk_size=args.chunk_size)
    bm25.build(articles)
    bm25.save(f"{args.index_dir}/bm25")
    chunks = bm25._chunks

    # ── MiniLM vector ─────────────────────────────────────────────────────────
    print("\n[2/3] Building MiniLM vector index...")
    minilm = VectorIndex(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="medrag_minilm",
        persist_dir=f"{args.index_dir}/chroma_minilm",
    )
    minilm.build(chunks)
    print("[MiniLM] Done")

    # ── PubMedBERT vector ─────────────────────────────────────────────────────
    if not args.skip_pubmedbert:
        print("\n[3/3] Building PubMedBERT vector index (medical domain)...")
        pubmedbert = VectorIndex(
            model_name="NeuML/pubmedbert-base-embeddings",
            collection_name="medrag_pubmedbert",
            persist_dir=f"{args.index_dir}/chroma_pubmedbert",
        )
        pubmedbert.build(chunks)
        print("[PubMedBERT] Done")
    else:
        print("\n[3/3] PubMedBERT skipped (--skip-pubmedbert)")

    print(f"\nIndex build complete: {len(chunks)} chunks from {len(articles)} articles")


if __name__ == "__main__":
    main()
