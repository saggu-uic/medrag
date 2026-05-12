from unittest.mock import MagicMock

from medrag.retrieval.hybrid import HybridRetriever


def _make_results(ids, scores, score_key):
    return [{
        "chunk_id": cid, "text": f"text {cid}", "article_id": cid,
        "title": f"Title {cid}", "journal": "", "author_count": 1,
        "citation_count": 5, "publication_year": 2020,
        score_key: score,
    } for cid, score in zip(ids, scores)]


def test_hybrid_retriever_deduplicates():
    bm25 = MagicMock()
    bm25.search.return_value = _make_results(["a", "b", "c"], [3.0, 2.0, 1.0], "bm25_score")

    vector = MagicMock()
    vector.search.return_value = _make_results(["b", "c", "d"], [0.9, 0.8, 0.7], "vector_score")

    retriever = HybridRetriever(bm25, vector)
    results = retriever.retrieve("test query", top_k=10)

    ids = [r["chunk_id"] for r in results]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in results"
    assert len(results) <= 10


def test_hybrid_retriever_respects_top_k():
    bm25 = MagicMock()
    bm25.search.return_value = _make_results(
        [str(i) for i in range(20)],
        [float(i) for i in range(20)],
        "bm25_score",
    )
    vector = MagicMock()
    vector.search.return_value = []

    retriever = HybridRetriever(bm25, vector)
    results = retriever.retrieve("query", top_k=5)
    assert len(results) == 5


def test_hybrid_score_is_weighted_combination():
    bm25 = MagicMock()
    bm25.search.return_value = _make_results(["x"], [1.0], "bm25_score")

    vector = MagicMock()
    vector.search.return_value = _make_results(["x"], [1.0], "vector_score")

    retriever = HybridRetriever(bm25, vector, bm25_weight=0.4, vector_weight=0.6)
    results = retriever.retrieve("query", top_k=1)

    assert results[0]["hybrid_score"] == 1.0
