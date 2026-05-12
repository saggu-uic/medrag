from medrag.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    hallucination_rate,
    mean_faithfulness_score,
)


def test_ndcg_perfect_retrieval():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=5) == 1.0


def test_ndcg_no_hits():
    assert ndcg_at_k(["x", "y"], {"a", "b"}, k=5) == 0.0


def test_recall_full():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=5) == 1.0


def test_recall_partial():
    score = recall_at_k(["a", "x", "y"], {"a", "b"}, k=3)
    assert score == 0.5


def test_hallucination_rate():
    results = [
        {"unsupported": True},
        {"unsupported": False},
        {"unsupported": True},
        {"unsupported": False},
    ]
    assert hallucination_rate(results) == 0.5


def test_mean_faithfulness():
    results = [{"faithfulness_score": 0.8}, {"faithfulness_score": 0.6}]
    assert mean_faithfulness_score(results) == 0.7
