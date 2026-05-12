from unittest.mock import MagicMock, patch

from medrag.verification.faithfulness import FaithfulnessVerifier, HallucinationRisk


def _make_chunks(n=3):
    return [{"chunk_id": f"c{i}", "text": f"source text {i}"} for i in range(n)]


def test_high_risk_when_no_chunks():
    verifier = FaithfulnessVerifier.__new__(FaithfulnessVerifier)
    verifier.entailment_threshold = 0.5
    verifier.low_risk_threshold = 0.7
    verifier.medium_risk_threshold = 0.4
    verifier._model = MagicMock()

    result = verifier.verify("some answer", [])
    assert result.risk == HallucinationRisk.HIGH
    assert result.unsupported is True


def test_low_risk_with_high_entailment():
    verifier = FaithfulnessVerifier.__new__(FaithfulnessVerifier)
    verifier.entailment_threshold = 0.5
    verifier.low_risk_threshold = 0.7
    verifier.medium_risk_threshold = 0.4
    verifier._model = MagicMock()
    verifier._model.predict.return_value = [0.85, 0.90, 0.80]

    result = verifier.verify("answer", _make_chunks(3))
    assert result.risk == HallucinationRisk.LOW
    assert result.faithfulness_score >= 0.7
    assert result.unsupported is False


def test_high_risk_with_low_entailment():
    verifier = FaithfulnessVerifier.__new__(FaithfulnessVerifier)
    verifier.entailment_threshold = 0.5
    verifier.low_risk_threshold = 0.7
    verifier.medium_risk_threshold = 0.4
    verifier._model = MagicMock()
    verifier._model.predict.return_value = [0.1, 0.2, 0.15]

    result = verifier.verify("answer", _make_chunks(3))
    assert result.risk == HallucinationRisk.HIGH
    assert result.unsupported is True


def test_medium_risk_boundary():
    verifier = FaithfulnessVerifier.__new__(FaithfulnessVerifier)
    verifier.entailment_threshold = 0.5
    verifier.low_risk_threshold = 0.7
    verifier.medium_risk_threshold = 0.4
    verifier._model = MagicMock()
    verifier._model.predict.return_value = [0.55, 0.45, 0.50]

    result = verifier.verify("answer", _make_chunks(3))
    assert result.risk == HallucinationRisk.MEDIUM
