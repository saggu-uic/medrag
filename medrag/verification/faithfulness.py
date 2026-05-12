from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from sentence_transformers import CrossEncoder


class HallucinationRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class VerificationResult:
    answer: str
    faithfulness_score: float
    risk: HallucinationRisk
    supporting_chunks: list[str]
    unsupported: bool

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "hallucination_risk": self.risk.value,
            "supporting_chunk_ids": self.supporting_chunks,
            "unsupported": self.unsupported,
        }


class FaithfulnessVerifier:
    """
    Scores whether a generated answer is entailed by the retrieved sources
    using an NLI cross-encoder (DeBERTa).

    Faithfulness score = mean entailment probability across source chunks.
    Risk tiers:
        LOW    — score >= 0.7  (answer well-supported)
        MEDIUM — score >= 0.4  (partial support)
        HIGH   — score <  0.4  (answer likely hallucinated)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        entailment_threshold: float = 0.5,
        low_risk_threshold: float = 0.7,
        medium_risk_threshold: float = 0.4,
    ):
        print(f"[FaithfulnessVerifier] Loading NLI model: {model_name}")
        self._model = CrossEncoder(model_name)
        self.entailment_threshold = entailment_threshold
        self.low_risk_threshold = low_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold

    def verify(self, answer: str, chunks: list[dict[str, Any]]) -> VerificationResult:
        if not answer or not chunks:
            return VerificationResult(
                answer=answer,
                faithfulness_score=0.0,
                risk=HallucinationRisk.HIGH,
                supporting_chunks=[],
                unsupported=True,
            )

        pairs = [(chunk["text"], answer) for chunk in chunks]
        raw_scores = self._model.predict(pairs)

        # DeBERTa NLI outputs [contradiction, neutral, entailment] per pair
        import numpy as np

        def _softmax(x, axis=-1):
            e = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)

        raw_scores = np.array(raw_scores)
        if raw_scores.ndim == 2:
            probs = _softmax(raw_scores, axis=1)
            entailment_scores = probs[:, 2].tolist()
        else:
            entailment_scores = _softmax(raw_scores).tolist()

        supporting = [
            chunks[i]["chunk_id"]
            for i, score in enumerate(entailment_scores)
            if score >= self.entailment_threshold
        ]

        faithfulness_score = max(entailment_scores) if entailment_scores else 0.0

        risk = self._assign_risk(faithfulness_score)

        return VerificationResult(
            answer=answer,
            faithfulness_score=faithfulness_score,
            risk=risk,
            supporting_chunks=supporting,
            unsupported=len(supporting) == 0,
        )

    def _assign_risk(self, score: float) -> HallucinationRisk:
        if score >= self.low_risk_threshold:
            return HallucinationRisk.LOW
        if score >= self.medium_risk_threshold:
            return HallucinationRisk.MEDIUM
        return HallucinationRisk.HIGH
