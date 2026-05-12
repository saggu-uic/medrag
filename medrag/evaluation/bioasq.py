from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class BioASQSample:
    question_id: str
    question: str
    ideal_answer: str
    relevant_doc_ids: set[str]
    question_type: str


class BioASQEvaluator:
    """
    Loads the BioASQ-Task B dataset and provides evaluation utilities.
    Uses the 'pubmed_qa' dataset from HuggingFace as a BioASQ proxy
    (same domain, publicly available without registration).
    """

    def __init__(self, split: str = "train", max_samples: int | None = None):
        self.split = split
        self.max_samples = max_samples
        self._samples: list[BioASQSample] = []

    def load(self) -> list[BioASQSample]:
        print("[BioASQEvaluator] Loading PubMedQA dataset...")
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split=self.split)

        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        self._samples = []
        for i, row in enumerate(dataset):
            self._samples.append(BioASQSample(
                question_id=str(row.get("pubid", i)),
                question=row["question"],
                ideal_answer=row["long_answer"],
                relevant_doc_ids={str(row.get("pubid", i))},
                question_type=row.get("final_decision", "unknown"),
            ))

        print(f"[BioASQEvaluator] Loaded {len(self._samples)} samples")
        return self._samples

    @property
    def samples(self) -> list[BioASQSample]:
        if not self._samples:
            self.load()
        return self._samples

    def run_retrieval_eval(
        self,
        retrieve_fn,
        top_k_values: list[int] = [5, 10, 20],
    ) -> dict[str, Any]:
        from medrag.evaluation.metrics import aggregate_retrieval_metrics

        all_retrieved, all_relevant = [], []

        for sample in self.samples:
            results = retrieve_fn(sample.question, top_k=max(top_k_values))
            retrieved_ids = [r["article_id"] for r in results]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(sample.relevant_doc_ids)

        return aggregate_retrieval_metrics(all_retrieved, all_relevant, k_values=top_k_values)

    def run_generation_eval(
        self,
        retrieve_fn,
        generate_fn,
        verify_fn,
        top_k: int = 5,
    ) -> dict[str, Any]:
        from medrag.evaluation.metrics import hallucination_rate, mean_faithfulness_score

        verification_results = []

        for sample in self.samples:
            chunks = retrieve_fn(sample.question, top_k=top_k)
            generation = generate_fn(sample.question, chunks)
            verification = verify_fn(generation["answer"], chunks)
            verification_results.append(verification.to_dict())

        return {
            "hallucination_rate": hallucination_rate(verification_results),
            "mean_faithfulness_score": mean_faithfulness_score(verification_results),
            "n_samples": len(verification_results),
        }
