from __future__ import annotations

from typing import Any

from transformers import pipeline, Pipeline


PROMPT_TEMPLATE = """Answer the medical question based on the context below.
Context: {context}
Question: {question}
Answer:"""


class RAGPipeline:
    """
    Generates answers from retrieved context using a seq2seq LLM.
    Keeps generation grounded by injecting only the top-K reranked chunks.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        print(f"[RAGPipeline] Loading model: {model_name}")
        self._pipe: Pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        context = self._build_context(chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        output = self._pipe(prompt)
        answer = output[0]["generated_text"].strip()

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": c["chunk_id"],
                    "title": c["title"],
                    "rerank_score": c.get("rerank_score", 0.0),
                }
                for c in chunks
            ],
        }

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]], max_chars: int = 1500) -> str:
        parts = []
        total = 0
        for i, chunk in enumerate(chunks, 1):
            snippet = chunk["text"][:400]
            entry = f"[{i}] {chunk['title']}\n{snippet}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)
