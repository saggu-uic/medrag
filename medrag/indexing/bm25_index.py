from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from medrag.ingestion.pmc_parser import Article


STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
})


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


class Chunk:
    def __init__(self, chunk_id: str, article_id: str, title: str, text: str,
                 journal: str, author_count: int, citation_count: int,
                 publication_year: int | None):
        self.chunk_id = chunk_id
        self.article_id = article_id
        self.title = title
        self.text = text
        self.journal = journal
        self.author_count = author_count
        self.citation_count = citation_count
        self.publication_year = publication_year

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


class BM25Index:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def _split_into_chunks(self, article: Article) -> list[Chunk]:
        """
        Index only the body text (evidence passages), NOT title/abstract.
        This keeps the question (= title) out of the indexed text so retrieval
        is a genuine semantic matching task, not trivial keyword lookup.
        Falls back to abstract if body is empty.
        """
        chunks = []

        source_text = article.body.strip() if article.body.strip() else article.abstract.strip()
        words = source_text.split()

        if not words:
            return chunks

        start, idx = 0, 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(
                chunk_id=f"{article.pmc_id}_{idx}",
                article_id=article.pmc_id,
                title=article.title,
                text=chunk_text,
                journal=article.journal,
                author_count=article.author_count,
                citation_count=article.citation_count,
                publication_year=article.publication_year,
            ))
            start += self.chunk_size - self.chunk_overlap
            idx += 1

        return chunks

    def build(self, articles: list[Article]) -> None:
        self._chunks = []
        for article in tqdm(articles, desc="Chunking articles"):
            self._chunks.extend(self._split_into_chunks(article))

        tokenized = [tokenize(c.text) for c in tqdm(self._chunks, desc="Tokenizing")]
        self._bm25 = BM25Okapi(tokenized)
        print(f"[BM25Index] Built index with {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")

        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)

        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {**self._chunks[i].to_dict(), "bm25_score": float(scores[i])}
            for i in top_indices
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        print(f"[BM25Index] Saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path / "bm25.pkl", "rb") as f:
            self._bm25 = pickle.load(f)
        self._chunks = []
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                self._chunks.append(Chunk.from_dict(json.loads(line)))
        print(f"[BM25Index] Loaded {len(self._chunks)} chunks from {path}")
