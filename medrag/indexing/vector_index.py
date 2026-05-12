from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from medrag.indexing.bm25_index import Chunk


class VectorIndex:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "medrag_articles",
        persist_dir: str | Path | None = None,
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_dir = str(persist_dir) if persist_dir else None
        self._model: SentenceTransformer | None = None
        self._collection = None
        self._client = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[VectorIndex] Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_collection(self):
        if self._collection is None:
            if self.persist_dir:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def build(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        model = self._get_model()
        collection = self._get_collection()

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()

            collection.add(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "article_id": c.article_id,
                        "title": c.title,
                        "journal": c.journal,
                        "author_count": c.author_count,
                        "citation_count": c.citation_count,
                        "publication_year": c.publication_year or 0,
                    }
                    for c in batch
                ],
            )

        print(f"[VectorIndex] Indexed {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        model = self._get_model()
        collection = self._get_collection()

        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "article_id": meta["article_id"],
                "title": meta["title"],
                "journal": meta["journal"],
                "author_count": meta["author_count"],
                "citation_count": meta["citation_count"],
                "publication_year": meta["publication_year"],
                "vector_score": 1.0 - (results["distances"][0][i] if results.get("distances") else i / top_k),
            })

        return output
