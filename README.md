# MedRAG — Medical Retrieval Augmented Generation

> Prototype clinical QA system built to evaluate retrieval strategies for medical literature search.

Retrieval evaluated on **NFCorpus** (BeIR biomedical IR benchmark — 3,633 PubMed abstracts, 323 queries).  
Generation evaluated on 20 real medical questions answered from the NFCorpus corpus.

---

## System Pipeline

```
Question
   │
   ├──► BM25 (keyword)       ─┐
   ├──► MiniLM (semantic)    ─┼──► Hybrid Fusion ──► Top 20 ──► Cross-Encoder Reranker ──► Top 5
   └──► PubMedBERT (medical) ─┘                                 (ms-marco-MiniLM-L-6-v2)      │
                                                                                               ▼
                                                                                      Fireworks LLM
                                                                                  (Llama 3.3 70B Instruct)
                                                                                               │
                                                                                               ▼
                                                                                      DeBERTa NLI
                                                                                      Faithfulness Check
                                                                                               │
                                                                                               ▼
                                                                              Answer + Risk (LOW / MEDIUM / HIGH)
```

---

## Retrieval Ablation (10 configs compared)

| # | Config | Description |
|---|--------|-------------|
| 1 | `bm25_only` | BM25 keyword search baseline |
| 2 | `bm25_reranked` | BM25 + cross-encoder reranker |
| 3 | `dense_minilm` | MiniLM dense vectors only |
| 4 | `dense_minilm_reranked` | MiniLM + cross-encoder reranker |
| 5 | `dense_pubmedbert` | PubMedBERT dense vectors only |
| 6 | `dense_pubmedbert_reranked` | PubMedBERT + cross-encoder reranker |
| 7 | `hybrid_minilm` | BM25 + MiniLM fusion |
| 8 | `hybrid_pubmedbert` | BM25 + PubMedBERT fusion |
| 9 | `hybrid_reranked` | BM25 + MiniLM fusion + cross-encoder reranker |
| 10 | `hybrid_pubmed_reranked` | BM25 + PubMedBERT fusion + cross-encoder reranker |

Metrics: **NDCG@5, NDCG@10, Recall@10** with **95% Bootstrap CI** (1000 samples, n=100 queries)

---

## Limitations

- **NLI faithfulness check is conservative** — with a strict context-grounded prompt, a well-aligned LLM (Llama 3.3 70B) correctly defers to retrieved context rather than asserting unsupported claims, resulting in 0% hallucination even on adversarial questions. Hallucination would surface with a weaker or unaligned model.
- **NLI scores "I cannot answer" as LOW risk** — when retrieved chunks lack relevant information, the LLM says "context insufficient" which is technically faithful but uninformative. The faithfulness score does not distinguish between a good grounded answer and a refusal.
- **NFCorpus retrieval scores are moderate** — NDCG@5 range of 0.34–0.46 reflects the genuine difficulty of the benchmark (blog-style queries vs PubMed abstracts). Higher scores would be expected on a corpus with direct question-answer alignment.
- **Small generation eval set** — 24 questions (20 standard + 4 adversarial) is sufficient for a prototype but not for statistical significance.
- **Single corpus domain** — evaluated on nutrition/health literature only. Performance on other medical domains (clinical notes, radiology, genomics) is untested.

---

## Future Work

- **Graph RAG** — extract medical entity-relationship graph (drugs, diseases, genes) using scispaCy, combine graph traversal with hybrid retrieval for richer answers on complex multi-hop medical questions
- **ColBERT reranking** — replace cross-encoder with ColBERT for faster reranking at scale
- **Query expansion** — expand medical queries with synonyms and related terms before retrieval
