"""Quick smoke test for the full pipeline."""
from medrag.indexing.bm25_index import BM25Index
from medrag.indexing.vector_index import VectorIndex
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import CrossEncoderReranker
from medrag.generation.pipeline import RAGPipeline
from medrag.verification.faithfulness import FaithfulnessVerifier
from medrag.evaluation.metrics import ndcg_at_k, hallucination_rate

print("Loading indexes...")
bm25 = BM25Index()
bm25.load("data/index/bm25")
vector = VectorIndex(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    persist_dir="data/index/chroma",
)
hybrid = HybridRetriever(bm25, vector)
reranker = CrossEncoderReranker()
generator = RAGPipeline()
verifier = FaithfulnessVerifier()

query = "What is the effect of metformin on blood glucose in type 2 diabetes?"
print(f"\nQuery: {query}\n")

print("--- BM25 only ---")
bm25_results = bm25.search(query, top_k=5)
for r in bm25_results[:3]:
    print(f"  [{r['bm25_score']:.2f}] {r['title'][:70]}")

print("\n--- Hybrid ---")
hybrid_results = hybrid.retrieve(query, top_k=20)
for r in hybrid_results[:3]:
    print(f"  [{r['hybrid_score']:.3f}] {r['title'][:70]}")

print("\n--- Reranked ---")
reranked = reranker.rerank(query, hybrid_results, top_k=5)
for r in reranked[:3]:
    print(f"  [{r['rerank_score']:.3f}] {r['title'][:70]}")

print("\n--- Generated answer ---")
generation = generator.generate(query, reranked)
print(f"  {generation['answer']}")

print("\n--- Faithfulness ---")
result = verifier.verify(generation["answer"], reranked)
print(f"  Score : {result.faithfulness_score:.3f}")
print(f"  Risk  : {result.risk.value}")
print(f"  Unsupported: {result.unsupported}")
