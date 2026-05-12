"""Step-by-step debug of evaluate.py"""
import traceback

print("Step 1: imports...")
from medrag.indexing.bm25_index import BM25Index
from medrag.indexing.vector_index import VectorIndex
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import CrossEncoderReranker
from medrag.generation.pipeline import RAGPipeline
from medrag.verification.faithfulness import FaithfulnessVerifier
from medrag.evaluation.bioasq import BioASQEvaluator
from medrag.evaluation.metrics import aggregate_retrieval_metrics
print("  OK")

print("Step 2: load indexes...")
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
print("  OK")

print("Step 3: load BioASQ (10 samples)...")
evaluator = BioASQEvaluator(max_samples=10)
samples = evaluator.load()
print(f"  Loaded {len(samples)} samples")
print(f"  Sample Q: {samples[0].question[:80]}")

print("Step 4: single retrieval eval...")
try:
    q = samples[0].question
    bm25_r = bm25.search(q, top_k=5)
    hybrid_r = hybrid.retrieve(q, top_k=20)
    reranked_r = reranker.rerank(q, hybrid_r, top_k=5)
    print(f"  BM25 top: {bm25_r[0]['title'][:60]}")
    print(f"  Reranked top: {reranked_r[0]['title'][:60]}")
except Exception:
    traceback.print_exc()

print("Step 5: run_retrieval_eval on 5 samples...")
try:
    evaluator2 = BioASQEvaluator(max_samples=5)
    evaluator2.load()
    metrics = evaluator2.run_retrieval_eval(
        retrieve_fn=lambda q, top_k: bm25.search(q, top_k=top_k),
        top_k_values=[5, 10],
    )
    print(f"  BM25 metrics: {metrics}")
except Exception:
    traceback.print_exc()

print("Step 6: generation + verification on 1 sample...")
try:
    chunks = reranker.rerank(samples[0].question, hybrid.retrieve(samples[0].question, top_k=20), top_k=5)
    gen = generator.generate(samples[0].question, chunks)
    ver = verifier.verify(gen["answer"], chunks)
    print(f"  Answer: {gen['answer'][:80]}")
    print(f"  Faithfulness: {ver.faithfulness_score:.3f} | Risk: {ver.risk.value}")
except Exception:
    traceback.print_exc()

print("\nAll steps done.")
