from setuptools import setup, find_packages

setup(
    name="medrag",
    version="0.1.0",
    description="Clinical QA retrieval system with cross-encoder reranking and NLI faithfulness verification",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "rank-bm25>=0.2.2",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.2",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
)
