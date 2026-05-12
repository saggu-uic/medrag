"""
Parses PMC XML articles and saves them as JSONL.

Usage:
    python scripts/ingest.py --data-dir data/pmc_articles --output data/processed/articles.jsonl
"""

import argparse
from pathlib import Path

from medrag.ingestion.pmc_parser import PMCParser


def main():
    parser = argparse.ArgumentParser(description="Ingest PMC XML articles")
    parser.add_argument("--data-dir", required=True, help="Directory containing PMC XML files")
    parser.add_argument("--output", default="data/processed/articles.jsonl")
    args = parser.parse_args()

    pmc_parser = PMCParser(data_dir=args.data_dir)
    count = pmc_parser.parse_and_save(output_path=args.output)
    print(f"Ingested {count} articles → {args.output}")


if __name__ == "__main__":
    main()
