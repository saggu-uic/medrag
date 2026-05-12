from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass
class Article:
    pmc_id: str
    title: str
    abstract: str
    body: str
    journal: str
    publication_year: int | None
    author_count: int
    citation_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.abstract} {self.body}".strip()


class PMCParser:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def parse_file(self, xml_path: Path) -> Article | None:
        try:
            soup = BeautifulSoup(xml_path.read_text(encoding="utf-8"), "xml")

            pmc_id = xml_path.stem

            title_el = soup.find("article-title")
            title = title_el.get_text(separator=" ").strip() if title_el else ""

            abstract_el = soup.find("abstract")
            abstract = ""
            if abstract_el:
                paragraphs = abstract_el.find_all("p")
                abstract = " ".join(p.get_text(separator=" ") for p in paragraphs) if paragraphs else abstract_el.get_text()

            body_el = soup.find("body")
            body = ""
            if body_el:
                paragraphs = body_el.find_all("p")
                body = " ".join(p.get_text(separator=" ") for p in paragraphs) if paragraphs else body_el.get_text()

            journal_el = soup.find("journal-title")
            journal = journal_el.get_text().strip() if journal_el else ""

            pub_year = None
            pub_date = soup.find("pub-date")
            if pub_date:
                year_el = pub_date.find("year")
                if year_el:
                    try:
                        pub_year = int(year_el.get_text())
                    except ValueError:
                        pass

            author_count = len(soup.find_all("contrib", {"contrib-type": "author"}))

            ref_list = soup.find("ref-list")
            citation_count = len(ref_list.find_all("ref")) if ref_list else 0

            return Article(
                pmc_id=pmc_id,
                title=title,
                abstract=abstract,
                body=body,
                journal=journal,
                publication_year=pub_year,
                author_count=author_count,
                citation_count=citation_count,
            )

        except Exception as e:
            print(f"[PMCParser] Failed to parse {xml_path.name}: {e}")
            return None

    def iter_articles(self) -> Iterator[Article]:
        xml_files = list(self.data_dir.rglob("*.xml"))
        for xml_path in tqdm(xml_files, desc="Parsing PMC articles"):
            article = self.parse_file(xml_path)
            if article:
                yield article

    def parse_and_save(self, output_path: str | Path) -> int:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for article in self.iter_articles():
                f.write(json.dumps(article.to_dict()) + "\n")
                count += 1

        print(f"[PMCParser] Saved {count} articles to {output_path}")
        return count

    @staticmethod
    def load_jsonl(path: str | Path) -> list[Article]:
        articles = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    articles.append(Article(**data))
        return articles
