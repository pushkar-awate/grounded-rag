"""Where documents come from: the bundled ML corpus, an uploaded file, or a
live arXiv search. All three produce the same Document objects the pipeline
indexes, so the rest of the system never cares about the source.
"""
from __future__ import annotations

import glob
import json
import os

from ragcore.ingest import Document, load_text_file

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(HERE, "data", "corpus")
GOV_CORPUS = os.path.join(HERE, "data", "governance_corpus.json")
ACCESS_CASES = os.path.join(HERE, "data", "access_cases.json")


def load_bundled_corpus(corpus_dir: str = CORPUS_DIR) -> list[Document]:
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        docs.append(load_text_file(path))
    return docs


def load_governance_corpus(path: str = GOV_CORPUS) -> list[Document]:
    """The synthetic classified corpus used by the access-control demo. Each
    document carries a classification level and optional need-to-know
    compartment that access-scoped retrieval enforces."""
    from govcore.access import Classification
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    docs: list[Document] = []
    for d in raw["documents"]:
        docs.append(Document(
            doc_id=d["doc_id"], title=d["title"], text=d["text"],
            classification=int(Classification.parse(d.get("classification", 0))),
            compartment=d.get("compartment"),
            meta={"classification_label": d.get("classification", "PUBLIC")},
        ))
    return docs


def load_access_cases(path: str = ACCESS_CASES) -> list[dict]:
    """Question + the clearance a caller needs to answer it legitimately."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_arxiv(query: str, max_results: int = 8) -> list[Document]:
    """Pull recent arXiv abstracts for a query via the public Atom API (stdlib
    only). Each paper becomes a Document of title + abstract."""
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    url = ("http://export.arxiv.org/api/query?search_query=%s"
           "&start=0&max_results=%d&sortBy=relevance"
           % (urllib.parse.quote(query), max_results))
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    docs: list[Document] = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        aid = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        short = aid.rstrip("/").split("/")[-1] or title[:20]
        if title and summary:
            docs.append(Document(doc_id="arxiv_%s" % short, title=title,
                                 text=summary, meta={"source": aid}))
    return docs
