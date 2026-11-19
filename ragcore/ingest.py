"""Load source documents and split them into retrieval chunks.

Chunking is where most RAG systems lose quality: too big and the reranker/LLM
drown in irrelevant text, too small and a single fact gets split across chunks
and is never retrieved whole. We chunk on paragraph/sentence boundaries with a
small overlap so a fact that straddles a boundary still lands intact in at least
one chunk. Every chunk keeps a back-pointer to its document and position so an
answer can cite exactly where a claim came from.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Document:
    """A source document before chunking."""
    doc_id: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a document, with enough provenance to cite it."""
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int          # 0-based index of this chunk within its document
    meta: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Short human-readable source tag, e.g. 'Attention Is All You Need #3'."""
        return "%s #%d" % (self.title or self.doc_id, self.position)


_WORD = re.compile(r"\S+")


def _split_paragraphs(text: str) -> list[str]:
    # Collapse blank-line runs into paragraph breaks, then split.
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, target_words: int = 180, overlap_words: int = 40) -> list[str]:
    """Greedy paragraph packing with word overlap.

    Paragraphs are packed together until adding the next would exceed
    ``target_words``; then a new chunk starts, seeded with the last
    ``overlap_words`` of the previous chunk so context carries across the seam.
    A single paragraph longer than the target is hard-split on word count.
    """
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    overlap_words = max(0, min(overlap_words, target_words - 1))

    # First, break any oversized paragraph into target-sized word windows.
    paras: list[str] = []
    for para in _split_paragraphs(text):
        words = _WORD.findall(para)
        if len(words) <= target_words:
            paras.append(para)
            continue
        for i in range(0, len(words), target_words):
            paras.append(" ".join(words[i:i + target_words]))

    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for para in paras:
        n = len(_WORD.findall(para))
        if cur and cur_n + n > target_words:
            chunks.append("\n\n".join(cur))
            tail = _WORD.findall(chunks[-1])[-overlap_words:] if overlap_words else []
            cur = [" ".join(tail)] if tail else []
            cur_n = len(tail)
        cur.append(para)
        cur_n += n
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c.strip()]


def chunk_documents(docs: Iterable[Document], target_words: int = 180,
                    overlap_words: int = 40) -> list[Chunk]:
    """Turn documents into a flat list of citable chunks."""
    out: list[Chunk] = []
    for doc in docs:
        for pos, piece in enumerate(chunk_text(doc.text, target_words, overlap_words)):
            out.append(Chunk(
                chunk_id="%s::%d" % (doc.doc_id, pos),
                doc_id=doc.doc_id,
                title=doc.title,
                text=piece,
                position=pos,
                meta=dict(doc.meta),
            ))
    return out


# --- loaders ---------------------------------------------------------------

def load_text_file(path: str) -> Document:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    base = os.path.splitext(os.path.basename(path))[0]
    return Document(doc_id=base, title=base.replace("_", " "), text=text,
                    meta={"source": path})


def load_pdf_file(path: str) -> Document:
    """Extract text from a PDF. pypdf is pure-Python and installs cleanly on
    Streamlit Cloud; we only ever READ PDFs here, never generate them."""
    from pypdf import PdfReader  # lazy import so tests don't need pypdf
    reader = PdfReader(path)
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(pages)
    base = os.path.splitext(os.path.basename(path))[0]
    title = (reader.metadata.title if reader.metadata and reader.metadata.title
             else base.replace("_", " "))
    return Document(doc_id=base, title=title, text=text, meta={"source": path})


def load_path(path: str) -> Document:
    """Load a single .txt/.md/.pdf file into a Document."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf_file(path)
    return load_text_file(path)
