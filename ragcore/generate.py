"""Grounded answer generation.

The whole point of this project: the model answers ONLY from retrieved context,
cites every claim with a [n] marker tied to a specific chunk, and explicitly
abstains when the context doesn't contain the answer. That last part - refusing
to answer rather than hallucinating - is what separates a trustworthy RAG system
from a confident liar, and it's measured directly in the eval harness.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Chunk

ABSTAIN_PHRASE = ("I don't have enough information in the provided documents "
                  "to answer that.")

GROUNDED_SYSTEM = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the numbered sources provided. Rules:\n"
    "1. Every factual claim must be followed by a citation like [1] or [2] "
    "naming the source it came from.\n"
    "2. Do not use any knowledge outside the provided sources.\n"
    "3. If the sources do not contain enough information to answer, reply with "
    "EXACTLY this sentence and nothing else: \"" + ABSTAIN_PHRASE + "\"\n"
    "4. Be concise and factual. Do not speculate."
)

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    abstained: bool
    cited_chunks: list[Chunk] = field(default_factory=list)
    context_chunks: list[Chunk] = field(default_factory=list)


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    lines = ["Sources:", ""]
    for i, ch in enumerate(chunks, start=1):
        lines.append("[%d] (%s)\n%s" % (i, ch.citation(), ch.text.strip()))
        lines.append("")
    lines.append("Question: %s" % query.strip())
    lines.append("")
    lines.append("Answer (cite sources with [n]):")
    return "\n".join(lines)


def _is_abstention(text: str) -> bool:
    core = "don't have enough information"
    return core in text.lower().replace("’", "'")


def parse_citations(text: str, chunks: list[Chunk]) -> list[Chunk]:
    """Map [n] markers in the answer back to the chunks they reference."""
    seen: list[Chunk] = []
    for m in _CITE.findall(text):
        idx = int(m) - 1
        if 0 <= idx < len(chunks) and chunks[idx] not in seen:
            seen.append(chunks[idx])
    return seen


def generate(query: str, chunks: list[Chunk], brain) -> Answer:
    """Produce a grounded Answer from the question and its context chunks."""
    if not chunks:
        return Answer(text=ABSTAIN_PHRASE, abstained=True, context_chunks=[])
    prompt = build_prompt(query, chunks)
    raw = brain.complete(prompt, system=GROUNDED_SYSTEM, temperature=0.0).strip()
    if _is_abstention(raw):
        return Answer(text=ABSTAIN_PHRASE, abstained=True, context_chunks=chunks)
    cited = parse_citations(raw, chunks)
    return Answer(text=raw, abstained=False, cited_chunks=cited,
                  context_chunks=chunks)
