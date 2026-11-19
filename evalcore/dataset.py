"""The gold set the RAG system is graded against.

Each case says what SHOULD happen: which document holds the answer (so we can
score retrieval), and - crucially - whether the corpus can answer it at all.
The "unanswerable" cases are the sharp edge of the suite: a RAG system that
scores well on answerable questions but invents answers to unanswerable ones is
worse than useless, and only a dataset that includes both can catch it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    question: str
    answerable: bool                       # can the corpus answer this?
    relevant_doc_ids: list[str] = field(default_factory=list)
    reference: str = ""                    # optional reference answer / key fact
    must_include: list[str] = field(default_factory=list)  # terms a good answer cites


def load_cases(path: str) -> list[EvalCase]:
    import json
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [EvalCase(**c) for c in raw]
