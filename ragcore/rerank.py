"""Second-stage reranking.

Bi-encoder retrieval (the embedding search) is fast but coarse: it scores the
query and a chunk independently. A cross-encoder reads the query and chunk
TOGETHER and scores their actual relevance, which is far more accurate - too
slow to run over the whole corpus, but perfect for re-ordering the top ~20
candidates the vector store already narrowed down. This two-stage
retrieve-then-rerank pattern is a major lever for retrieval quality in practical
RAG, so it gets its own stage here.
"""
from __future__ import annotations

from typing import Protocol

from .ingest import Chunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk],
               top_k: int) -> list[tuple[Chunk, float]]: ...


class IdentityReranker:
    """Keeps the retrieval order. Used in tests and as a no-dep fallback."""

    def rerank(self, query: str, chunks: list[Chunk],
               top_k: int) -> list[tuple[Chunk, float]]:
        return [(c, 0.0) for c in chunks[:top_k]]


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder (ms-marco-MiniLM), loaded lazily."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _lazy(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[Chunk],
               top_k: int) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        model = self._lazy()
        scores = model.predict([(query, c.text) for c in chunks])
        ranked = sorted(zip(chunks, (float(s) for s in scores)),
                        key=lambda p: p[1], reverse=True)
        return ranked[:top_k]


def default_reranker():
    try:
        import sentence_transformers  # noqa: F401
        return CrossEncoderReranker()
    except Exception:
        return IdentityReranker()
