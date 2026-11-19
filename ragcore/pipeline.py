"""The end-to-end RAG pipeline: retrieve -> rerank -> generate.

One object wires the four stages together so the app, the CLI, and the eval
harness all exercise the EXACT same code path - the only way an eval number can
be trusted to describe the thing users actually hit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .generate import Answer, generate
from .ingest import Chunk, Document, chunk_documents
from .store import VectorStore


@dataclass
class RagResult:
    query: str
    answer: Answer
    retrieved: list[tuple[Chunk, float]] = field(default_factory=list)
    reranked: list[tuple[Chunk, float]] = field(default_factory=list)

    @property
    def context(self) -> list[Chunk]:
        return [c for c, _ in self.reranked]


class RagPipeline:
    def __init__(self, embedder, store: VectorStore, reranker, brain,
                 retrieve_k: int = 20, context_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.brain = brain
        self.retrieve_k = retrieve_k
        self.context_k = context_k

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        qvec = self.embedder.embed([query])[0]
        return self.store.search(qvec, k=self.retrieve_k)

    def answer(self, query: str) -> RagResult:
        retrieved = self.retrieve(query)
        chunks = [c for c, _ in retrieved]
        reranked = self.reranker.rerank(query, chunks, top_k=self.context_k)
        context = [c for c, _ in reranked]
        ans = generate(query, context, self.brain)
        return RagResult(query=query, answer=ans, retrieved=retrieved,
                         reranked=reranked)


def build_store(documents: list[Document], embedder,
                target_words: int = 180, overlap_words: int = 40) -> VectorStore:
    """Chunk documents, embed them, and build a ready-to-search vector store."""
    chunks = chunk_documents(documents, target_words, overlap_words)
    if not chunks:
        raise ValueError("no chunks produced from documents")
    embeddings = embedder.embed([c.text for c in chunks])
    return VectorStore(chunks, embeddings)
