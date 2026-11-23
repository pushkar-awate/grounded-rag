"""The end-to-end RAG pipeline: retrieve -> rerank -> generate.

One object wires the stages together so the app, the CLI, and the eval harness
all exercise the EXACT same code path - the only way an eval number can be
trusted to describe the thing users actually hit.

Governance is opt-in and composes on top of the same path. Give the pipeline a
Principal and retrieval is access-scoped: passages above the caller's clearance
are filtered out before rerank and generation, so if the only passages that
would answer are off-limits, the context comes back empty and the system
abstains instead of answering from data the caller cannot see. Give it a PII
flag and retrieved context is redacted before the model reads it. Give it an
AuditLog and every answer-or-abstain decision is recorded. With none of these
set the pipeline behaves exactly as the plain RAG system it started as.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .generate import Answer, generate
from .ingest import Chunk, Document, chunk_documents
from .store import VectorStore

from govcore.access import Principal, allow_predicate, chunk_classification
from govcore.audit import AuditLog
from govcore.pii import redact_chunks

# Human-readable gloss for each machine abstain_reason.
_ABSTAIN_GLOSS = {
    "no_context": "No passages were retrieved for this question.",
    "insufficient_support": ("The retrieved passages did not contain enough to "
                             "answer, so the system declined to guess."),
    "access_scoped": ("Every passage that could answer this sits above the "
                      "caller's clearance and was withheld, so the system "
                      "abstained instead of answering from data the caller "
                      "cannot see."),
}


@dataclass
class RagResult:
    query: str
    answer: Answer
    retrieved: list[tuple[Chunk, float]] = field(default_factory=list)
    reranked: list[tuple[Chunk, float]] = field(default_factory=list)
    access_withheld: int = 0          # top passages hidden on access grounds
    pii_redactions: int = 0           # identifiers scrubbed from context
    principal: str = ""               # who asked (role label)
    decision_id: str = ""             # audit record id, if logged

    @property
    def context(self) -> list[Chunk]:
        return [c for c, _ in self.reranked]

    @property
    def top_score(self) -> Optional[float]:
        return self.reranked[0][1] if self.reranked else None

    @property
    def abstain_explanation(self) -> str:
        """A sentence a reviewer can read, built from the machine reason and the
        best supporting score."""
        if not self.answer.abstained:
            return ""
        base = _ABSTAIN_GLOSS.get(self.answer.abstain_reason,
                                  "The system declined to answer.")
        if self.answer.abstain_reason == "insufficient_support" and self.reranked:
            base += " Best supporting score was %.3f." % self.reranked[0][1]
        if self.access_withheld:
            base += " %d passage(s) were withheld on access grounds." % self.access_withheld
        return base


class RagPipeline:
    def __init__(self, embedder, store: VectorStore, reranker, brain,
                 retrieve_k: int = 20, context_k: int = 5,
                 principal: Optional[Principal] = None, pii: bool = False,
                 pii_kinds=None, audit: Optional[AuditLog] = None,
                 corpus_sig: str = ""):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.brain = brain
        self.retrieve_k = retrieve_k
        self.context_k = context_k
        self.principal = principal
        self.pii = pii
        self.pii_kinds = pii_kinds
        self.audit = audit
        self.corpus_sig = corpus_sig

    def _embed(self, query: str):
        return self.embedder.embed([query])[0]

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        qvec = self._embed(query)
        if self.principal is None:
            return self.store.search(qvec, k=self.retrieve_k)
        return self.store.search(qvec, k=self.retrieve_k,
                                 allow=allow_predicate(self.principal))

    def _model_name(self) -> str:
        return getattr(self.brain, "model", type(self.brain).__name__)

    def answer(self, query: str) -> RagResult:
        qvec = self._embed(query)
        withheld = 0
        if self.principal is None:
            retrieved = self.store.search(qvec, k=self.retrieve_k)
        else:
            allow = allow_predicate(self.principal)
            # What a caller WOULD see ungoverned, to report what access removed.
            raw = self.store.search(qvec, k=self.retrieve_k)
            withheld = sum(1 for c, _ in raw if not allow(c))
            retrieved = self.store.search(qvec, k=self.retrieve_k, allow=allow)

        chunks = [c for c, _ in retrieved]
        reranked = self.reranker.rerank(query, chunks, top_k=self.context_k)
        context = [c for c, _ in reranked]

        pii_redactions = 0
        if self.pii and context:
            context, pii_redactions = redact_chunks(context, self.pii_kinds)

        ans = generate(query, context, self.brain)

        # Attribute the abstention to access control precisely: if the caller
        # abstained but the SAME system over the unfiltered context would have
        # answered, then access scoping is the reason it refused. A genuinely
        # unanswerable question abstains either way and keeps its own reason.
        if self.principal is not None and ans.abstained and withheld > 0:
            raw_chunks = [c for c, _ in raw]
            shadow_ctx = [c for c, _ in self.reranker.rerank(
                query, raw_chunks, top_k=self.context_k)]
            if not generate(query, shadow_ctx, self.brain).abstained:
                ans.abstain_reason = "access_scoped"

        principal_label = self.principal.label if self.principal else ""
        result = RagResult(query=query, answer=ans, retrieved=retrieved,
                           reranked=reranked, access_withheld=withheld,
                           pii_redactions=pii_redactions,
                           principal=principal_label)

        if self.audit is not None:
            entry = self.audit.record(
                principal=self.principal.role if self.principal else "ungoverned",
                clearance=(self.principal.clearance.label
                           if self.principal else "n/a"),
                question=query,
                decision="abstain" if ans.abstained else "answer",
                abstain_reason=ans.abstain_reason,
                retrieved=[{"chunk_id": c.chunk_id,
                            "classification": int(chunk_classification(c)),
                            "score": round(float(s), 4)} for c, s in reranked],
                cited=[c.chunk_id for c in ans.cited_chunks],
                access_withheld=withheld,
                pii_redactions=pii_redactions,
                model=self._model_name(),
                corpus=self.corpus_sig,
            )
            result.decision_id = entry["decision_id"]

        return result


def build_store(documents: list[Document], embedder,
                target_words: int = 180, overlap_words: int = 40) -> VectorStore:
    """Chunk documents, embed them, and build a ready-to-search vector store."""
    chunks = chunk_documents(documents, target_words, overlap_words)
    if not chunks:
        raise ValueError("no chunks produced from documents")
    embeddings = embedder.embed([c.text for c in chunks])
    return VectorStore(chunks, embeddings)
