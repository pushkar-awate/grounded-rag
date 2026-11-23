"""grounded-rag: a retrieval-augmented QA pipeline that cites its sources and
refuses to answer when the documents don't support it."""
from .ingest import Chunk, Document, chunk_documents, chunk_text, load_path
from .embed import Embedder, HashingEmbedder, STEmbedder, default_embedder
from .store import VectorStore
from .rerank import (CrossEncoderReranker, IdentityReranker, Reranker,
                     default_reranker)
from .brain import (ExtractiveBrain, GroqBrain, MockBrain, NaiveBrain,
                    default_brain, has_groq_key)
from .generate import ABSTAIN_PHRASE, Answer, generate
from .pipeline import RagPipeline, RagResult, build_store

__all__ = [
    "Chunk", "Document", "chunk_documents", "chunk_text", "load_path",
    "Embedder", "HashingEmbedder", "STEmbedder", "default_embedder",
    "VectorStore",
    "CrossEncoderReranker", "IdentityReranker", "Reranker", "default_reranker",
    "ExtractiveBrain", "GroqBrain", "MockBrain", "NaiveBrain", "default_brain",
    "has_groq_key",
    "ABSTAIN_PHRASE", "Answer", "generate",
    "RagPipeline", "RagResult", "build_store",
]
