"""Turn text into vectors.

Two implementations behind one interface:

  * STEmbedder - sentence-transformers (all-MiniLM-L6-v2). Small, fast on CPU,
    free, and the industry-standard baseline. This is what the live app uses.
  * HashingEmbedder - a deterministic, dependency-free bag-of-hashed-words
    vector. No torch, no model download. Unit tests use it so the test suite
    runs in milliseconds in CI, and it doubles as an offline fallback if the
    model can't be loaded.

Both return L2-normalised rows, so a dot product IS cosine similarity and the
vector store can use plain inner-product search.
"""
from __future__ import annotations

import re
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic hashed bag-of-words. Keyword-level signal, zero deps."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in _WORD.findall(text.lower()):
                # Stable hash across processes (Python's hash() is salted).
                h = 2166136261
                for ch in tok:
                    h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
                out[i, h % self.dim] += 1.0
        return _l2_normalise(out)


class STEmbedder:
    """sentence-transformers wrapper, model loaded lazily on first use."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dim = 384  # all-MiniLM-L6-v2

    def _lazy(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._lazy()
        vecs = model.encode(texts, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype("float32")


def default_embedder():
    """STEmbedder if sentence-transformers is importable, else the hash fallback."""
    try:
        import sentence_transformers  # noqa: F401
        return STEmbedder()
    except Exception:
        return HashingEmbedder()
