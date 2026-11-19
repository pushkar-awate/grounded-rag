"""A tiny vector index over chunks.

FAISS when it's installed (fast, what the app ships with); a NumPy brute-force
fallback otherwise, so the library is importable and testable anywhere. Vectors
are pre-normalised by the embedder, so inner product == cosine similarity and we
can use the simplest possible index (IndexFlatIP) with no training step.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np

from .ingest import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        self.chunks = chunks
        self.embeddings = embeddings.astype("float32")
        self._index = self._build_faiss(self.embeddings)

    @staticmethod
    def _build_faiss(embeddings: np.ndarray):
        try:
            import faiss
        except Exception:
            return None
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def search(self, query_vec: np.ndarray, k: int = 8) -> list[tuple[Chunk, float]]:
        q = query_vec.astype("float32").reshape(1, -1)
        k = min(k, len(self.chunks))
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
        else:
            sims = (self.embeddings @ q[0])
            order = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in order]
        return [(self.chunks[i], float(s)) for i, s in pairs if i >= 0]

    # --- persistence -------------------------------------------------------

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(dir_path, "chunks.jsonl"), "w", encoding="utf-8") as fh:
            for ch in self.chunks:
                fh.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        embeddings = np.load(os.path.join(dir_path, "embeddings.npy"))
        chunks: list[Chunk] = []
        with open(os.path.join(dir_path, "chunks.jsonl"), "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))
        return cls(chunks, embeddings)

    @classmethod
    def exists(cls, dir_path: str) -> bool:
        return (os.path.exists(os.path.join(dir_path, "embeddings.npy")) and
                os.path.exists(os.path.join(dir_path, "chunks.jsonl")))
