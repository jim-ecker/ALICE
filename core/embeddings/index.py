from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np


class EmbeddingIndex:
    def __init__(self, chunk_ids: list[str], matrix: np.ndarray) -> None:
        self._chunk_ids = chunk_ids
        self._matrix = matrix  # (N, D) float32, L2-normalised
        self._lock = threading.Lock()

    @classmethod
    def build(cls, chunk_ids: list[str], vectors: np.ndarray) -> "EmbeddingIndex":
        """L2-normalise rows and build the index."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = (vectors / norms).astype(np.float32)
        return cls(chunk_ids, normalised)

    @classmethod
    def load(cls, path: Path) -> "EmbeddingIndex":
        data = np.load(path, allow_pickle=True)
        chunk_ids = list(data["chunk_ids"])
        matrix = data["matrix"].astype(np.float32)
        return cls(chunk_ids, matrix)

    def save(self, path: Path) -> None:
        """Atomically write to a temp file then replace."""
        path = Path(path)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, chunk_ids=np.array(self._chunk_ids, dtype=object), matrix=self._matrix)
        os.replace(tmp, path)

    def query(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score) pairs sorted by descending cosine similarity."""
        with self._lock:
            if len(self._chunk_ids) == 0:
                return []
            norm = np.linalg.norm(vector)
            if norm == 0:
                return []
            query_norm = (vector / norm).astype(np.float32)
            scores = self._matrix @ query_norm
            n = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -n)[-n:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [(self._chunk_ids[i], float(scores[i])) for i in top_indices]

    def score_chunks(self, vector: np.ndarray, chunk_ids: list[str]) -> dict[str, float]:
        """Return cosine similarity to `vector` for each requested chunk_id."""
        with self._lock:
            if not chunk_ids or len(self._chunk_ids) == 0:
                return {cid: 0.0 for cid in chunk_ids}
            norm = np.linalg.norm(vector)
            if norm == 0:
                return {cid: 0.0 for cid in chunk_ids}
            query_norm = (vector / norm).astype(np.float32)
            id_to_idx = {cid: i for i, cid in enumerate(self._chunk_ids)}
            return {
                cid: float(self._matrix[id_to_idx[cid]] @ query_norm)
                if (cid in id_to_idx) else 0.0
                for cid in chunk_ids
            }

    @property
    def chunk_id_set(self) -> set[str]:
        """Set of all chunk IDs currently in the index."""
        return set(self._chunk_ids)

    def merge(self, other: "EmbeddingIndex") -> "EmbeddingIndex":
        """Return a new EmbeddingIndex with chunks from *other* appended.

        Chunk IDs already present in *self* are skipped — no duplicates are
        introduced. Both matrices must already be L2-normalised (as guaranteed
        by EmbeddingIndex.build()), so rows can be concatenated directly.
        """
        existing_ids = set(self._chunk_ids)
        new_ids = [cid for cid in other._chunk_ids if cid not in existing_ids]
        if not new_ids:
            return self
        new_id_set = set(new_ids)
        new_rows = other._matrix[
            [i for i, cid in enumerate(other._chunk_ids) if cid in new_id_set]
        ]
        combined_ids = self._chunk_ids + new_ids
        combined_matrix = np.vstack([self._matrix, new_rows])
        return EmbeddingIndex(combined_ids, combined_matrix)

    def update_index(self, new: "EmbeddingIndex") -> None:
        """Thread-safe swap of the underlying data."""
        with self._lock:
            self._chunk_ids = new._chunk_ids
            self._matrix = new._matrix

    def __len__(self) -> int:
        return len(self._chunk_ids)
