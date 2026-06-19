from __future__ import annotations

import numpy as np

from core.embeddings.client import EmbeddingsClient


class PCAProjection:
    def __init__(self, components: np.ndarray, mean: np.ndarray) -> None:
        self.components = components.astype(np.float32)  # (d, emb)
        self.mean = mean.astype(np.float32)  # (emb,)

    @classmethod
    def fit(cls, entity_vectors: np.ndarray, d: int) -> "PCAProjection":
        x = np.asarray(entity_vectors, dtype=np.float32)
        mean = x.mean(axis=0)
        centered = x - mean
        n_features = centered.shape[1]
        k = min(d, n_features)
        # SVD of the centered data: rows of Vt are principal directions.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:k]
        if k < d:
            pad = np.zeros((d - k, n_features), dtype=np.float32)
            components = np.vstack([components, pad])
        return cls(components.astype(np.float32), mean)

    def project(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        return (self.components @ (v - self.mean)).astype(np.float32)


class EntityEmbeddingProvider:
    def __init__(
        self,
        names: list[str],
        matrix: np.ndarray,
        embed_client: EmbeddingsClient,
        projection: PCAProjection,
    ) -> None:
        self.names = names
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.embed_client = embed_client
        self.projection = projection
        self._idx = {name: i for i, name in enumerate(names)}

    def entity_vector(self, name: str) -> np.ndarray:
        i = self._idx.get(name)
        if i is None:
            raw = self.embed_client.embed_one(name)
            return self.projection.project(raw)
        return self.projection.project(self.matrix[i])

    def query_vector(self, query: str) -> np.ndarray:
        raw = self.embed_client.embed_one(query)
        return self.projection.project(raw)


def build_query_section(
    anchors: list[str],
    provider: EntityEmbeddingProvider,
    d: int,
) -> dict[str, np.ndarray]:
    """Phase 0 (Deviation D2): bare projected entity vector, no query modulation."""
    return {anchor: provider.entity_vector(anchor) for anchor in anchors}
