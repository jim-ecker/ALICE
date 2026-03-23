from __future__ import annotations

import numpy as np

from core.embeddings.client import EmbeddingsClient
from core.graph.retrieval import RetrievedTriple


class EmbeddingRelevanceScorer:
    """Scores triples by cosine similarity between query vector and triple embedding.

    The triple is embedded as "{subject} {relation} {object_}" — a natural language
    rendering that places it in the same semantic space as the query.
    """

    def __init__(self, embed_client: EmbeddingsClient) -> None:
        self._client = embed_client

    def score_batch(
        self, query_vec: np.ndarray, triples: list[RetrievedTriple]
    ) -> list[float]:
        """Return a relevance score in [0, 1] for each triple (same order as input)."""
        if not triples:
            return []

        texts = [f"{t.subject} {t.relation} {t.object_}" for t in triples]
        vecs = self._client.embed(texts)  # (N, D) float32, unnormalised

        # L2-normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs_norm = vecs / norms

        q_norm = query_vec / max(float(np.linalg.norm(query_vec)), 1e-9)
        raw = vecs_norm @ q_norm.astype(np.float32)
        scores = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)

        # Cosine can be negative; clip to [0, 1]
        return [float(max(0.0, float(s))) for s in scores]
