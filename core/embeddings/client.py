from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingsConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "token"
    model: str = "nomic-embed-text"
    batch_size: int = 64
    max_chars: int = 20000


class EmbeddingsClient:
    def __init__(self, cfg: EmbeddingsConfig) -> None:
        self._cfg = cfg

    def _client(self):
        from openai import OpenAI
        return OpenAI(base_url=self._cfg.base_url, api_key=self._cfg.api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning shape (N, D) float32 array."""
        client = self._client()
        all_vectors: list[list[float]] = []
        batch_size = self._cfg.batch_size
        max_chars = self._cfg.max_chars
        truncated = [t[:max_chars] for t in texts]
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i : i + batch_size]
            response = client.embeddings.create(model=self._cfg.model, input=batch)
            items = sorted(response.data, key=lambda x: x.index)
            all_vectors.extend(item.embedding for item in items)
        return np.array(all_vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text, returning shape (D,) float32 array."""
        return self.embed([text])[0]
