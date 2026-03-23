from __future__ import annotations

import numpy as np

from core.graph import KuzuStore
from core.embeddings.client import EmbeddingsClient
from core.embeddings.index import EmbeddingIndex


def build_index(
    store: KuzuStore,
    client: EmbeddingsClient,
    *,
    show_progress: bool = False,
) -> EmbeddingIndex:
    """Read all chunks from the store, embed them, and return an EmbeddingIndex."""
    chunks = store.read_chunks()
    if not chunks:
        return EmbeddingIndex.build([], np.empty((0, 1), dtype=np.float32))

    chunk_ids = [c.id for c in chunks]
    texts = [c.content for c in chunks]

    if show_progress:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
        cfg = client._cfg
        batch_size = cfg.batch_size
        all_vectors: list[np.ndarray] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("Building embeddings", total=len(texts))
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = client.embed(batch)
                all_vectors.append(vecs)
                progress.advance(task, len(batch))
        vectors = np.vstack(all_vectors)
    else:
        vectors = client.embed(texts)

    return EmbeddingIndex.build(chunk_ids, vectors)
