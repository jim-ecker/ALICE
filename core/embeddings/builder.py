from __future__ import annotations

import numpy as np

from core.graph import KuzuStore
from core.embeddings.client import EmbeddingsClient
from core.embeddings.index import EmbeddingIndex


def update_index(
    store: KuzuStore,
    client: EmbeddingsClient,
    existing: EmbeddingIndex | None = None,
    *,
    show_progress: bool = False,
) -> tuple[EmbeddingIndex, dict]:
    """Incrementally update *existing* with any chunks in *store* not yet indexed.

    Returns ``(index, stats)`` where stats has keys:
      - ``total``          — total chunks in the graph
      - ``already_indexed``— chunks already present in *existing*
      - ``new_embedded``   — chunks embedded this call

    Cold-start (existing is None): embeds all chunks, equivalent to build_index().
    Re-ingest of same corpus: new_embedded == 0, zero embed calls made.
    """
    all_chunks = store.read_chunks()
    total = len(all_chunks)

    if existing is None or len(existing) == 0:
        already_indexed = 0
        new_chunks = all_chunks
    else:
        indexed = existing.chunk_id_set
        new_chunks = [c for c in all_chunks if c.id not in indexed]
        already_indexed = total - len(new_chunks)

    stats = {"total": total, "already_indexed": already_indexed, "new_embedded": len(new_chunks)}

    if not new_chunks:
        empty = EmbeddingIndex.build([], np.empty((0, 1), dtype=np.float32))
        return (existing if existing is not None else empty), stats

    chunk_ids = [c.id for c in new_chunks]
    texts = [c.content for c in new_chunks]

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
            task = progress.add_task("Embedding new chunks", total=len(texts))
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = client.embed(batch)
                all_vectors.append(vecs)
                progress.advance(task, len(batch))
        vectors = np.vstack(all_vectors)
    else:
        vectors = client.embed(texts)

    mini = EmbeddingIndex.build(chunk_ids, vectors)

    if existing is None or len(existing) == 0:
        return mini, stats
    return existing.merge(mini), stats


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
