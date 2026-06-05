from __future__ import annotations

import numpy as np
import kuzu

from core.embeddings.client import EmbeddingsClient
from core.embeddings.index import EmbeddingIndex
from core.graph.retrieval import _match_anchor_entities, find_entity_chunks, find_trust_paths, retrieve_context
from core.scoring.base import ScoredRetrievalResult, TripleScorer


class Retriever:
    def __init__(
        self,
        index: EmbeddingIndex,
        embed_client: EmbeddingsClient,
        conn: kuzu.Connection,
        scorer: TripleScorer,
        top_k: int = 8,
        hop_depth: int = 1,
        max_hop2_entities: int = 20,
        path_retrieval: bool = False,
        max_trust_paths: int = 50,
    ) -> None:
        self._index = index
        self._embed_client = embed_client
        self._conn = conn
        self._scorer = scorer
        self._top_k = top_k
        self._hop_depth = hop_depth
        self._max_hop2_entities = max_hop2_entities
        self._path_retrieval = path_retrieval
        self._max_trust_paths = max_trust_paths

    def retrieve(self, query: str) -> ScoredRetrievalResult:
        """Embed query → top-k chunks (+ entity augmentation) → graph context → scored trust bundles."""
        query_vec: np.ndarray = self._embed_client.embed_one(query)
        hits = self._index.query(query_vec, top_k=self._top_k)
        embedding_ids = [chunk_id for chunk_id, _score in hits]

        # Augment with chunks whose triples involve entities named in the query
        entity_ids = find_entity_chunks(
            self._conn, query,
            hop_depth=self._hop_depth,
            max_hop2_entities=self._max_hop2_entities,
        )
        # Union, preserving embedding order first (they stay highest-ranked)
        seen: set[str] = set(embedding_ids)
        augmented_ids = embedding_ids + [c for c in entity_ids if c not in seen]

        # Trust-Propagated Path Retrieval (TPPR): augment with chunks from
        # highest-trust evidence chains rooted at question entities.
        if self._path_retrieval:
            anchor_entities = _match_anchor_entities(self._conn, query)
            paths = find_trust_paths(self._conn, anchor_entities, max_paths=self._max_trust_paths)
            seen_aug: set[str] = set(augmented_ids)
            augmented_ids = augmented_ids + [
                cid for p in paths for cid in p.chunk_ids if cid not in seen_aug
            ]

        result = retrieve_context(self._conn, augmented_ids)

        chunk_map = {c.chunk_id: c for c in result.chunks}
        trust_bundles = self._scorer.score(
            query=query,
            query_vec=query_vec,
            triples=result.triples,
            chunk_map=chunk_map,
        )

        return ScoredRetrievalResult(chunks=result.chunks, trust_bundles=trust_bundles, embedding_chunk_ids=embedding_ids).rerank()

    def update_index(self, new_index: EmbeddingIndex) -> None:
        """Thread-safe swap of the embedding index."""
        self._index.update_index(new_index)
