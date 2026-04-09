from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from core.graph.retrieval import CitationChunk, RetrievedTriple


@dataclass
class TrustBundle:
    """All trust signals for a single retrieved triple."""
    triple: RetrievedTriple
    ingest_certainty: float        # grounded ingest confidence stored on RELATES_TO.certainty_score
    relevance_score: float | None  # cosine(query_vec, embed(triple_text)) — query alignment
    grounding_score: float | None  # LLM entailment: does source chunk text support this triple?
    provenance_count: int          # distinct chunks that independently yielded same (s, r, o)
    composite_trust: float         # weighted combination of active signals


@dataclass
class ScoredRetrievalResult:
    """Retrieval result with per-triple trust scores."""
    chunks: list[CitationChunk]
    trust_bundles: list[TrustBundle]  # scored and optionally filtered
    embedding_chunk_ids: list[str] = field(default_factory=list)  # chunk_ids from embedding search, ordered by cosine similarity

    def rerank(self) -> "ScoredRetrievalResult":
        """Return a new ScoredRetrievalResult with chunks and triples sorted by composite_trust descending."""
        # Build rank lookup: embedding-retrieved chunks get their cosine-sim rank (lower = better);
        # entity-only chunks get INF so they lose tie-breaks against embedding chunks.
        emb_rank: dict[str, int] = {cid: i for i, cid in enumerate(self.embedding_chunk_ids)}
        INF = len(self.embedding_chunk_ids)

        # Sort primary by composite_trust descending, secondary by embedding rank ascending.
        # This ensures that when the same (s, r, o) triple exists in multiple chunks, the
        # instance from the most query-relevant chunk wins deduplication rather than whichever
        # happened to have a marginally higher ingest certainty.
        sorted_bundles = sorted(
            self.trust_bundles,
            key=lambda b: (-b.composite_trust, emb_rank.get(b.triple.chunk_id, INF)),
        )

        # Deduplicate: keep highest-scored bundle per unique (subject, relation, object)
        seen: set[tuple[str, str, str]] = set()
        deduped: list[TrustBundle] = []
        for b in sorted_bundles:
            key = (b.triple.subject, b.triple.relation, b.triple.object_)
            if key not in seen:
                seen.add(key)
                deduped.append(b)

        # Best composite_trust score per chunk (chunks with no triples score 0.0)
        chunk_scores: dict[str, float] = {}
        for b in deduped:
            cid = b.triple.chunk_id
            if cid not in chunk_scores:
                chunk_scores[cid] = b.composite_trust

        sorted_chunks = sorted(
            self.chunks,
            key=lambda c: chunk_scores.get(c.chunk_id, 0.0),
            reverse=True,
        )
        return ScoredRetrievalResult(chunks=sorted_chunks, trust_bundles=deduped, embedding_chunk_ids=self.embedding_chunk_ids)


class TripleScorer(ABC):
    """Computes TrustBundles for a batch of retrieved triples."""

    @abstractmethod
    def score(
        self,
        query: str,
        query_vec: np.ndarray,
        triples: list[RetrievedTriple],
        chunk_map: dict[str, CitationChunk],
    ) -> list[TrustBundle]: ...
