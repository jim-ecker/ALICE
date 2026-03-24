from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.graph.retrieval import CitationChunk, RetrievedTriple
from core.scoring.base import TrustBundle, TripleScorer


@dataclass
class ScoringConfig:
    # Signal weights — must be non-negative; they are normalised internally
    ingest_certainty_weight: float = 0.4
    relevance_weight: float = 0.4
    provenance_weight: float = 0.2
    grounding_weight: float = 0.0   # only active when grounding_enabled = True

    # Whether to run the expensive LLM entailment pass
    grounding_enabled: bool = False

    # Keep only the top-k triples by composite score per response (None = keep all)
    relevance_filter_top_k: int | None = None

    # ingest_certainty dampening — applied before weighting
    # Transform order: effective = min(raw, cap) ^ exponent
    # Defaults (1.0, 1.0) are identity — no dampening unless opted in via alice.toml
    ingest_certainty_cap: float = 1.0       # clip raw LLM confidence to this ceiling
    ingest_certainty_exponent: float = 1.0  # >1 curves the capped score downward


def _dampen_ingest(raw: float, cap: float, exponent: float) -> float:
    """Apply cap then power-curve to a raw ingest_certainty value."""
    return min(raw, cap) ** exponent


def _normalise_provenance(count: int) -> float:
    """Map provenance count to [0, 1]. 1 source → 0.0, 5+ sources → 1.0."""
    return min(count - 1, 4) / 4.0


class WeightedCompositeScorer(TripleScorer):
    """Combines ingest certainty, embedding relevance, provenance, and optional
    LLM grounding into a single composite trust score per triple.

    Sub-scorers are optional; pass None to disable. If both relevance_scorer and
    grounding_scorer are None the composite equals ingest_certainty.
    """

    def __init__(
        self,
        cfg: ScoringConfig,
        relevance_scorer=None,   # EmbeddingRelevanceScorer | None
        provenance_scorer=None,  # ProvenanceScorer | None
        grounding_scorer=None,   # GroundingScorer | None
    ) -> None:
        self._cfg = cfg
        self._rel = relevance_scorer
        self._prov = provenance_scorer
        self._gnd = grounding_scorer

    def score(
        self,
        query: str,
        query_vec: np.ndarray,
        triples: list[RetrievedTriple],
        chunk_map: dict[str, CitationChunk],
    ) -> list[TrustBundle]:
        if not triples:
            return []

        cfg = self._cfg

        def _clean(v: float | None, default: float = 0.0) -> float | None:
            """Replace NaN/inf with default; pass through None."""
            if v is None:
                return None
            return default if not math.isfinite(v) else v

        # ── Ingest certainty (always available) ──────────────────────────────
        # Guard against NaN stored in the DB, then apply optional dampening transform.
        ingest_scores = [
            _dampen_ingest(
                _clean(t.certainty_score, 0.0) or 0.0,
                cfg.ingest_certainty_cap,
                cfg.ingest_certainty_exponent,
            )
            for t in triples
        ]

        # ── Embedding relevance ───────────────────────────────────────────────
        if self._rel is not None:
            try:
                raw_rel = self._rel.score_batch(query_vec, triples)
                relevance_scores: list[float | None] = [_clean(s, 0.0) for s in raw_rel]
            except Exception:
                relevance_scores = [None] * len(triples)
        else:
            relevance_scores = [None] * len(triples)

        # ── Provenance ────────────────────────────────────────────────────────
        if self._prov is not None:
            try:
                prov_counts = self._prov.count_batch(triples)
            except Exception:
                prov_counts = [1] * len(triples)
        else:
            prov_counts = [1] * len(triples)

        # ── Grounding (expensive, opt-in) ─────────────────────────────────────
        if cfg.grounding_enabled and self._gnd is not None:
            try:
                raw_gnd = self._gnd.score_batch(triples, chunk_map)
                grounding_scores: list[float | None] = [_clean(s, 0.0) for s in raw_gnd]
            except Exception:
                grounding_scores = [None] * len(triples)
        else:
            grounding_scores = [None] * len(triples)

        # ── Composite ─────────────────────────────────────────────────────────
        bundles: list[TrustBundle] = []
        for i, triple in enumerate(triples):
            rel_s = relevance_scores[i]
            prov_s = _normalise_provenance(prov_counts[i])
            gnd_s = grounding_scores[i]
            ingest_s = ingest_scores[i]

            # Build weighted sum over active signals
            total_w = cfg.ingest_certainty_weight
            weighted = cfg.ingest_certainty_weight * ingest_s

            if rel_s is not None:
                total_w += cfg.relevance_weight
                weighted += cfg.relevance_weight * rel_s

            total_w += cfg.provenance_weight
            weighted += cfg.provenance_weight * prov_s

            if gnd_s is not None and cfg.grounding_enabled:
                total_w += cfg.grounding_weight
                weighted += cfg.grounding_weight * gnd_s

            composite = weighted / total_w if total_w > 0 else 0.0
            # Final guard: if anything slipped through, fall back to ingest_certainty
            if not math.isfinite(composite):
                composite = ingest_s

            bundles.append(
                TrustBundle(
                    triple=triple,
                    ingest_certainty=ingest_s,
                    relevance_score=rel_s,
                    grounding_score=gnd_s,
                    provenance_count=prov_counts[i],
                    composite_trust=composite,
                )
            )

        # ── Optional top-k filter ─────────────────────────────────────────────
        if cfg.relevance_filter_top_k is not None:
            bundles.sort(key=lambda b: b.composite_trust, reverse=True)
            bundles = bundles[: cfg.relevance_filter_top_k]

        return bundles
