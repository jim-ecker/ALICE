"""Per-triple trust-signal vector and the shared evidence-signal computation
used by both the ingest pipeline and the backfill script."""

from __future__ import annotations

from dataclasses import dataclass

from core.llm.base import LLMBackend
from core.trust.config import TrustConfig
from core.trust.ptrue import compute_ptrue


@dataclass
class TrustSignals:
    ptrue_score: float | None = None
    ptrue_method: str | None = None
    ptrue_samples: int | None = None
    nli_entailment_score: float | None = None
    nli_contradiction_score: float | None = None
    nli_model: str | None = None
    self_consistency_freq: float | None = None
    self_consistency_count: int | None = None
    self_consistency_runs: int | None = None


def compute_evidence_signals(
    backend: LLMBackend,
    *,
    subject: str,
    relation: str,
    object_: str,
    evidence_sentence: str,
    cfg: TrustConfig,
) -> TrustSignals:
    """P(True) + NLI for one triple against its evidence sentence.
    Self-consistency needs the whole-chunk rerun context and is merged in by
    the caller (see generate_triples)."""
    signals = TrustSignals()

    if cfg.ptrue_enabled:
        result = compute_ptrue(
            backend,
            evidence_sentence=evidence_sentence,
            subject=subject,
            relation=relation,
            object_=object_,
            cfg=cfg,
        )
        if result is not None:
            signals.ptrue_score = result.score
            signals.ptrue_method = result.method
            signals.ptrue_samples = result.samples

    if cfg.nli_enabled:
        from core.trust.nli import score_entailment
        from core.trust.ptrue import verbalize_triple

        scores = score_entailment(
            evidence_sentence,
            verbalize_triple(subject, relation, object_),
            model_id=cfg.nli_model,
            device=cfg.nli_device,
            max_premise_chars=cfg.nli_max_premise_chars,
        )
        if scores is not None:
            signals.nli_entailment_score, signals.nli_contradiction_score = scores
            signals.nli_model = cfg.nli_model

    return signals
