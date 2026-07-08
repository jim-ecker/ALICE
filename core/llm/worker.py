"""Module-level worker functions for ProcessPoolExecutor-based parallel extraction.

Functions must be at module level to be picklable by ProcessPoolExecutor.
Each worker process loads its own model instance via init_worker().
"""

_backend = None    # one instance per worker process
_trust_cfg = None  # TrustConfig | None — shared, read-only


def init_worker(cfg, trust_cfg=None) -> None:   # cfg: LLMConfig
    global _backend, _trust_cfg
    from core.llm.factory import create_backend
    _backend = create_backend(cfg)
    _trust_cfg = trust_cfg
    if trust_cfg is not None and trust_cfg.enabled and trust_cfg.nli_enabled:
        # Warm up the NLI singleton so the model loads before the first chunk,
        # not in the middle of extraction. No-op if the trust extra is absent.
        from core.trust.nli import score_entailment
        score_entailment(
            "warm up.", "warm up.",
            model_id=trust_cfg.nli_model,
            device=trust_cfg.nli_device,
        )


def extract_chunk(chunk_id: str, document_id: str, content: str) -> tuple[str, list]:
    from core.llm.generate_triple import generate_triples
    triples = generate_triples(chunk_id, content, _backend, trust_cfg=_trust_cfg)
    return (document_id, triples)
