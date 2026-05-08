"""Live expert querying for the experiment evaluation workbench.

Builds an isolated Retriever per expert without mutating the global ServiceState.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def query_expert(
    expert_slug: str,
    question: str,
    cfg: Any,            # ChatConfig
    embed_client: Any,   # shared EmbeddingsClient (HTTP — safe to reuse)
    llm: Any,            # shared LLMBackend (HTTP — safe to reuse)
    scoring_cfg: Any,    # shared ScoringConfig
) -> str:
    """Query one virtual expert's knowledge base.

    Opens the expert's DB, runs retrieval, builds prompt, calls LLM.
    Closes the DB before returning. Does NOT modify global ServiceState.
    """
    import kuzu

    from core.embeddings.index import EmbeddingIndex
    from core.graph.kuzu_store import _BUFFER_POOL_SIZE
    from core.scoring import (
        EmbeddingRelevanceScorer,
        ProvenanceScorer,
        WeightedCompositeScorer,
    )
    from services.chat.prompt import build_prompt
    from services.chat.retriever import Retriever
    from services.experts.manager import ExpertRegistry
    from services.experts.paths import build_expert_paths

    registry = ExpertRegistry(cfg.experts_dir)
    meta = registry.get(expert_slug)
    paths = build_expert_paths(cfg.experts_dir, expert_slug)

    db = kuzu.Database(str(paths.db_path), buffer_pool_size=_BUFFER_POOL_SIZE)
    try:
        conn1 = kuzu.Connection(db)
        conn2 = kuzu.Connection(db)

        index = EmbeddingIndex.load(paths.embeddings_path)

        relevance_scorer = EmbeddingRelevanceScorer(embed_client)
        provenance_scorer = ProvenanceScorer(conn1)
        scorer = WeightedCompositeScorer(
            cfg=scoring_cfg,
            relevance_scorer=relevance_scorer,
            provenance_scorer=provenance_scorer,
        )

        retriever = Retriever(
            index,
            embed_client,
            conn2,
            scorer,
            top_k=cfg.top_k_chunks,
            hop_depth=cfg.entity_hop_depth,
            max_hop2_entities=cfg.max_hop2_entities,
        )

        retrieval = retriever.retrieve(question)
        messages, _ = build_prompt(
            question,
            retrieval,
            [],
            max_context_chunks=cfg.max_context_chunks,
            expert_name=meta.name if meta else None,
            expert_persona=meta.personality if meta else None,
            expert_persona_strength=meta.personality_strength if meta else 1.0,
        )
        return llm.chat(messages, cfg.max_tokens)
    finally:
        try:
            db.close()
        except Exception:
            pass


def query_generic_llm(question: str, llm: Any, max_tokens: int = 1024) -> str:
    """Answer a question using the LLM alone, without any knowledge graph context."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. Answer the following question "
                "as accurately and concisely as you can based on your training knowledge."
            ),
        },
        {"role": "user", "content": question},
    ]
    return llm.chat(messages, max_tokens)
