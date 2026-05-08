"""Shared query-runner logic for ALICE evaluation experiments.

Provides session setup, profile building, per-query execution, and metric
extraction. Both run_query_matrix.py and run_dataset_suite.py import from here.
"""
from __future__ import annotations

import math
import re
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_head(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

_PROFILE_KNOWN_KEYS = {
    # Retrieval
    "top_k_chunks", "max_context_chunks", "entity_hop_depth", "max_hop2_entities",
    "max_tokens", "min_composite_trust",
    # Scoring
    "ingest_certainty_weight", "relevance_weight", "provenance_weight",
    "grounding_weight", "relevance_filter_top_k",
    "ingest_certainty_cap", "ingest_certainty_exponent",
    # Meta (not passed to retriever)
    "enabled", "description",
}


def load_profiles(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Profiles file must decode to a dict: {path}")
    payload.setdefault("global", {})
    payload.setdefault("profiles", {})
    if not payload["profiles"]:
        raise ValueError(f"Profiles file missing [profiles.*] sections: {path}")
    return payload


def build_profile_opts(global_cfg: dict, profile_cfg: dict) -> dict[str, Any]:
    merged = {**global_cfg, **profile_cfg}
    unknown = sorted(k for k in merged if k not in _PROFILE_KNOWN_KEYS)
    if unknown:
        print(f"[warn] Unknown profile keys (ignored): {unknown}")
    return {k: v for k, v in merged.items() if k in _PROFILE_KNOWN_KEYS}


# ---------------------------------------------------------------------------
# Session (one DB open per profile run)
# ---------------------------------------------------------------------------

@dataclass
class EvalSession:
    retriever: Any     # services.chat.retriever.Retriever
    llm: Any           # LLMBackend
    db: Any            # kuzu.Database
    max_tokens: int
    max_context_chunks: int
    min_composite_trust: float


def open_session(
    db_path: Path,
    embeddings_path: Path,
    opts: dict[str, Any],
    embed_cfg,
    scoring_cfg_base,
    llm_cfg,
) -> EvalSession:
    """Open a kuzu DB and build a Retriever + LLM for one profile run."""
    import kuzu
    from core.embeddings.client import EmbeddingsClient
    from core.embeddings.index import EmbeddingIndex
    from core.graph.kuzu_store import _BUFFER_POOL_SIZE
    from core.llm.factory import create_backend
    from core.scoring import (
        EmbeddingRelevanceScorer,
        ProvenanceScorer,
        WeightedCompositeScorer,
    )
    from core.scoring.composite import ScoringConfig
    from services.chat.retriever import Retriever

    db = kuzu.Database(str(db_path), buffer_pool_size=_BUFFER_POOL_SIZE)
    index = EmbeddingIndex.load(embeddings_path)
    embed_client = EmbeddingsClient(embed_cfg)
    llm = create_backend(llm_cfg)

    scoring_cfg = ScoringConfig(
        ingest_certainty_weight=float(opts.get("ingest_certainty_weight", scoring_cfg_base.ingest_certainty_weight)),
        relevance_weight=float(opts.get("relevance_weight", scoring_cfg_base.relevance_weight)),
        provenance_weight=float(opts.get("provenance_weight", scoring_cfg_base.provenance_weight)),
        grounding_weight=float(opts.get("grounding_weight", scoring_cfg_base.grounding_weight)),
        grounding_enabled=scoring_cfg_base.grounding_enabled,
        relevance_filter_top_k=int(opts["relevance_filter_top_k"]) if "relevance_filter_top_k" in opts else scoring_cfg_base.relevance_filter_top_k,
        ingest_certainty_cap=float(opts.get("ingest_certainty_cap", scoring_cfg_base.ingest_certainty_cap)),
        ingest_certainty_exponent=float(opts.get("ingest_certainty_exponent", scoring_cfg_base.ingest_certainty_exponent)),
    )

    relevance_scorer = EmbeddingRelevanceScorer(embed_client)
    provenance_scorer = ProvenanceScorer(kuzu.Connection(db))
    scorer = WeightedCompositeScorer(
        cfg=scoring_cfg,
        relevance_scorer=relevance_scorer,
        provenance_scorer=provenance_scorer,
    )

    retriever = Retriever(
        index,
        embed_client,
        kuzu.Connection(db),
        scorer,
        top_k=int(opts.get("top_k_chunks", 15)),
        hop_depth=int(opts.get("entity_hop_depth", 2)),
        max_hop2_entities=int(opts.get("max_hop2_entities", 20)),
    )

    return EvalSession(
        retriever=retriever,
        llm=llm,
        db=db,
        max_tokens=int(opts.get("max_tokens", 2048)),
        max_context_chunks=int(opts.get("max_context_chunks", 20)),
        min_composite_trust=float(opts.get("min_composite_trust", 0.0)),
    )


def close_session(session: EvalSession) -> None:
    try:
        session.db.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def run_query(session: EvalSession, query: str) -> tuple[str, Any, float]:
    """Run one query. Returns (answer, retrieval, elapsed_seconds)."""
    from core.scoring.base import ScoredRetrievalResult
    from services.chat.prompt import build_prompt

    t0 = time.perf_counter()
    retrieval = session.retriever.retrieve(query)

    if session.min_composite_trust > 0.0:
        filtered = [b for b in retrieval.trust_bundles if b.composite_trust >= session.min_composite_trust]
        retrieval = ScoredRetrievalResult(
            chunks=retrieval.chunks,
            trust_bundles=filtered,
            embedding_chunk_ids=retrieval.embedding_chunk_ids,
        )

    messages, _ = build_prompt(query, retrieval, [], max_context_chunks=session.max_context_chunks)
    answer = session.llm.chat(messages, session.max_tokens)
    elapsed = time.perf_counter() - t0
    return answer, retrieval, elapsed


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(answer: str, retrieval: Any) -> dict[str, Any]:
    """Pull per-query metrics from an answer string and retrieval result."""
    citations = set(re.findall(r"Fact_\d+", answer))
    bundles = retrieval.trust_bundles

    avg_composite = _safe_avg(b.composite_trust for b in bundles)
    avg_ingest = _safe_avg(b.ingest_certainty for b in bundles)
    rel_vals = [b.relevance_score for b in bundles if b.relevance_score is not None]
    avg_relevance = _safe_avg(rel_vals)

    return {
        "chunks_retrieved": len(retrieval.chunks),
        "facts_retrieved": len(bundles),
        "avg_composite_trust": round(avg_composite, 4),
        "avg_ingest_certainty": round(avg_ingest, 4),
        "avg_relevance_score": round(avg_relevance, 4),
        "citations_count": len(citations),
    }


def _safe_avg(values) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Answer scoring (EM / F1)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    return " ".join(value.split())


def answer_em(prediction: str, gold_answers: list[str]) -> float:
    pred = normalize_text(prediction)
    return 1.0 if gold_answers and any(pred == normalize_text(a) for a in gold_answers) else 0.0


def answer_f1(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens or not gold_answers:
        return 0.0

    def f1_one(gold: str) -> float:
        gold_tokens = normalize_text(gold).split()
        if not gold_tokens:
            return 0.0
        gold_counts: dict[str, int] = {}
        for t in gold_tokens:
            gold_counts[t] = gold_counts.get(t, 0) + 1
        common = 0
        for t in pred_tokens:
            if gold_counts.get(t, 0) > 0:
                common += 1
                gold_counts[t] -= 1
        if common == 0:
            return 0.0
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(f1_one(a) for a in gold_answers)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def summarize(rows: list[dict[str, Any]], *, include_qa_metrics: bool = False) -> dict[str, Any]:
    if not rows:
        base = {
            "count": 0,
            "avg_runtime_seconds": 0.0,
            "avg_chunks_retrieved": 0.0,
            "avg_facts_retrieved": 0.0,
            "avg_composite_trust": 0.0,
            "avg_citations_count": 0.0,
        }
        if include_qa_metrics:
            base["avg_answer_em"] = 0.0
            base["avg_answer_f1"] = 0.0
        return base

    def avg(key: str) -> float:
        return round(_safe_avg(float(r.get(key) or 0.0) for r in rows), 4)

    result = {
        "count": len(rows),
        "avg_runtime_seconds": avg("runtime_seconds"),
        "avg_chunks_retrieved": avg("chunks_retrieved"),
        "avg_facts_retrieved": avg("facts_retrieved"),
        "avg_composite_trust": avg("avg_composite_trust"),
        "avg_citations_count": avg("citations_count"),
    }
    if include_qa_metrics:
        result["avg_answer_em"] = avg("answer_em")
        result["avg_answer_f1"] = avg("answer_f1")
    return result
