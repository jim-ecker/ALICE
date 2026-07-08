from __future__ import annotations

import json
import sys
from pathlib import Path

from core.graph.kuzu_store import KuzuStore
from core.graph.retrieval import retrieve_context
from core.llm.base import LLMBackend
from core.trust.config import TrustConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from rescore_trust_signals import rescore  # noqa: E402


def _write_doc_and_chunk(store: KuzuStore) -> None:
    store.write_document("doc-1", "https://example.com/doc", "pdf", "Example Doc")
    store.write_chunk("chunk-1", "doc-1", "NASA manages JWST.", "Intro", 1)


def test_trust_signal_columns_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "signals.db"
    with KuzuStore(db_path) as store:
        _write_doc_and_chunk(store)
        store.write_triple(
            subject="NASA",
            subject_type="ORGANIZATION",
            relation="manages",
            object_="JWST",
            object_type="PROGRAM",
            certainty_score=0.88,
            chunk_id="chunk-1",
            evidence_text="NASA manages JWST.",
            confidence_version="v3_signals",
            ptrue_score=0.91,
            ptrue_method="logprob",
            ptrue_samples=0,
            nli_entailment_score=0.87,
            nli_contradiction_score=0.02,
            nli_model="test-nli-model",
            self_consistency_freq=0.75,
            self_consistency_count=3,
            self_consistency_runs=4,
        )

        result = retrieve_context(store._conn, ["chunk-1"])

    assert len(result.triples) == 1
    triple = result.triples[0]
    assert triple.confidence_version == "v3_signals"
    assert triple.ptrue_score == 0.91
    assert triple.ptrue_method == "logprob"
    assert triple.ptrue_samples == 0
    assert triple.nli_entailment_score == 0.87
    assert triple.nli_contradiction_score == 0.02
    assert triple.nli_model == "test-nli-model"
    assert triple.self_consistency_freq == 0.75
    assert triple.self_consistency_count == 3
    assert triple.self_consistency_runs == 4


def test_legacy_rows_read_as_null_signals(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with KuzuStore(db_path) as store:
        _write_doc_and_chunk(store)
        store._conn.execute("MERGE (e:Entity {name: 'NASA'}) SET e.type = 'ORGANIZATION'")
        store._conn.execute("MERGE (e:Entity {name: 'JWST'}) SET e.type = 'PROGRAM'")
        store._conn.execute(
            """
            MATCH (s:Entity {name: 'NASA'}), (o:Entity {name: 'JWST'})
            CREATE (s)-[:RELATES_TO {
                relation: 'manages',
                certainty_score: 0.72,
                chunk_id: 'chunk-1'
            }]->(o)
            """
        )

        result = retrieve_context(store._conn, ["chunk-1"])

    triple = result.triples[0]
    assert triple.ptrue_score is None
    assert triple.ptrue_method is None
    assert triple.nli_entailment_score is None
    assert triple.self_consistency_freq is None


class RescoreFakeBackend(LLMBackend):
    def __init__(self, yes_no: float) -> None:
        self._yes_no = yes_no
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
        del messages, max_tokens
        return json.dumps([])

    def score_yes_no(self, messages: list[dict[str, str]], *, max_tokens: int = 4) -> float | None:
        del messages, max_tokens
        self.calls += 1
        return self._yes_no


def test_rescore_updates_null_signal_columns_and_is_resumable(tmp_path: Path) -> None:
    db_path = tmp_path / "rescore.db"
    with KuzuStore(db_path) as store:
        _write_doc_and_chunk(store)
        store.write_triple(
            subject="NASA",
            subject_type="ORGANIZATION",
            relation="manages",
            object_="JWST",
            object_type="PROGRAM",
            certainty_score=0.88,
            chunk_id="chunk-1",
            evidence_text="NASA manages JWST.",
            evidence_char_start=0,
            evidence_char_end=18,
            confidence_version="v2_evidence_hybrid",
        )

        backend = RescoreFakeBackend(yes_no=0.9)
        cfg = TrustConfig(enabled=True, nli_enabled=False)

        stats = rescore(store._conn, backend, cfg, limit=None, only="ptrue", force=False)
        assert stats.scored == 1
        assert backend.calls == 1

        result = retrieve_context(store._conn, ["chunk-1"])
        triple = result.triples[0]
        assert triple.ptrue_score == 0.9
        assert triple.ptrue_method == "logprob"
        # confidence_version is left untouched on backfilled rows
        assert triple.confidence_version == "v2_evidence_hybrid"

        # Second run finds nothing to do — the NULL filter makes reruns resumable
        stats2 = rescore(store._conn, backend, cfg, limit=None, only="ptrue", force=False)
        assert stats2.scored == 0
        assert backend.calls == 1
