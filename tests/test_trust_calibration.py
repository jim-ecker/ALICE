from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from core.graph.kuzu_store import KuzuStore
from core.trust.calibration import (
    apply_calibrator,
    build_feature_matrix,
    expected_calibration_error,
    fit_calibrator,
    load_artifact,
    save_artifact,
)
from core.trust.judge import parse_verdict
from core.trust.ppi import cohens_kappa, ppi_mean

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from mint_calibration_labels import sample_triples  # noqa: E402


# ── judge parsing ────────────────────────────────────────────────────────────

def test_parse_verdict_variants() -> None:
    assert parse_verdict("Supported") == "supported"
    assert parse_verdict("NotSupported") == "not_supported"
    assert parse_verdict("not supported") == "not_supported"
    assert parse_verdict("Verdict: Unclear.") == "unclear"
    assert parse_verdict("The passage clearly says so: Supported") == "supported"
    assert parse_verdict("hmm") is None
    assert parse_verdict("") is None


# ── calibration math ─────────────────────────────────────────────────────────

def _synthetic_rows(n: int, seed: int = 7) -> tuple[list[dict], list[int]]:
    """Signals informative of the label, with some v3 values missing."""
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        label = rng.random() < 0.6
        good = 0.75 if label else 0.3
        row = {
            "ptrue_score": min(1.0, max(0.0, rng.gauss(good, 0.15))) if rng.random() > 0.2 else None,
            "nli_entailment_score": min(1.0, max(0.0, rng.gauss(good, 0.1))),
            "nli_contradiction_score": min(1.0, max(0.0, rng.gauss(1 - good, 0.1))),
            "self_consistency_freq": None,  # e.g. backfilled rows
            "evidence_alignment_score": 1.0 if rng.random() > 0.3 else 0.8,
            "entity_anchor_score": 1.0,
            "evidence_scope_score": 0.9,
            "raw_certainty_score": min(1.0, max(0.0, rng.gauss(0.85, 0.05))),
        }
        rows.append(row)
        labels.append(int(label))
    return rows, labels


def test_feature_matrix_imputes_and_flags_missing() -> None:
    rows = [
        {"ptrue_score": 0.8, "nli_entailment_score": 0.9, "nli_contradiction_score": 0.1,
         "self_consistency_freq": None, "evidence_alignment_score": 1.0,
         "entity_anchor_score": 1.0, "evidence_scope_score": 1.0, "raw_certainty_score": 0.9},
        {"ptrue_score": None, "nli_entailment_score": 0.7, "nli_contradiction_score": 0.2,
         "self_consistency_freq": None, "evidence_alignment_score": 0.8,
         "entity_anchor_score": 0.75, "evidence_scope_score": 0.8, "raw_certainty_score": 0.7},
    ]
    X, means = build_feature_matrix(rows)
    assert means["ptrue_score"] == pytest.approx(0.8)  # mean of non-missing
    assert X[1, 0] == pytest.approx(0.8)               # imputed
    assert X[0, 8] == 0.0 and X[1, 8] == 1.0           # ptrue_missing indicator
    assert X[0, 10] == 1.0 and X[1, 10] == 1.0         # consistency_missing indicator


def test_ece_zero_when_perfectly_calibrated() -> None:
    import numpy as np

    probs = np.array([0.25] * 4 + [0.75] * 4)
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 0])  # 1/4 and 3/4 empirical
    assert expected_calibration_error(probs, labels, n_bins=4) == pytest.approx(0.0)


def test_fit_calibrator_learns_and_round_trips(tmp_path: Path) -> None:
    rows, labels = _synthetic_rows(400)
    result = fit_calibrator(rows, labels, metadata={"source": "synthetic"})

    assert result.metrics["auroc"] > 0.8          # informative signals are learned
    assert result.metrics["ece_calibrated"] < 0.1 # isotonic yields calibrated output
    assert result.artifact["version"] == "v4_calibrated"

    path = tmp_path / "calibrator.json"
    save_artifact(result.artifact, path)
    loaded = load_artifact(path)

    strong = dict(rows[0], ptrue_score=0.95, nli_entailment_score=0.97, nli_contradiction_score=0.02)
    weak = dict(rows[0], ptrue_score=0.1, nli_entailment_score=0.15, nli_contradiction_score=0.8)
    p_strong = apply_calibrator(loaded, strong)
    p_weak = apply_calibrator(loaded, weak)
    assert 0.0 <= p_weak < p_strong <= 1.0


def test_fit_calibrator_rejects_single_class() -> None:
    rows, _ = _synthetic_rows(20)
    with pytest.raises(ValueError):
        fit_calibrator(rows, [1] * 20)


# ── PPI ──────────────────────────────────────────────────────────────────────

def test_ppi_mean_corrects_judge_bias() -> None:
    # Judge says 1 for 80/100; on the audit the judge over-calls by 0.2
    judge_all = [1.0] * 80 + [0.0] * 20
    judge_audit = [1.0] * 10 + [0.0] * 10
    human_audit = [1.0] * 6 + [0.0] * 14
    est = ppi_mean(judge_all, judge_audit, human_audit)
    assert est.judge_mean_all == pytest.approx(0.8)
    assert est.rectifier == pytest.approx(0.2)
    assert est.theta == pytest.approx(0.6)
    assert est.ci_low < 0.6 < est.ci_high

def test_ppi_unbiased_judge_has_zero_rectifier() -> None:
    judge_all = [1.0, 0.0] * 50
    audit = [1.0, 0.0] * 10
    est = ppi_mean(judge_all, audit, list(audit))
    assert est.rectifier == 0.0
    assert est.theta == pytest.approx(0.5)


def test_cohens_kappa() -> None:
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # Independent labels — kappa near 0
    assert abs(cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


# ── DB sampling ──────────────────────────────────────────────────────────────

def test_sample_triples_stratifies_by_document(tmp_path: Path) -> None:
    db_path = tmp_path / "sample.db"
    with KuzuStore(db_path) as store:
        for doc in range(2):
            store.write_document(f"doc-{doc}", None, "text", f"Doc {doc}")
            store.write_chunk(f"chunk-{doc}", f"doc-{doc}", f"Content of doc {doc}.", None, 1)
        # doc-0 has 6 triples, doc-1 has 2 — round-robin should still pull from both
        for doc, count in ((0, 6), (1, 2)):
            for i in range(count):
                store.write_triple(
                    subject=f"S{doc}-{i}", subject_type="T", relation="rel",
                    object_=f"O{doc}-{i}", object_type="T",
                    certainty_score=0.9, chunk_id=f"chunk-{doc}",
                    evidence_text=f"evidence {doc}-{i}",
                )

        sampled = sample_triples(store._conn, 4, seed=1)

    assert len(sampled) == 4
    docs = {r["document_id"] for r in sampled}
    assert docs == {"doc-0", "doc-1"}          # both strata represented
    assert all(r["chunk_text"].startswith("Content") for r in sampled)
    assert all("raw_certainty_score" in r for r in sampled)
