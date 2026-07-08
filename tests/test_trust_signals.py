from __future__ import annotations

import json

import pytest

from core.llm.base import LLMBackend
from core.llm.generate_triple import generate_triples
from core.trust.config import TrustConfig
from core.trust.consistency import canonical_key, count_across_runs
from core.trust.nli import nli_available
from core.trust.ptrue import compute_ptrue, parse_yes_no, vote_posterior_mean

_JWST_ROW = {
    "subject": "NASA",
    "subject_type": "ORGANIZATION",
    "relation": "manages",
    "object": "James Webb Space Telescope program",
    "object_type": "PROGRAM",
    "certainty": 0.8,
    "evidence_text": "NASA manages the James Webb Space Telescope program.",
}
_JWST_CONTENT = "NASA manages the James Webb Space Telescope program."


class TrustFakeBackend(LLMBackend):
    def __init__(
        self,
        payload: list[dict],
        *,
        yes_no: float | None = None,
        samples: list[str] | None = None,
    ) -> None:
        self._payload = payload
        self._yes_no = yes_no
        self._samples = samples or []

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
        del messages, max_tokens
        return json.dumps(self._payload)

    def score_yes_no(self, messages: list[dict[str, str]], *, max_tokens: int = 4) -> float | None:
        del messages, max_tokens
        return self._yes_no

    def sample_n(self, messages, *, n, temperature=0.7, max_tokens=1024) -> list[str]:
        del messages, temperature, max_tokens
        return self._samples[:n]


def test_vote_posterior_mean() -> None:
    assert vote_posterior_mean(0, 4) == pytest.approx(1 / 6)
    assert vote_posterior_mean(4, 4) == pytest.approx(5 / 6)
    assert vote_posterior_mean(2, 4) == pytest.approx(0.5)


def test_parse_yes_no_handles_preamble_and_case() -> None:
    assert parse_yes_no("Yes.") is True
    assert parse_yes_no("  no") is False
    assert parse_yes_no("I think the answer is Yes") is True
    assert parse_yes_no("The text does not support it. No.") is False
    assert parse_yes_no("unsure") is None
    assert parse_yes_no("") is None


def test_compute_ptrue_prefers_logprobs() -> None:
    backend = TrustFakeBackend([], yes_no=0.83)
    result = compute_ptrue(
        backend,
        evidence_sentence=_JWST_CONTENT,
        subject="NASA",
        relation="manages",
        object_="JWST",
        cfg=TrustConfig(enabled=True),
    )
    assert result is not None
    assert result.method == "logprob"
    assert result.score == 0.83
    assert result.samples == 0


def test_compute_ptrue_vote_fallback() -> None:
    backend = TrustFakeBackend([], yes_no=None, samples=["Yes", "Yes", "No", "Yes", "huh"])
    result = compute_ptrue(
        backend,
        evidence_sentence=_JWST_CONTENT,
        subject="NASA",
        relation="manages",
        object_="JWST",
        cfg=TrustConfig(enabled=True, ptrue_fallback_samples=5),
    )
    assert result is not None
    assert result.method == "vote"
    assert result.samples == 4  # "huh" is unparseable and dropped
    assert result.score == pytest.approx((3 + 1) / (4 + 2))


def test_compute_ptrue_returns_none_when_no_votes_parse() -> None:
    backend = TrustFakeBackend([], yes_no=None, samples=["hmm", "???"])
    result = compute_ptrue(
        backend,
        evidence_sentence=_JWST_CONTENT,
        subject="NASA",
        relation="manages",
        object_="JWST",
        cfg=TrustConfig(enabled=True, ptrue_fallback_samples=2),
    )
    assert result is None


def test_canonical_key_normalizes_case_and_whitespace() -> None:
    assert canonical_key("NASA  Langley", "Develops", "Drones\n") == canonical_key(
        "nasa langley", "develops", "drones"
    )


def test_count_across_runs_frequency_and_beta() -> None:
    key = canonical_key("NASA", "manages", "JWST")
    other = canonical_key("ESA", "manages", "Ariane")
    results = count_across_runs({key}, [{key, other}, {other}])
    result = results[key]
    assert result.count == 2
    assert result.runs == 3
    assert result.freq == pytest.approx(2 / 3)
    assert result.alpha == 3.0
    assert result.beta == 2.0


def test_generate_triples_attaches_signals() -> None:
    rerun_with = json.dumps([_JWST_ROW])
    rerun_without = json.dumps([])
    backend = TrustFakeBackend([_JWST_ROW], yes_no=0.83, samples=[rerun_with, rerun_without])
    cfg = TrustConfig(
        enabled=True,
        nli_enabled=False,
        self_consistency_enabled=True,
        self_consistency_k=2,
    )

    triples = generate_triples("chunk-1", _JWST_CONTENT, backend, trust_cfg=cfg)

    assert len(triples) == 1
    triple = triples[0]
    assert triple.confidence_version == "v3_signals"
    assert triple.ptrue_score == 0.83
    assert triple.ptrue_method == "logprob"
    assert triple.ptrue_samples == 0
    assert triple.self_consistency_count == 2  # base run + one rerun
    assert triple.self_consistency_runs == 3
    assert triple.self_consistency_freq == pytest.approx(2 / 3)
    assert triple.nli_entailment_score is None
    # v2 scoring is unchanged underneath
    assert triple.certainty_score > 0.6


def test_generate_triples_without_trust_cfg_is_pure_v2() -> None:
    backend = TrustFakeBackend([_JWST_ROW], yes_no=0.83)

    triples = generate_triples("chunk-1", _JWST_CONTENT, backend)

    assert len(triples) == 1
    triple = triples[0]
    assert triple.confidence_version == "v2_evidence_hybrid"
    assert triple.ptrue_score is None
    assert triple.nli_entailment_score is None
    assert triple.self_consistency_freq is None


def test_generate_triples_with_trust_disabled_is_pure_v2() -> None:
    backend = TrustFakeBackend([_JWST_ROW], yes_no=0.83)

    triples = generate_triples("chunk-1", _JWST_CONTENT, backend, trust_cfg=TrustConfig())

    assert triples[0].confidence_version == "v2_evidence_hybrid"
    assert triples[0].ptrue_score is None


def test_nli_skipped_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.trust.nli as nli_module

    monkeypatch.setattr(nli_module, "nli_available", lambda: False)
    backend = TrustFakeBackend([_JWST_ROW], yes_no=0.83)
    cfg = TrustConfig(enabled=True, nli_enabled=True)

    triples = generate_triples("chunk-1", _JWST_CONTENT, backend, trust_cfg=cfg)

    assert len(triples) == 1
    assert triples[0].confidence_version == "v3_signals"
    assert triples[0].ptrue_score == 0.83
    assert triples[0].nli_entailment_score is None
    assert triples[0].nli_model is None


@pytest.mark.skipif(not nli_available(), reason="trust extra (transformers/torch) not installed")
def test_nli_scores_negation() -> None:
    from core.trust.config import DEFAULT_NLI_MODEL
    from core.trust.nli import score_entailment

    scores = score_entailment(
        "The battery does not power the controller.",
        "battery powers controller.",
        model_id=DEFAULT_NLI_MODEL,
    )
    assert scores is not None
    p_entail, p_contra = scores
    assert p_contra > p_entail

    supported = score_entailment(
        "The battery powers the controller during flight.",
        "battery powers controller.",
        model_id=DEFAULT_NLI_MODEL,
    )
    assert supported is not None
    assert supported[0] > supported[1]
