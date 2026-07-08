from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass
class TrustConfig:
    """Settings for the v3 measurable trust-signal layer ([trust] in alice.toml).

    With enabled=False (the default) the extraction pipeline behaves exactly as
    before — no extra LLM calls, no NLI model load, v2 confidence only.
    """

    enabled: bool = False

    # P(True) verification — one short LLM call per unique triple
    ptrue_enabled: bool = True
    ptrue_fallback_samples: int = 5   # m votes when the backend has no logprobs
    ptrue_temperature: float = 1.0    # sampling temperature for the vote fallback
    # Vote-path token budget. Reasoning models (gpt-oss) burn tokens on hidden
    # reasoning before the one-word answer lands in content — too small a budget
    # yields empty responses and no parseable votes.
    ptrue_vote_max_tokens: int = 256

    # NLI entailment — local cross-encoder, requires `uv sync --extra trust`
    nli_enabled: bool = True
    nli_model: str = DEFAULT_NLI_MODEL
    nli_device: str = "auto"          # auto -> mps / cuda / cpu
    nli_max_premise_chars: int = 1000

    # Self-consistency — k extra full-chunk extractions, (k+1)x ingest cost
    self_consistency_enabled: bool = False
    self_consistency_k: int = 3
    self_consistency_temperature: float = 0.7

    # Calibration judge — the LLM that labels (triple, chunk) pairs for the
    # Phase 2 calibration set. Should differ from the extraction model so the
    # extractor isn't grading its own homework; human audit corrects the rest.
    judge_backend: str = "openai-compatible"  # "openai-compatible" | "anthropic"
    judge_model: str = ""
    judge_base_url: str = "http://localhost:11434/v1"
    judge_api_key_env: str = ""               # env var holding the API key, e.g. ANTHROPIC_API_KEY


def _load_toml(start_dir: Path | None) -> dict:
    import tomllib

    search = (start_dir or Path.cwd()).resolve()
    for directory in [search, *search.parents]:
        candidate = directory / "alice.toml"
        if candidate.exists():
            with open(candidate, "rb") as f:
                return tomllib.load(f)
    return {}


def load_trust_config(start_dir: Path | None = None) -> TrustConfig:
    raw = _load_toml(start_dir).get("trust", {})
    defaults = TrustConfig()
    return TrustConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        ptrue_enabled=bool(raw.get("ptrue_enabled", defaults.ptrue_enabled)),
        ptrue_fallback_samples=int(raw.get("ptrue_fallback_samples", defaults.ptrue_fallback_samples)),
        ptrue_temperature=float(raw.get("ptrue_temperature", defaults.ptrue_temperature)),
        ptrue_vote_max_tokens=int(raw.get("ptrue_vote_max_tokens", defaults.ptrue_vote_max_tokens)),
        nli_enabled=bool(raw.get("nli_enabled", defaults.nli_enabled)),
        nli_model=str(raw.get("nli_model", defaults.nli_model)),
        nli_device=str(raw.get("nli_device", defaults.nli_device)),
        nli_max_premise_chars=int(raw.get("nli_max_premise_chars", defaults.nli_max_premise_chars)),
        self_consistency_enabled=bool(
            raw.get("self_consistency_enabled", defaults.self_consistency_enabled)
        ),
        self_consistency_k=int(raw.get("self_consistency_k", defaults.self_consistency_k)),
        self_consistency_temperature=float(
            raw.get("self_consistency_temperature", defaults.self_consistency_temperature)
        ),
        judge_backend=str(raw.get("judge_backend", defaults.judge_backend)),
        judge_model=str(raw.get("judge_model", defaults.judge_model)),
        judge_base_url=str(raw.get("judge_base_url", defaults.judge_base_url)),
        judge_api_key_env=str(raw.get("judge_api_key_env", defaults.judge_api_key_env)),
    )
