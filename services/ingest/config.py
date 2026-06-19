from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestConfig:
    min_ingest_confidence: float = 0.6


def _load_toml(start_dir: Path | None) -> dict:
    import tomllib

    search = (start_dir or Path.cwd()).resolve()
    for directory in [search, *search.parents]:
        candidate = directory / "alice.toml"
        if candidate.exists():
            with open(candidate, "rb") as f:
                return tomllib.load(f)
    return {}


def load_ingest_config(start_dir: Path | None = None) -> IngestConfig:
    data = _load_toml(start_dir)
    ingest_raw = data.get("ingest", {})
    return IngestConfig(
        min_ingest_confidence=float(ingest_raw.get("min_ingest_confidence", 0.6)),
    )


def load_ingest_llm_config(start_dir: Path | None = None):
    """Return an LLMConfig from [ingest_llm] if present, else None (caller falls back to [llm])."""
    from core.llm.config import LLMConfig

    data = _load_toml(start_dir)
    raw = data.get("ingest_llm")
    if raw is None:
        return None
    return LLMConfig(
        backend=raw.get("backend", "auto"),
        model=raw.get("model", ""),
        base_url=raw.get("base_url", ""),
        api_key=raw.get("api_key", "token"),
        workers=raw.get("workers", 2),
    )
