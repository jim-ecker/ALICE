from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestConfig:
    min_ingest_confidence: float = 0.6


def load_ingest_config(start_dir: Path | None = None) -> IngestConfig:
    import tomllib

    search = (start_dir or Path.cwd()).resolve()
    data: dict = {}
    for directory in [search, *search.parents]:
        candidate = directory / "alice.toml"
        if candidate.exists():
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            break

    ingest_raw = data.get("ingest", {})
    return IngestConfig(
        min_ingest_confidence=float(ingest_raw.get("min_ingest_confidence", 0.6)),
    )
