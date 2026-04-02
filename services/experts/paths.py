from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpertPaths:
    expert_dir: Path
    db_path: Path
    embeddings_path: Path
    downloads_dir: Path
    meta_path: Path


def default_experts_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def build_expert_paths(experts_dir: Path | str, slug: str) -> ExpertPaths:
    experts_root = Path(experts_dir)
    expert_dir = experts_root / slug
    return ExpertPaths(
        expert_dir=expert_dir,
        db_path=expert_dir / f"{slug}.db",
        embeddings_path=expert_dir / f"{slug}.embeddings.npz",
        downloads_dir=expert_dir / "downloads",
        meta_path=expert_dir / f"{slug}.meta.json",
    )
