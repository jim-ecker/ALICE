from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExpertMeta:
    name: str
    slug: str
    aliases: list[str]
    personality: str
    queries_ingested: list[str]
    max_docs: int
    created_at: str
    expertise_areas: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.expertise_areas is None:
            self.expertise_areas = []

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "aliases": self.aliases,
            "personality": self.personality,
            "queries_ingested": self.queries_ingested,
            "max_docs": self.max_docs,
            "created_at": self.created_at,
            "expertise_areas": self.expertise_areas,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertMeta":
        return cls(
            name=data["name"],
            slug=data["slug"],
            aliases=data.get("aliases", []),
            personality=data.get("personality", ""),
            queries_ingested=data.get("queries_ingested", []),
            max_docs=data.get("max_docs", 30),
            created_at=data.get("created_at", ""),
            expertise_areas=data.get("expertise_areas", []),
        )


class ExpertRegistry:
    def __init__(self, experts_dir: Path) -> None:
        self.experts_dir = Path(experts_dir)
        self.experts_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[ExpertMeta]:
        metas = []
        for meta_file in sorted(self.experts_dir.glob("*.meta.json")):
            try:
                with open(meta_file) as f:
                    metas.append(ExpertMeta.from_dict(json.load(f)))
            except Exception:
                pass
        return metas

    def get(self, slug: str) -> ExpertMeta | None:
        meta_file = self.experts_dir / f"{slug}.meta.json"
        if not meta_file.exists():
            return None
        with open(meta_file) as f:
            return ExpertMeta.from_dict(json.load(f))

    def create(self, name: str, max_docs: int = 30, personality: str = "") -> ExpertMeta:
        slug = self.slug_for(name)
        meta = ExpertMeta(
            name=name,
            slug=slug,
            aliases=[],
            personality=personality,
            queries_ingested=[],
            max_docs=max_docs,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write(meta)
        return meta

    def update(self, slug: str, **kwargs) -> ExpertMeta | None:
        meta = self.get(slug)
        if meta is None:
            return None
        for key, value in kwargs.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
        self._write(meta)
        return meta

    def delete(self, slug: str) -> None:
        for suffix in (".db", ".embeddings.npz", ".meta.json"):
            p = self.experts_dir / f"{slug}{suffix}"
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

    def _write(self, meta: ExpertMeta) -> None:
        meta_file = self.experts_dir / f"{meta.slug}.meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

    @staticmethod
    def slug_for(name: str) -> str:
        slug = name.lower().replace(" ", "_")
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        return slug
