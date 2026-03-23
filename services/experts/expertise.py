"""Infer expertise areas from an expert's NTRS subject categories."""
from __future__ import annotations

from pathlib import Path


def _get_citation_urls(db_path: Path) -> list[str]:
    """Query all document citation URLs from the expert's Kuzu DB."""
    try:
        import kuzu

        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)
        try:
            result = conn.execute("MATCH (d:Document) RETURN d.source_url")
            urls = []
            while result.has_next():
                row = result.get_next()
                if row[0] and isinstance(row[0], str):
                    urls.append(row[0])
            return urls
        finally:
            conn.close()
            db.close()
    except Exception:
        return []


def compute_expertise_areas(db_path: Path, llm_cfg=None) -> list[str]:
    """Return unique NTRS subject categories for all documents in the expert's DB.

    Fetches subject category metadata from the NTRS API using the citation URLs
    already stored in the knowledge graph. No LLM required.
    """
    if not Path(db_path).exists():
        return []

    urls = _get_citation_urls(db_path)
    if not urls:
        return []

    from services.ingest.ntrs import ntrs_id_from_url
    import requests

    seen: set[str] = set()
    areas: list[str] = []

    for url in urls:
        ntrs_id = ntrs_id_from_url(url)
        if not ntrs_id:
            continue
        try:
            resp = requests.get(
                f"https://ntrs.nasa.gov/api/citations/{ntrs_id}", timeout=15
            )
            resp.raise_for_status()
            for cat in resp.json().get("subjectCategories", []):
                if cat and cat not in seen:
                    seen.add(cat)
                    areas.append(cat)
        except Exception:
            continue

    return areas
