from __future__ import annotations

from pathlib import Path

from services.experts.manager import ExpertMeta


def ingest_for_expert(
    meta: ExpertMeta,
    author: str,
    experts_dir: Path,
    llm_cfg,
    embed_client,
    *,
    on_download=None,
    on_downloads_complete=None,
    on_chunk=None,
    on_extract_start=None,
    on_chunk_extracted=None,
):
    """Run the ingest pipeline for an expert using NTRS author filtering.

    Uses the NTRS `author` query parameter (a dedicated author filter, not a
    text search) so results are scoped to papers authored by `author`.
    Multiple calls accumulate — existing graph data is not wiped.
    """
    from services.ingest.service import Ingest

    experts_dir = Path(experts_dir)
    db_path = experts_dir / f"{meta.slug}.db"
    embeddings_path = experts_dir / f"{meta.slug}.embeddings.npz"
    downloads_dir = experts_dir / "downloads"

    ingest_svc = Ingest(
        db_path=db_path,
        llm_cfg=llm_cfg,
        embed_client=embed_client,
        embeddings_path=embeddings_path,
        downloads_dir=downloads_dir,
    )
    result, _ = ingest_svc.run(
        "*",
        author=author,
        max_docs=meta.max_docs,
        on_download=on_download,
        on_downloads_complete=on_downloads_complete,
        on_chunk=on_chunk,
        on_extract_start=on_extract_start,
        on_chunk_extracted=on_chunk_extracted,
    )
    return result
