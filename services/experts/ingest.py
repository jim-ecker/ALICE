from __future__ import annotations

from pathlib import Path

from services.experts.manager import ExpertMeta
from services.experts.paths import build_expert_paths

_ALL_EXTS = ("pdf", "pptx", "docx", "html", "htm", "md", "adoc", "xlsx", "xls")


def ingest_for_expert(
    meta: ExpertMeta,
    author: str,
    experts_dir: Path,
    llm_cfg,
    embed_client,
    *,
    on_search_complete=None,
    on_download=None,
    on_download_failed=None,
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
    paths = build_expert_paths(experts_dir, meta.slug)
    paths.expert_dir.mkdir(parents=True, exist_ok=True)

    ingest_svc = Ingest(
        db_path=paths.db_path,
        llm_cfg=llm_cfg,
        embed_client=embed_client,
        embeddings_path=paths.embeddings_path,
        downloads_dir=paths.downloads_dir,
    )
    result, _ = ingest_svc.run(
        "*",
        author=author,
        max_docs=meta.max_docs,
        on_search_complete=on_search_complete,
        on_download=on_download,
        on_download_failed=on_download_failed,
        on_downloads_complete=on_downloads_complete,
        on_chunk=on_chunk,
        on_extract_start=on_extract_start,
        on_chunk_extracted=on_chunk_extracted,
    )
    return result


def ingest_folder_for_expert(
    meta: ExpertMeta,
    folder: Path,
    experts_dir: Path,
    llm_cfg,
    embed_client,
    confirmed_docs: list[tuple[Path, dict]],
    *,
    on_chunk=None,
    on_extract_start=None,
    on_chunk_extracted=None,
):
    """Ingest local documents from a folder into an expert's knowledge graph.

    confirmed_docs: list of (file_path, meta_dict) tuples with pre-verified metadata.
    Multiple calls accumulate — existing graph data is not wiped.
    """
    from services.ingest.service import Ingest

    experts_dir = Path(experts_dir)
    paths = build_expert_paths(experts_dir, meta.slug)
    paths.expert_dir.mkdir(parents=True, exist_ok=True)

    ingest_svc = Ingest(
        db_path=paths.db_path,
        llm_cfg=llm_cfg,
        embed_client=embed_client,
        embeddings_path=paths.embeddings_path,
    )
    result, _ = ingest_svc.ingest_local_files(
        confirmed_docs,
        on_chunk=on_chunk,
        on_extract_start=on_extract_start,
        on_chunk_extracted=on_chunk_extracted,
    )
    return result
