from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from core.embeddings.client import EmbeddingsClient
from core.embeddings.index import EmbeddingIndex
from core.llm.config import LLMConfig


def _chunk_document(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor — parses and chunks one PDF."""
    from services.ingest.parser import DoclingParser, make_source_document
    from services.ingest.chunker import DoclingChunker
    from services.ingest.pipeline import IngestionPipeline

    path, title, citation_url, delete_after = args
    if path.suffix.lower() in (".xlsx", ".xls"):  # Excel — bypasses Docling
        from services.ingest.excel import chunk_excel
        source, chunks = chunk_excel(path, source_url=citation_url)
    else:
        pipeline = IngestionPipeline(parser=DoclingParser(), chunker=DoclingChunker())
        source = make_source_document(path, source_url=citation_url)
        chunks = pipeline.run(source)
    if delete_after:
        path.unlink(missing_ok=True)
    return source, chunks, title, citation_url


@dataclass
class DownloadResult:
    docs_added: int
    chunks_added: int
    doc_details: list[tuple[str, str, int]]  # (title, citation_url, chunk_count)


@dataclass
class IngestResult:
    docs_added: int
    chunks_added: int
    index_size: int


class Ingest:
    def __init__(
        self,
        db_path: Path,
        *,
        llm_cfg: LLMConfig | None = None,
        embed_client: EmbeddingsClient | None = None,
        embeddings_path: Path | None = None,
        downloads_dir: Path = Path("downloads"),
        download_workers: int = 10,
        chunk_workers: int = 4,
    ) -> None:
        self._db_path = Path(db_path)
        self._llm_cfg = llm_cfg
        self._embed_client = embed_client
        self._embeddings_path = embeddings_path
        self._downloads_dir = Path(downloads_dir)
        self._download_workers = download_workers
        self._chunk_workers = chunk_workers

    def download(
        self,
        query: str,
        center: str | None = None,
        max_docs: int = 20,
        author: str | None = None,
        offset: int | None = None,
        on_download: "callable[[str], None] | None" = None,
        on_downloads_complete: "callable[[list[tuple[str, str]]], None] | None" = None,
        on_chunk: "callable[[str, int], None] | None" = None,
    ) -> DownloadResult:
        """Search NTRS, download PDFs, chunk, write documents and chunks to the graph.

        on_download(title) is called after each PDF is downloaded.
        on_downloads_complete([(title, url), ...]) is called after all PDFs are downloaded, before chunking.
        on_chunk(title, chunk_count) is called after each document is chunked.

        offset: if None, the stored next-offset for this query is used and auto-advanced after the run.
                If provided explicitly, that value is used and the stored offset is updated to offset+max_docs.
        """
        from services.ingest.ntrs import search, download_pdf, CENTER_CODES
        from services.ingest.parser import DoclingParser, make_source_document
        from services.ingest.chunker import DoclingChunker
        from services.ingest.pipeline import IngestionPipeline
        from core.graph import KuzuStore

        self._downloads_dir.mkdir(parents=True, exist_ok=True)

        center_code = CENTER_CODES.get(center.lower()) if center else None

        import hashlib
        total_chunks = 0
        doc_details = []
        query_key = f"{query}|{center or ''}|{author or ''}"
        with KuzuStore(self._db_path) as store:
            # Resolve offset: use stored value if not explicitly provided
            if offset is None:
                offset = store.get_ingest_offset(query_key)

            records = search(query, center=center_code, max_docs=max_docs, author=author, offset=offset)
            if not records:
                return DownloadResult(docs_added=0, chunks_added=0, doc_details=[])

            # Pre-filter: skip records whose citation_url is already in the graph,
            # avoiding unnecessary network downloads entirely.
            records_to_fetch = []
            for record in records:
                if store.document_exists_by_url(record.citation_url):
                    print(f"Skipping '{record.title[:60]}' (already in graph)")
                    if on_chunk:
                        on_chunk(record.title, 0)
                else:
                    records_to_fetch.append(record)

            # Advance stored offset by max_docs regardless of how many were new,
            # so the next run fetches the next page of NTRS results.
            store.set_ingest_offset(query_key, offset + max_docs)

            if not records_to_fetch:
                return DownloadResult(docs_added=0, chunks_added=0, doc_details=[])

            # Parallel download — only records not already in graph
            downloaded: list[tuple[Path, str, str]] = []
            with ThreadPoolExecutor(max_workers=min(self._download_workers, len(records_to_fetch))) as executor:
                futures = {executor.submit(download_pdf, record, self._downloads_dir): record for record in records_to_fetch}
                for future in as_completed(futures):
                    record = futures[future]
                    try:
                        dest = future.result()
                        downloaded.append((dest, record.title, record.citation_url))
                        if on_download:
                            on_download(record.title)
                    except Exception as e:
                        print(f"Failed to download '{record.title[:60]}...': {e}")

            if not downloaded:
                return DownloadResult(docs_added=0, chunks_added=0, doc_details=[])

            if on_downloads_complete:
                on_downloads_complete([(title, url) for _, title, url in downloaded])

            # Parallel chunking — ThreadPoolExecutor: threads share one copy of the Docling
            # model weights in the main process (no per-worker reload, no shutdown hang).
            # Belt-and-suspenders: also check by SHA256 hash in case the URL changed.
            to_chunk: list[tuple[Path, str, str]] = []
            for path, title, url in downloaded:
                doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
                if store.document_exists(doc_id):
                    print(f"Skipping '{title[:60]}' (already ingested)")
                    if on_chunk:
                        on_chunk(title, 0)
                else:
                    to_chunk.append((path, title, url))

            if to_chunk:
                with ThreadPoolExecutor(max_workers=min(self._chunk_workers, len(to_chunk))) as executor:
                    futures2 = {
                        executor.submit(_chunk_document, (path, title, url, True)): (path, title, url)
                        for path, title, url in to_chunk
                    }
                    for future in as_completed(futures2):
                        try:
                            source, chunks, title, citation_url = future.result()
                            store.write_document_with_chunks(source, title, chunks)
                            total_chunks += len(chunks)
                            doc_details.append((title, citation_url, len(chunks)))
                            if on_chunk:
                                on_chunk(title, len(chunks))
                        except Exception as e:
                            _, title, _ = futures2[future]
                            print(f"Failed to chunk '{title[:60]}...': {e}")

        return DownloadResult(docs_added=len(doc_details), chunks_added=total_chunks, doc_details=doc_details)

    def extract(
        self,
        dashboard_port: int = 8765,
        min_certainty: float = 0.5,
        on_extract_start: "callable[[int], None] | None" = None,
        on_chunk_extracted: "callable[[int, int, int], None] | None" = None,
    ) -> int:
        """Extract triples from all chunks in the graph. Returns total triples written.

        on_extract_start(total_chunks) is called after chunks are read, before the executor starts.
        on_chunk_extracted(chunks_done, total_chunks, triples_this_chunk) is called after each chunk.
        """
        from core.graph import KuzuStore
        from core.llm.worker import extract_chunk, init_worker
        from services.ingest.dashboard import advance_chunk, init_document, push_triple, start as start_dashboard

        if self._llm_cfg is None:
            raise ValueError("llm_cfg is required for extraction")

        with KuzuStore(self._db_path) as store:
            chunks = store.read_chunks(unextracted_only=True)
            if not chunks:
                return 0

            chunk_id_set = {c.id for c in chunks}
            start_dashboard(port=dashboard_port)
            for doc in store.read_document_chunk_counts(chunk_ids=chunk_id_set):
                init_document(doc.document_id, doc.source_url, doc.title, doc.total_chunks)
            print(f"Dashboard: http://127.0.0.1:{dashboard_port}")

            total_triples = 0
            chunks_done = 0
            total_chunks = len(chunks)
            if on_extract_start:
                on_extract_start(total_chunks)
            # Use ThreadPoolExecutor so the LLM backend (MLX/Metal) stays in the main process.
            # ProcessPoolExecutor with spawn hangs on macOS because Metal cannot initialise
            # in a spawned child process.
            with ThreadPoolExecutor(
                max_workers=self._llm_cfg.workers,
                initializer=init_worker,
                initargs=(self._llm_cfg,),
            ) as executor:
                future_to_chunk = {
                    executor.submit(extract_chunk, c.id, c.document_id, c.content): c
                    for c in chunks
                }
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    triples_this_chunk = 0
                    try:
                        _doc_id, triples = future.result()
                        for triple in triples:
                            if triple.certainty_score < min_certainty:
                                continue
                            store.write_triple(
                                subject=triple.subject.name,
                                subject_type=triple.subject.type,
                                relation=triple.relation,
                                object_=triple.object_.name,
                                object_type=triple.object_.type,
                                certainty_score=triple.certainty_score,
                                chunk_id=triple.chunk_id,
                            )
                            push_triple(
                                subject=triple.subject.name,
                                subject_type=triple.subject.type,
                                relation=triple.relation,
                                object_=triple.object_.name,
                                object_type=triple.object_.type,
                                certainty=triple.certainty_score,
                            )
                            total_triples += 1
                            triples_this_chunk += 1
                    except Exception as exc:
                        print(f"[extract] chunk {chunk.id} failed: {exc}")
                    finally:
                        chunks_done += 1
                        try:
                            store.mark_chunks_extracted([chunk.id])
                        except Exception as stamp_exc:
                            print(f"[extract] failed to stamp chunk {chunk.id}: {stamp_exc}")
                        advance_chunk(chunk.document_id)
                        if on_chunk_extracted:
                            on_chunk_extracted(chunks_done, total_chunks, triples_this_chunk)

        return total_triples

    def build_index(self) -> EmbeddingIndex:
        """Incrementally update (or cold-build) the embedding index.

        If an embeddings file already exists, only chunks missing from that
        index are embedded; the rest are reused as-is.  Prints a one-line
        stats summary: total / already_indexed / newly embedded.
        """
        from core.embeddings.builder import update_index
        from core.embeddings.index import EmbeddingIndex
        from core.graph import KuzuStore

        if self._embed_client is None:
            raise ValueError("embed_client is required for building the index")

        existing: EmbeddingIndex | None = None
        if self._embeddings_path and Path(self._embeddings_path).exists():
            existing = EmbeddingIndex.load(self._embeddings_path)

        with KuzuStore(self._db_path) as store:
            index, stats = update_index(store, self._embed_client, existing)

        print(
            f"Embedding index: {stats['total']} total  |  "
            f"{stats['already_indexed']} already indexed  |  "
            f"{stats['new_embedded']} newly embedded"
        )
        if self._embeddings_path:
            index.save(self._embeddings_path)
        return index

    def ingest_local_files(
        self,
        confirmed_docs: list[tuple[Path, dict]],
        dashboard_port: int = 8765,
        on_chunk: "callable[[str, int], None] | None" = None,
        on_extract_start: "callable[[int], None] | None" = None,
        on_chunk_extracted: "callable[[int, int, int], None] | None" = None,
    ) -> "tuple[IngestResult, EmbeddingIndex]":
        """Ingest a list of local PDFs with pre-confirmed metadata.

        confirmed_docs: list of (pdf_path, meta_dict) where meta_dict has at least 'title'.
        """
        from core.graph import KuzuStore

        total_chunks = 0
        doc_details = []

        with KuzuStore(self._db_path) as store:
            with ThreadPoolExecutor(max_workers=min(self._chunk_workers, max(1, len(confirmed_docs)))) as executor:
                futures = {
                    executor.submit(
                        _chunk_document,
                        (path, meta.get("title") or path.stem, f"file://{path.absolute()}", False),
                    ): (path, meta)
                    for path, meta in confirmed_docs
                }
                for future in as_completed(futures):
                    path, meta = futures[future]
                    try:
                        source, chunks, title, citation_url = future.result()
                        store.write_document_with_chunks(source, title, chunks)
                        total_chunks += len(chunks)
                        doc_details.append((title, citation_url, len(chunks)))
                        if on_chunk:
                            on_chunk(title, len(chunks))
                    except Exception as e:
                        print(f"Failed to chunk '{path.name}': {e}")

        if total_chunks > 0:
            self.extract(
                dashboard_port=dashboard_port,
                on_extract_start=on_extract_start,
                on_chunk_extracted=on_chunk_extracted,
            )
        index = self.build_index()
        result = IngestResult(
            docs_added=len(doc_details),
            chunks_added=total_chunks,
            index_size=len(index),
        )
        return result, index

    def run(
        self,
        query: str,
        *,
        center: str | None = None,
        max_docs: int = 20,
        author: str | None = None,
        offset: int | None = None,
        dashboard_port: int = 8765,
        on_download=None,
        on_downloads_complete=None,
        on_chunk=None,
        on_extract_start=None,
        on_chunk_extracted=None,
    ) -> tuple[IngestResult, EmbeddingIndex]:
        """End-to-end pipeline: download -> extract -> build index."""
        dl = self.download(
            query, center=center, max_docs=max_docs, author=author, offset=offset,
            on_download=on_download,
            on_downloads_complete=on_downloads_complete,
            on_chunk=on_chunk,
        )
        if dl.chunks_added > 0:
            self.extract(dashboard_port=dashboard_port, on_extract_start=on_extract_start, on_chunk_extracted=on_chunk_extracted)
        index = self.build_index()
        return IngestResult(docs_added=dl.docs_added, chunks_added=dl.chunks_added, index_size=len(index)), index
