from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
from pathlib import Path

from core.embeddings.client import EmbeddingsClient
from core.embeddings.index import EmbeddingIndex
from core.llm.config import LLMConfig

LOGGER = logging.getLogger(__name__)


def _get_visible_gpu_ids() -> list[int]:
    """Parse CUDA_VISIBLE_DEVICES into a list of physical GPU IDs.

    Returns an empty list when the variable is unset or CUDA is unavailable.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw or raw in ("NoDeviceFiles", "-1"):
        return []
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]
    except ValueError:
        return []


def _get_usable_gpu_ids(min_free_gb: float = 3.0) -> list[int]:
    """Return physical GPU IDs from CUDA_VISIBLE_DEVICES that have >= min_free_gb free.

    Uses nvidia-smi so no CUDA context is created in the main process.
    Falls back to all visible IDs if nvidia-smi is unavailable.
    """
    import subprocess

    gpu_ids = _get_visible_gpu_ids()
    if not gpu_ids:
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return gpu_ids
        free_mib: dict[int, int] = {}
        for line in result.stdout.strip().splitlines():
            idx_str, free_str = line.split(",")
            free_mib[int(idx_str.strip())] = int(free_str.strip())
        min_mib = int(min_free_gb * 1024)
        usable = [gid for gid in gpu_ids if free_mib.get(gid, 0) >= min_mib]
        if not usable:
            print(
                f"[ingest] Warning: no GPU among {gpu_ids} has >= {min_free_gb} GiB free "
                f"(free MiB per GPU: {free_mib}). Falling back to all visible GPUs."
            )
            return gpu_ids
        skipped = [gid for gid in gpu_ids if gid not in usable]
        if skipped:
            print(f"[ingest] Skipping GPU(s) {skipped}: insufficient free memory for docling (< {min_free_gb} GiB).")
        return usable
    except FileNotFoundError:
        return gpu_ids  # nvidia-smi not available
    except Exception:
        return gpu_ids


# Per-process GPU ID set by the worker initializer.
_worker_gpu_id: int | None = None


def _init_chunk_worker(gpu_queue: multiprocessing.Queue) -> None:
    """Claim a GPU ID from the queue and bind this process to it via CUDA_VISIBLE_DEVICES."""
    global _worker_gpu_id
    try:
        _worker_gpu_id = gpu_queue.get_nowait()
    except Exception:
        _worker_gpu_id = None
    if _worker_gpu_id is not None:
        # Remap "CUDA device 0" inside this process to the assigned physical GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)


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


def _make_chunk_executor(num_workers: int) -> tuple[ProcessPoolExecutor, multiprocessing.Queue]:
    """Build a ProcessPoolExecutor whose workers are each pinned to a distinct GPU.

    Uses 'spawn' to give every worker a clean CUDA context.  GPU IDs are drawn
    from CUDA_VISIBLE_DEVICES; if none are available workers run on CPU/default.
    Returns (executor, gpu_queue) — caller must enter/exit the executor as a
    context manager.
    """
    ctx = multiprocessing.get_context("spawn")
    gpu_ids = _get_usable_gpu_ids()
    gpu_queue: multiprocessing.Queue = ctx.Queue()
    for i in range(num_workers):
        if gpu_ids:
            gpu_queue.put(gpu_ids[i % len(gpu_ids)])
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_chunk_worker,
        initargs=(gpu_queue,),
    )
    return executor, gpu_queue


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
        on_search_complete: "callable[[int], None] | None" = None,
        on_download: "callable[[str], None] | None" = None,
        on_download_failed: "callable[[str, str], None] | None" = None,
        on_downloads_complete: "callable[[list[tuple[str, str]]], None] | None" = None,
        on_chunk: "callable[[str, int], None] | None" = None,
    ) -> DownloadResult:
        """Search NTRS, download PDFs, chunk, write documents and chunks to the graph.

        on_download(title) is called after each PDF is downloaded.
        on_downloads_complete([(title, url), ...]) is called after all PDFs are downloaded, before chunking.
        on_chunk(title, chunk_count) is called after each document is chunked.

        offset: if None, the stored next-offset for this query is used and auto-advanced after the run.
                If provided explicitly, that value is used and the stored offset is updated to the
                furthest raw NTRS offset scanned during the run.
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

            # Fetch records from NTRS, skipping any already in the graph, and
            # keep fetching additional pages until we have max_docs new records
            # or NTRS is exhausted.
            records_to_fetch: list = []
            seen_records: list = []
            current_offset = offset
            record_status: dict[str, str] = {}

            def persist_safe_offset() -> int:
                safe_offset = offset
                for record in seen_records:
                    status = record_status.get(record.citation_url, "pending")
                    if status in {"already_in_graph", "committed", "already_ingested"}:
                        safe_offset = max(safe_offset, record.resume_offset)
                    else:
                        break
                store.set_ingest_offset(query_key, safe_offset)
                return safe_offset

            try:
                while len(records_to_fetch) < max_docs:
                    batch, next_offset = search(
                        query, center=center_code,
                        max_docs=max_docs - len(records_to_fetch),
                        author=author,
                        offset=current_offset,
                    )
                    if not batch:
                        break
                    for record in batch:
                        seen_records.append(record)
                        if store.document_exists_by_url(record.citation_url):
                            print(f"Skipping '{record.title[:60]}' (already in graph)")
                            record_status[record.citation_url] = "already_in_graph"
                        else:
                            records_to_fetch.append(record)
                            record_status[record.citation_url] = "pending"
                            if len(records_to_fetch) >= max_docs:
                                break
                    if next_offset <= current_offset:
                        break  # NTRS exhausted, no forward progress
                    current_offset = next_offset

                if on_search_complete:
                    on_search_complete(len(records_to_fetch))
                LOGGER.info(
                    "Prepared %d NTRS records for download for query=%r author=%r center=%r",
                    len(records_to_fetch),
                    query,
                    author,
                    center,
                )

                if not records_to_fetch:
                    return DownloadResult(docs_added=0, chunks_added=0, doc_details=[])

                # Parallel download — only records not already in graph
                downloaded: list[tuple[Path, object]] = []
                download_executor = ThreadPoolExecutor(max_workers=min(self._download_workers, len(records_to_fetch)))
                try:
                    futures = {
                        download_executor.submit(download_pdf, record, self._downloads_dir): record
                        for record in records_to_fetch
                    }
                    for future in as_completed(futures):
                        record = futures[future]
                        try:
                            dest = future.result()
                            downloaded.append((dest, record))
                            record_status[record.citation_url] = "downloaded"
                            if on_download:
                                on_download(record.title)
                        except Exception as e:
                            record_status[record.citation_url] = "download_failed"
                            LOGGER.warning("Failed to download %r from %s: %s", record.title, record.citation_url, e)
                            if on_download_failed:
                                on_download_failed(record.title, str(e))
                            print(f"Failed to download '{record.title[:60]}...': {e}")
                except BaseException:
                    download_executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    download_executor.shutdown(wait=True)

                if not downloaded:
                    return DownloadResult(docs_added=0, chunks_added=0, doc_details=[])

                if on_downloads_complete:
                    on_downloads_complete([(record.title, record.citation_url) for _, record in downloaded])

                # Belt-and-suspenders: also check by SHA256 hash in case the URL changed.
                to_chunk: list[tuple[Path, object]] = []
                for path, record in downloaded:
                    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
                    if store.document_exists(doc_id):
                        print(f"Skipping '{record.title[:60]}' (already ingested)")
                        record_status[record.citation_url] = "already_ingested"
                        if on_chunk:
                            on_chunk(record.title, 0)
                    else:
                        to_chunk.append((path, record))

                if to_chunk:
                    num_workers = min(self._chunk_workers, len(to_chunk))
                    chunk_executor, _ = _make_chunk_executor(num_workers)
                    try:
                        futures2 = {
                            chunk_executor.submit(
                                _chunk_document,
                                (path, record.title, record.citation_url, True),
                            ): (path, record)
                            for path, record in to_chunk
                        }
                        for future in as_completed(futures2):
                            _, record = futures2[future]
                            try:
                                source, chunks, title, citation_url = future.result()
                                store.write_document_with_chunks(source, title, chunks)
                                total_chunks += len(chunks)
                                doc_details.append((title, citation_url, len(chunks)))
                                record_status[record.citation_url] = "committed"
                                if on_chunk:
                                    on_chunk(title, len(chunks))
                            except Exception as e:
                                record_status[record.citation_url] = "chunk_failed"
                                print(f"Failed to chunk '{record.title[:60]}...': {e}")
                    except BaseException:
                        chunk_executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    else:
                        chunk_executor.shutdown(wait=True)
            finally:
                persist_safe_offset()

        return DownloadResult(docs_added=len(doc_details), chunks_added=total_chunks, doc_details=doc_details)

    def ingest_one(
        self,
        record: "NTRSRecord",
        dashboard_port: int = 8765,
        on_download=None,
        on_chunk=None,
        on_extract_start=None,
        on_chunk_extracted=None,
    ) -> "tuple[IngestResult, EmbeddingIndex]":
        """Ingest a single pre-fetched NTRS record. No offset tracking."""
        from services.ingest.ntrs import download_pdf
        from core.graph import KuzuStore
        import hashlib

        self._downloads_dir.mkdir(parents=True, exist_ok=True)
        with KuzuStore(self._db_path) as store:
            if store.document_exists_by_url(record.citation_url):
                print(f"'{record.title[:60]}' is already in the graph.")
                index = self.build_index()
                return IngestResult(docs_added=0, chunks_added=0, index_size=len(index)), index

            dest = download_pdf(record, self._downloads_dir)
            if on_download:
                on_download(record.title)

            doc_id = hashlib.sha256(dest.read_bytes()).hexdigest()
            if store.document_exists(doc_id):
                print(f"'{record.title[:60]}' already ingested (content hash match).")
                index = self.build_index()
                return IngestResult(docs_added=0, chunks_added=0, index_size=len(index)), index

            executor, _ = _make_chunk_executor(1)
            with executor:
                source, chunks, title, citation_url = executor.submit(
                    _chunk_document,
                    (dest, record.title, record.citation_url, True),
                ).result()
            store.write_document_with_chunks(source, title, chunks)
            if on_chunk:
                on_chunk(title, len(chunks))

        if chunks:
            self.extract(
                dashboard_port=dashboard_port,
                on_extract_start=on_extract_start,
                on_chunk_extracted=on_chunk_extracted,
            )
        index = self.build_index()
        return IngestResult(docs_added=1, chunks_added=len(chunks), index_size=len(index)), index

    def extract(
        self,
        dashboard_port: int = 8765,
        min_ingest_confidence: float = 0.6,
        min_certainty: float | None = None,
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
        if min_certainty is not None:
            min_ingest_confidence = min_certainty

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
            executor = ThreadPoolExecutor(
                max_workers=self._llm_cfg.workers,
                initializer=init_worker,
                initargs=(self._llm_cfg,),
            )
            try:
                future_to_chunk = {
                    executor.submit(extract_chunk, c.id, c.document_id, c.content): c
                    for c in chunks
                }
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    triples_this_chunk = 0
                    try:
                        _doc_id, triples = future.result()
                        kept_triples = store.write_chunk_triples(
                            chunk.id,
                            triples,
                            min_ingest_confidence=min_ingest_confidence,
                        )
                        for triple in kept_triples:
                            push_triple(
                                subject=triple.subject.name,
                                subject_type=triple.subject.type,
                                relation=triple.relation,
                                object_=triple.object_.name,
                                object_type=triple.object_.type,
                                ingest_confidence=triple.certainty_score,
                                extractor_certainty=triple.raw_certainty_score,
                                evidence_text=triple.evidence_text,
                            )
                            total_triples += 1
                            triples_this_chunk += 1
                    except Exception as exc:
                        print(f"[extract] chunk {chunk.id} failed: {exc}")
                    finally:
                        chunks_done += 1
                        advance_chunk(chunk.document_id)
                        if on_chunk_extracted:
                            on_chunk_extracted(chunks_done, total_chunks, triples_this_chunk)
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

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
            num_workers = min(self._chunk_workers, max(1, len(confirmed_docs)))
            executor, _ = _make_chunk_executor(num_workers)
            with executor:
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
        on_search_complete=None,
        on_download=None,
        on_download_failed=None,
        on_downloads_complete=None,
        on_chunk=None,
        on_extract_start=None,
        on_chunk_extracted=None,
    ) -> tuple[IngestResult, EmbeddingIndex]:
        """End-to-end pipeline: download -> extract -> build index."""
        from core.graph import KuzuStore

        dl = self.download(
            query, center=center, max_docs=max_docs, author=author, offset=offset,
            on_search_complete=on_search_complete,
            on_download=on_download,
            on_download_failed=on_download_failed,
            on_downloads_complete=on_downloads_complete,
            on_chunk=on_chunk,
        )
        with KuzuStore(self._db_path) as store:
            pending_chunks = store.count_unextracted_chunks()
        if pending_chunks > 0:
            self.extract(dashboard_port=dashboard_port, on_extract_start=on_extract_start, on_chunk_extracted=on_chunk_extracted)
        index = self.build_index()
        return IngestResult(docs_added=dl.docs_added, chunks_added=dl.chunks_added, index_size=len(index)), index
