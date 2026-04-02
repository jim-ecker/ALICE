from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(help="ALICE — Automated Literature Ingestion and Concept Extraction")


def download_and_display(ingest, query: str, *, location: str | None, max_docs: int) -> None:
    """Run Ingest.download() with live progress bars, then print the two tables."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    typer.echo(f"Searching NTRS for '{query}'" + (f" at {location}" if location else "") + "...")

    dl_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    dl_progress.start()
    dl_task = dl_progress.add_task("Searching NTRS...", total=max(1, max_docs))

    chunk_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    chunk_task_holder: list = []
    failures: list[tuple[str, str]] = []
    records_found: list[int | None] = [None]

    def on_search_complete(total: int) -> None:
        records_found[0] = total
        dl_progress.update(
            dl_task,
            description="Downloading PDFs..." if total else "No downloadable PDFs found",
            total=max(1, total),
        )

    def on_download(title: str) -> None:
        dl_progress.update(dl_task, description=f"Downloading PDFs... {title[:48]}")
        dl_progress.advance(dl_task)

    def on_download_failed(title: str, error: str) -> None:
        failures.append((title, error))

    def on_downloads_complete(records: list[tuple[str, str]]) -> None:
        dl_progress.stop()
        t = Table(title="Downloaded Documents")
        t.add_column("Title", ratio=1)
        t.add_column("URL", style="blue", no_wrap=True)
        for title, url in records:
            t.add_row(title, url)
        rprint(t)
        chunk_progress.start()
        chunk_task_holder.append(chunk_progress.add_task("Chunking...", total=len(records)))

    def on_chunk(title: str, chunk_count: int) -> None:
        if chunk_task_holder:
            chunk_progress.advance(chunk_task_holder[0])

    result = ingest.download(
        query, center=location, max_docs=max_docs,
        on_search_complete=on_search_complete,
        on_download=on_download,
        on_download_failed=on_download_failed,
        on_downloads_complete=on_downloads_complete,
        on_chunk=on_chunk,
    )

    if chunk_task_holder:
        chunk_progress.stop()

    if not result.doc_details:
        if failures:
            typer.echo(
                f"Found {records_found[0] or len(failures)} candidate documents, but all downloads failed."
            )
            for title, error in failures:
                typer.echo(f"  - {title}: {error}")
        else:
            typer.echo("No downloadable documents found.")
        return

    chunk_table = Table(title="Chunks per Document")
    chunk_table.add_column("Title", ratio=1)
    chunk_table.add_column("URL", style="blue", no_wrap=True)
    chunk_table.add_column("Chunks", justify="right")
    for title, url, chunk_count in result.doc_details:
        chunk_table.add_row(title, url, str(chunk_count))
    rprint(chunk_table)
    typer.echo(f"Produced {result.chunks_added} chunks across {result.docs_added} documents.")


def extract_with_progress(
    ingest,
    *,
    dashboard_port: int = 8765,
    min_ingest_confidence: float = 0.6,
) -> int:
    """Run Ingest.extract() with a live terminal progress bar. Returns total triples."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Waiting...", total=None)

        def on_extract_start(total_chunks: int) -> None:
            progress.update(task, description="Loading model...", total=total_chunks)

        def on_chunk_extracted(chunks_done: int, total_chunks: int, triples_this_chunk: int) -> None:
            progress.update(task, completed=chunks_done, total=total_chunks,
                            description=f"Extracting triples... (+{triples_this_chunk})")

        return ingest.extract(
            dashboard_port=dashboard_port,
            min_ingest_confidence=min_ingest_confidence,
            on_extract_start=on_extract_start,
            on_chunk_extracted=on_chunk_extracted,
        )


@app.command()
def download(
    query: str = typer.Argument(..., help="Search query (e.g. 'robotics')"),
    location: str = typer.Option(None, help="NASA center name (e.g. 'langley', 'goddard', 'ames')"),
    max_docs: int = typer.Option(20, help="Maximum number of documents to download"),
    output_dir: Path = typer.Option(Path("downloads"), help="Directory to save downloaded PDFs"),
    workers: int = typer.Option(10, help="Number of parallel download workers"),
    chunk_workers: int = typer.Option(4, help="Number of parallel Docling chunking workers"),
    db_path: Path = typer.Option(Path("alice.db"), help="Path to the Kuzu graph database"),
):
    """Download research papers from NASA NTRS."""
    from services.ingest.ntrs import CENTER_CODES
    from services.ingest.service import Ingest

    if location and location.lower() not in CENTER_CODES:
        typer.echo(f"Unknown location '{location}'. Valid options: {', '.join(CENTER_CODES)}", err=True)
        raise typer.Exit(1)

    ingest = Ingest(
        db_path=db_path,
        downloads_dir=output_dir,
        download_workers=workers,
        chunk_workers=chunk_workers,
    )
    download_and_display(ingest, query, location=location, max_docs=max_docs)


@app.command()
def reset(
    db_path: Path = typer.Option(Path("alice.db"), help="Path to the Kuzu graph database"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete the database, returning the system to a clean state."""
    if not yes:
        typer.confirm(f"This will permanently delete {db_path}. Continue?", abort=True)
    if db_path.exists():
        db_path.unlink()
        typer.echo(f"Deleted {db_path}.")
    else:
        typer.echo(f"{db_path} does not exist.")


@app.command()
def reset_extraction(
    db_path: Path = typer.Option(Path("alice.db"), help="Path to the Kuzu graph database"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Remove all extracted entities and relations from the graph, keeping documents and chunks."""
    from core.graph import KuzuStore

    if not yes:
        typer.confirm("This will delete all entities and relations from the graph. Continue?", abort=True)

    store = KuzuStore(db_path)
    store.clear_extraction()
    typer.echo("Extraction state cleared. Documents and chunks are intact.")


@app.command()
def reextract_document(
    title: str = typer.Argument(..., help="Substring of document title to re-extract"),
    db_path: Path = typer.Option(Path("alice.db"), help="Path to the Kuzu graph database"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Clear extracted triples for a document and mark it for re-extraction.

    After running this command, run 'ingest extract' to re-extract the cleared document.
    Other already-extracted documents are unaffected.
    """
    from core.graph import KuzuStore

    store = KuzuStore(db_path)

    r = store._conn.execute(
        "MATCH (d:Document) WHERE d.title CONTAINS $title RETURN d.id, d.title",
        parameters={"title": title},
    )
    matches = []
    while r.has_next():
        row = r.get_next()
        matches.append((row[0], row[1]))

    if not matches:
        typer.echo(f"No documents found matching '{title}'.")
        raise typer.Exit(1)

    table = Table(title="Matching Documents")
    table.add_column("ID")
    table.add_column("Title", ratio=1)
    for doc_id, doc_title in matches:
        table.add_row(doc_id, doc_title)
    rprint(table)

    if not yes:
        typer.confirm(
            f"Clear triples for {len(matches)} document(s) and mark for re-extraction?",
            abort=True,
        )

    for doc_id, doc_title in matches:
        store.clear_document_extraction(doc_id)
        typer.echo(f"Cleared triples for '{doc_title}'.")

    typer.echo("\nDone. Now run 'ingest extract --db-path <path>' to re-extract.")


@app.command()
def extract(
    db_path: Path = typer.Option(Path("alice.db"), help="Path to the Kuzu graph database"),
    model: str | None = typer.Option(None, help="LLM model name (overrides alice.toml)"),
    backend: str | None = typer.Option(None, help="mlx | vllm | openai-compatible (overrides alice.toml)"),
    base_url: str | None = typer.Option(None, help="Base URL for OpenAI-compatible endpoint"),
    api_key: str | None = typer.Option(None, help="API key for OpenAI-compatible endpoint"),
    min_ingest_confidence: float | None = typer.Option(
        None,
        "--min-ingest-confidence",
        help="Minimum grounded ingest confidence for triples to be stored",
    ),
    min_certainty: float | None = typer.Option(
        None,
        "--min-certainty",
        help="Deprecated alias for --min-ingest-confidence",
    ),
    dashboard_port: int = typer.Option(8765, help="Port for the progress dashboard"),
    llm_workers: int | None = typer.Option(None, help="Parallel LLM worker processes (overrides alice.toml)"),
):
    """Extract entities and relations from chunks and write triples to the graph."""
    from core.llm.config import resolve_config
    from services.ingest.config import load_ingest_config
    from services.ingest.service import Ingest

    cfg = resolve_config(
        cli_model=model,
        cli_backend=backend,
        cli_base_url=base_url,
        cli_api_key=api_key,
        cli_workers=llm_workers,
        start_dir=db_path.parent,
    )
    ingest_cfg = load_ingest_config(db_path.parent)
    threshold = ingest_cfg.min_ingest_confidence
    if min_certainty is not None:
        typer.echo("Warning: --min-certainty is deprecated; use --min-ingest-confidence instead.")
        threshold = min_certainty
    if min_ingest_confidence is not None:
        threshold = min_ingest_confidence
    typer.echo(f"Backend: {cfg.backend}, model: {cfg.model}, workers: {cfg.workers}")
    typer.echo(f"Min ingest confidence: {threshold:.2f}")

    ingest = Ingest(db_path=db_path, llm_cfg=cfg)
    total_triples = extract_with_progress(
        ingest,
        dashboard_port=dashboard_port,
        min_ingest_confidence=threshold,
    )
    typer.echo(f"Wrote {total_triples} triples.")


if __name__ == "__main__":
    app()
