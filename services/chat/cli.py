from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.table import Table

chat_app = typer.Typer(help="ALICE chat service")


@chat_app.command()
def serve(
    db_path: Path = typer.Option(Path("chat.db"), "--db-path", help="Path to the chat KuzuDB database"),
    host: str = typer.Option("127.0.0.1", help="Host to bind the server"),
    port: int = typer.Option(8766, help="Port to bind the server"),
    model: Optional[str] = typer.Option(None, help="LLM model name (overrides alice.toml)"),
    backend: Optional[str] = typer.Option(None, help="mlx | openai-compatible (overrides alice.toml)"),
    base_url: Optional[str] = typer.Option(None, help="Base URL for OpenAI-compatible endpoint"),
    api_key: Optional[str] = typer.Option(None, help="API key for OpenAI-compatible endpoint"),
) -> None:
    """Start the ALICE chat server."""
    from services.chat.server import start

    start(
        db_path,
        host=host,
        port=port,
        model=model,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
    )


@chat_app.command()
def ingest(
    query: str = typer.Argument(..., help="Search query for NASA NTRS (e.g. 'robotics')"),
    db_path: Path = typer.Option(Path("chat.db"), "--db-path", help="Path to the chat KuzuDB database"),
    max_docs: int = typer.Option(20, help="Maximum number of documents to ingest"),
    location: Optional[str] = typer.Option(None, help="NASA center name (e.g. 'langley')"),
    model: Optional[str] = typer.Option(None, help="LLM model name (overrides alice.toml)"),
    backend: Optional[str] = typer.Option(None, help="mlx | openai-compatible (overrides alice.toml)"),
    base_url: Optional[str] = typer.Option(None, help="Base URL for OpenAI-compatible endpoint"),
    api_key: Optional[str] = typer.Option(None, help="API key for OpenAI-compatible endpoint"),
    workers: Optional[int] = typer.Option(None, help="Parallel LLM worker processes (overrides alice.toml)"),
    dashboard_port: int = typer.Option(8765, help="Port for the extraction progress dashboard"),
) -> None:
    """Download, chunk, extract triples, and embed documents into the chat knowledge graph."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
    from core.llm.config import resolve_config
    from services.chat.service import Chat

    llm_cfg = resolve_config(
        cli_model=model,
        cli_backend=backend,
        cli_base_url=base_url,
        cli_api_key=api_key,
        cli_workers=workers,
    )
    chat = Chat(db_path=db_path, llm_cfg=llm_cfg)

    typer.echo(f"Searching NTRS for '{query}'" + (f" at {location}" if location else "") + "...")

    dl_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    dl_progress.start()
    dl_task = dl_progress.add_task("Downloading PDFs...", total=max_docs)

    chunk_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    model_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"))
    extract_progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn())

    state: dict = {
        "chunk_task": None,
        "extract_task": None,
        "url_map": {},
        "chunk_details": [],
        "chunk_progress_stopped": False,
        "model_progress_started": False,
        "model_progress_stopped": False,
        "extract_progress_started": False,
    }

    def on_download(title: str) -> None:
        dl_progress.advance(dl_task)

    def on_downloads_complete(records: list[tuple[str, str]]) -> None:
        dl_progress.stop()
        for t, u in records:
            state["url_map"][t] = u
        tbl = Table(title="Downloaded Documents")
        tbl.add_column("Title", ratio=1)
        tbl.add_column("URL", style="blue", no_wrap=True)
        for t, u in records:
            tbl.add_row(t, u)
        rprint(tbl)
        chunk_progress.start()
        state["chunk_task"] = chunk_progress.add_task("Chunking...", total=len(records))

    def on_chunk(title: str, chunk_count: int) -> None:
        state["chunk_details"].append((title, state["url_map"].get(title, ""), chunk_count))
        if state["chunk_task"] is not None:
            chunk_progress.advance(state["chunk_task"])

    def on_extract_start(total_chunks: int) -> None:
        if not state["chunk_progress_stopped"]:
            chunk_progress.stop()
            state["chunk_progress_stopped"] = True
            tbl = Table(title="Chunks per Document")
            tbl.add_column("Title", ratio=1)
            tbl.add_column("URL", style="blue", no_wrap=True)
            tbl.add_column("Chunks", justify="right")
            for t, u, c in state["chunk_details"]:
                tbl.add_row(t, u, str(c))
            rprint(tbl)
        model_progress.start()
        model_progress.add_task("Loading model...", total=None)
        state["model_progress_started"] = True

    def on_chunk_extracted(chunks_done: int, total_chunks: int, triples_this_chunk: int) -> None:
        if state["model_progress_started"] and not state["model_progress_stopped"]:
            model_progress.stop()
            state["model_progress_stopped"] = True
        if not state["chunk_progress_stopped"]:
            chunk_progress.stop()
            state["chunk_progress_stopped"] = True
            tbl = Table(title="Chunks per Document")
            tbl.add_column("Title", ratio=1)
            tbl.add_column("URL", style="blue", no_wrap=True)
            tbl.add_column("Chunks", justify="right")
            for t, u, c in state["chunk_details"]:
                tbl.add_row(t, u, str(c))
            rprint(tbl)
        if not state["extract_progress_started"]:
            extract_progress.start()
            state["extract_task"] = extract_progress.add_task("Extracting triples...", total=total_chunks)
            state["extract_progress_started"] = True
        extract_progress.update(
            state["extract_task"], completed=chunks_done, total=total_chunks,
            description=f"Extracting triples... (+{triples_this_chunk})",
        )

    result = chat.ingest(
        query,
        center=location,
        max_docs=max_docs,
        dashboard_port=dashboard_port,
        on_download=on_download,
        on_downloads_complete=on_downloads_complete,
        on_chunk=on_chunk,
        on_extract_start=on_extract_start,
        on_chunk_extracted=on_chunk_extracted,
    )

    if not state["chunk_progress_stopped"] and state["chunk_task"] is not None:
        chunk_progress.stop()
    if state["model_progress_started"] and not state["model_progress_stopped"]:
        model_progress.stop()
    if state["extract_progress_started"]:
        extract_progress.stop()

    if result.docs_added == 0:
        typer.echo("No downloadable documents found.")
        return

    typer.echo(f"Done: {result.docs_added} docs, {result.chunks_added} chunks, index: {result.index_size}")


@chat_app.command()
def backup(
    backup_dir: str = typer.Option("backups", help="Directory to store backups"),
) -> None:
    """Copy the chat database and embedding index to a timestamped backup. Run with server stopped."""
    import shutil
    from datetime import datetime
    from services.chat.config import load_chat_config

    chat_cfg, _, _, _ = load_chat_config()
    sources = [p for p in (chat_cfg.db_path, chat_cfg.embeddings_path) if p.exists()]
    if not sources:
        typer.echo("Nothing to back up.")
        raise typer.Exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = chat_cfg.db_path.parent / backup_dir / stamp
    dest.mkdir(parents=True, exist_ok=True)

    for src in sources:
        shutil.copy2(src, dest / src.name)
        typer.echo(f"Backed up {src} → {dest / src.name}")

    typer.echo(f"Backup complete: {dest}")


@chat_app.command()
def restore(
    backup_path: str = typer.Argument(..., help="Path to the backup directory (e.g. backups/20240320_143022)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Restore the chat database and embedding index from a backup. Run with server stopped."""
    import shutil
    from pathlib import Path
    from services.chat.config import load_chat_config

    chat_cfg, _, _, _ = load_chat_config()
    src_dir = Path(backup_path)
    if not src_dir.exists():
        typer.echo(f"Backup not found: {src_dir}", err=True)
        raise typer.Exit(1)

    targets = list(src_dir.iterdir())
    if not targets:
        typer.echo("Backup directory is empty.", err=True)
        raise typer.Exit(1)

    if not yes:
        typer.confirm(f"This will overwrite {chat_cfg.db_path} and {chat_cfg.embeddings_path}. Continue?", abort=True)

    for src in targets:
        dest = chat_cfg.db_path.parent / src.name
        shutil.copy2(src, dest)
        typer.echo(f"Restored {src} → {dest}")

    typer.echo("Restore complete.")


@chat_app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete the chat database and embedding index, returning to a clean state."""
    from services.chat.config import load_chat_config

    chat_cfg, _, _, _ = load_chat_config()
    targets = [p for p in (chat_cfg.db_path, chat_cfg.embeddings_path) if p.exists()]
    if not targets:
        typer.echo("Nothing to delete.")
        return

    if not yes:
        files = ", ".join(str(p) for p in targets)
        typer.confirm(f"This will permanently delete: {files}. Continue?", abort=True)

    for p in targets:
        p.unlink()
        typer.echo(f"Deleted {p}.")


@chat_app.command()
def skip_unextracted(
    db_path: Path = typer.Option(Path("chat.db"), "--db-path", help="Path to the chat KuzuDB database"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Stamp all unextracted chunks as skipped so they are excluded from future extract runs."""
    from core.graph import KuzuStore

    if not yes:
        typer.confirm("This will mark all unextracted chunks as skipped. Continue?", abort=True)

    store = KuzuStore(db_path)
    count = store.skip_unextracted_chunks()
    typer.echo(f"Skipped {count} unextracted chunk(s).")


# ── Root app combining ingest + chat + experts ────────────────────────────────

root_app = typer.Typer(help="ALICE — Knowledge Graph CLI")

from services.ingest import cli as ingest_cli  # noqa: E402
from services.experts import cli as experts_cli  # noqa: E402

root_app.add_typer(ingest_cli.app, name="ingest")
root_app.add_typer(chat_app, name="chat")
root_app.add_typer(experts_cli.app, name="experts")
