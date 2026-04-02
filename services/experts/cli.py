from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Manage ALICE virtual experts", invoke_without_command=True)
LOGGER = logging.getLogger(__name__)


def _resolve_llm_cfg(start_dir: Path):
    from core.llm.config import resolve_config

    return resolve_config(
        cli_model=None,
        cli_backend=None,
        cli_base_url=None,
        cli_api_key=None,
        cli_workers=None,
        start_dir=start_dir,
    )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Launch the interactive ALICE Expert Manager TUI."""
    if ctx.invoked_subcommand is None:
        _interactive_tui()


def _interactive_tui() -> None:
    import questionary
    from services.chat.config import load_chat_config
    from services.experts.manager import ExpertRegistry

    chat_cfg, embed_cfg, _, llm_cfg = load_chat_config()
    experts_dir = chat_cfg.experts_dir
    registry = ExpertRegistry(experts_dir)

    while True:
        rprint(Panel("[bold blue]ALICE Expert Manager[/bold blue]", expand=False))
        choice = questionary.select(
            "What would you like to do?",
            choices=["List experts", "Create expert", "Manage expert", "Exit"],
        ).ask()

        if choice is None or choice == "Exit":
            break
        elif choice == "List experts":
            _list_experts(registry)
        elif choice == "Create expert":
            _create_expert(registry, experts_dir, embed_cfg, llm_cfg)
        elif choice == "Manage expert":
            _manage_expert_menu(registry, experts_dir, embed_cfg, llm_cfg)


def _list_experts(registry) -> None:
    experts = registry.list()
    if not experts:
        rprint("[yellow]No experts found.[/yellow]")
        return
    tbl = Table(title="Virtual Experts")
    tbl.add_column("Name")
    tbl.add_column("Slug")
    tbl.add_column("Max Docs", justify="right")
    tbl.add_column("Queries Ingested")
    tbl.add_column("Personality")
    for e in experts:
        persona = (e.personality[:50] + "…") if len(e.personality) > 50 else (e.personality or "(none)")
        tbl.add_row(
            e.name,
            e.slug,
            str(e.max_docs),
            ", ".join(e.queries_ingested) if e.queries_ingested else "(none)",
            persona,
        )
    rprint(tbl)


def _create_expert(registry, experts_dir, embed_cfg, llm_cfg) -> None:
    import questionary

    name = questionary.text("Expert's full name:").ask()
    if not name:
        return

    max_docs_str = questionary.text("Max docs to ingest:", default="30").ask()
    try:
        max_docs = int(max_docs_str)
    except (ValueError, TypeError):
        max_docs = 30

    personality = questionary.text("Personality description (optional, Enter to skip):").ask() or ""

    confirmed = questionary.confirm(f"Create expert '{name}' and run ingest now?").ask()
    if not confirmed:
        return

    meta = registry.create(name, max_docs=max_docs, personality=personality)
    rprint(f"[green]Created expert:[/green] {meta.name} (slug: {meta.slug})")
    _run_ingest_for_expert(meta, meta.name, registry, experts_dir, embed_cfg, llm_cfg)


def _run_ingest_for_expert(meta, query_name, registry, experts_dir, embed_cfg, llm_cfg) -> None:
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    from core.embeddings.client import EmbeddingsClient
    from services.experts.ingest import ingest_for_expert
    from services.experts.paths import build_expert_paths

    if llm_cfg is None:
        llm_cfg = _resolve_llm_cfg(Path(experts_dir))
    embed_client = EmbeddingsClient(embed_cfg)

    rprint(f"[dim]Searching NTRS for author: {query_name!r}[/dim]")

    dl_progress = Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn()
    )
    dl_progress.start()
    dl_task = dl_progress.add_task("Searching NTRS...", total=max(1, meta.max_docs))

    extract_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    )

    state: dict = {
        "extract_task": None,
        "extract_started": False,
        "dl_stopped": False,
        "records_found": None,
        "download_failures": [],
    }

    def on_search_complete(total):
        state["records_found"] = total
        dl_progress.update(
            dl_task,
            description="Downloading PDFs..." if total else "No downloadable PDFs found",
            total=max(1, total),
        )

    def on_download(title):
        dl_progress.update(dl_task, description=f"Downloading PDFs... {title[:48]}")
        dl_progress.advance(dl_task)

    def on_download_failed(title, error):
        state["download_failures"].append((title, error))
        LOGGER.warning("Expert download failed for %s: %s", title, error)

    def on_downloads_complete(records):
        dl_progress.stop()
        state["dl_stopped"] = True
        if records:
            tbl = Table(title="Downloaded Documents")
            tbl.add_column("Title", ratio=1)
            tbl.add_column("URL", style="blue", no_wrap=True)
            for t, u in records:
                tbl.add_row(t, u)
            rprint(tbl)

    def on_extract_start(total_chunks):
        extract_progress.start()
        state["extract_task"] = extract_progress.add_task(
            "Extracting triples...", total=total_chunks
        )
        state["extract_started"] = True

    def on_chunk_extracted(done, total, triples):
        if state["extract_task"] is not None:
            extract_progress.update(
                state["extract_task"],
                completed=done,
                description=f"Extracting triples... (+{triples})",
            )

    result = ingest_for_expert(
        meta,
        query_name,
        experts_dir,
        llm_cfg,
        embed_client,
        on_search_complete=on_search_complete,
        on_download=on_download,
        on_download_failed=on_download_failed,
        on_downloads_complete=on_downloads_complete,
        on_extract_start=on_extract_start,
        on_chunk_extracted=on_chunk_extracted,
    )

    if not state["dl_stopped"]:
        dl_progress.stop()
    if state["extract_started"]:
        extract_progress.stop()

    if result.docs_added > 0:
        rprint(f"[green]Ingested:[/green] {result.docs_added} docs, {result.chunks_added} chunks")
        updated_queries = list(meta.queries_ingested) + [query_name]
        db_path = build_expert_paths(experts_dir, meta.slug).db_path
        from services.experts.expertise import compute_expertise_areas
        areas = compute_expertise_areas(db_path)
        registry.update(meta.slug, queries_ingested=updated_queries, expertise_areas=areas)
        if areas:
            rprint(f"[dim]Expertise areas:[/dim] {', '.join(areas)}")
    elif state["download_failures"]:
        rprint(
            f"[red]Found {state['records_found'] or len(state['download_failures'])} candidate documents, "
            f"but {len(state['download_failures'])} download(s) failed.[/red]"
        )
        tbl = Table(title="Download Failures")
        tbl.add_column("Title", ratio=1)
        tbl.add_column("Error")
        for title, error in state["download_failures"]:
            tbl.add_row(title, error)
        rprint(tbl)
    else:
        rprint(f"[red]No documents found for '{query_name}'.[/red]")


def _manage_expert_menu(registry, experts_dir, embed_cfg, llm_cfg) -> None:
    import questionary

    experts = registry.list()
    if not experts:
        rprint("[yellow]No experts found.[/yellow]")
        return

    names = [e.name for e in experts] + ["Back"]
    chosen_name = questionary.select("Select expert to manage:", choices=names).ask()
    if not chosen_name or chosen_name == "Back":
        return

    meta = next((e for e in experts if e.name == chosen_name), None)
    if not meta:
        return

    while True:
        rprint(Panel(f"[bold]Managing:[/bold] {meta.name}", expand=False))
        action = questionary.select(
            "Action:",
            choices=[
                "View details",
                "Re-ingest (refresh NTRS publications)",
                "Refresh expertise areas",
                "Add alias (ingest under alternate name)",
                "Edit personality",
                "Reset database (delete DB + embeddings)",
                "Delete expert",
                "Back",
            ],
        ).ask()

        if action is None or action == "Back":
            break
        elif action == "View details":
            _view_expert(meta, experts_dir)
        elif action == "Re-ingest (refresh NTRS publications)":
            _run_ingest_for_expert(meta, meta.name, registry, experts_dir, embed_cfg, llm_cfg)
            meta = registry.get(meta.slug) or meta
        elif action == "Refresh expertise areas":
            from services.experts.expertise import compute_expertise_areas
            from services.experts.paths import build_expert_paths

            db_path = build_expert_paths(experts_dir, meta.slug).db_path
            rprint("[dim]Fetching subject categories from NTRS...[/dim]")
            areas = compute_expertise_areas(db_path)
            if areas:
                registry.update(meta.slug, expertise_areas=areas)
                meta = registry.get(meta.slug) or meta
                rprint(f"[green]Expertise areas updated:[/green] {', '.join(areas)}")
            else:
                rprint("[yellow]No subject categories found.[/yellow]")
        elif action == "Add alias (ingest under alternate name)":
            alias = questionary.text("Alias name to ingest:").ask()
            if alias:
                _run_ingest_for_expert(meta, alias, registry, experts_dir, embed_cfg, llm_cfg)
                aliases = list(meta.aliases) + [alias]
                registry.update(meta.slug, aliases=aliases)
                meta = registry.get(meta.slug) or meta
        elif action == "Edit personality":
            new_persona = questionary.text("New personality:", default=meta.personality).ask()
            if new_persona is not None:
                registry.update(meta.slug, personality=new_persona)
                meta = registry.get(meta.slug) or meta
                rprint("[green]Personality updated.[/green]")
        elif action == "Reset database (delete DB + embeddings)":
            confirmed = questionary.confirm(
                f"Delete DB and embeddings for '{meta.name}'?"
            ).ask()
            if confirmed:
                from services.experts.paths import build_expert_paths

                paths = build_expert_paths(experts_dir, meta.slug)
                for p in (paths.db_path, paths.embeddings_path):
                    if p.exists():
                        p.unlink()
                if paths.downloads_dir.exists():
                    shutil.rmtree(paths.downloads_dir)
                rprint("[green]Database and embeddings deleted.[/green]")
        elif action == "Delete expert":
            confirmed = questionary.confirm(
                f"Permanently delete expert '{meta.name}' and all their data?"
            ).ask()
            if confirmed:
                registry.delete(meta.slug)
                rprint(f"[red]Deleted expert '{meta.name}'.[/red]")
                break


def _view_expert(meta, experts_dir) -> None:
    from services.experts.paths import build_expert_paths

    paths = build_expert_paths(experts_dir, meta.slug)
    db_exists = paths.db_path.exists()
    emb_exists = paths.embeddings_path.exists()
    tbl = Table(title=f"Expert: {meta.name}")
    tbl.add_column("Field", style="bold")
    tbl.add_column("Value")
    tbl.add_row("Name", meta.name)
    tbl.add_row("Slug", meta.slug)
    tbl.add_row("Aliases", ", ".join(meta.aliases) or "(none)")
    tbl.add_row("Personality", meta.personality or "(none)")
    tbl.add_row("Queries Ingested", ", ".join(meta.queries_ingested) or "(none)")
    tbl.add_row("Max Docs", str(meta.max_docs))
    tbl.add_row("Created At", meta.created_at)
    tbl.add_row("DB Exists", "✓" if db_exists else "✗")
    tbl.add_row("Embeddings Exist", "✓" if emb_exists else "✗")
    rprint(tbl)
