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
    from services.ingest.config import load_ingest_llm_config

    chat_cfg, embed_cfg, _, chat_llm_cfg = load_chat_config()
    experts_dir = chat_cfg.experts_dir
    registry = ExpertRegistry(experts_dir)
    llm_cfg = load_ingest_llm_config(Path(experts_dir)) or chat_llm_cfg

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

    personality = questionary.text("Personality description (optional, Enter to skip):").ask() or ""

    allowed_raw = questionary.text(
        "Allowed users (comma-separated NASA emails, Enter to make public):"
    ).ask() or ""
    allowed_users = [e.strip().lower() for e in allowed_raw.split(",") if e.strip()]

    kg_source = questionary.select(
        "Knowledge graph source:",
        choices=["NTRS (search by author name)", "Local folder (upload documents)", "None (persona only)"],
    ).ask()
    if kg_source is None:
        return

    if kg_source == "NTRS (search by author name)":
        max_docs_str = questionary.text("Max docs to ingest:", default="30").ask()
        try:
            max_docs = int(max_docs_str)
        except (ValueError, TypeError):
            max_docs = 30

        confirmed = questionary.confirm(f"Create expert '{name}' and run NTRS ingest now?").ask()
        if not confirmed:
            return

        meta = registry.create(name, max_docs=max_docs, personality=personality, allowed_users=allowed_users)
        rprint(f"[green]Created expert:[/green] {meta.name} (slug: {meta.slug})")
        _run_ingest_for_expert(meta, meta.name, registry, experts_dir, embed_cfg, llm_cfg)

    elif kg_source == "Local folder (upload documents)":
        folder_raw = questionary.text("Path to folder containing documents:").ask()
        if not folder_raw:
            return
        folder = Path(folder_raw).expanduser().resolve()
        if not folder.is_dir():
            rprint(f"[red]Not a directory: {folder}[/red]")
            return

        confirmed = questionary.confirm(f"Create expert '{name}' and ingest from {folder}?").ask()
        if not confirmed:
            return

        meta = registry.create(name, max_docs=999, personality=personality, allowed_users=allowed_users)
        rprint(f"[green]Created expert:[/green] {meta.name} (slug: {meta.slug})")
        _run_ingest_folder_for_expert(meta, folder, registry, experts_dir, embed_cfg, llm_cfg)

    else:  # None
        confirmed = questionary.confirm(f"Create persona-only expert '{name}' (no knowledge graph)?").ask()
        if not confirmed:
            return
        meta = registry.create(name, max_docs=0, personality=personality, allowed_users=allowed_users)
        rprint(f"[green]Created expert:[/green] {meta.name} (slug: {meta.slug})")
        rprint("[dim]No knowledge graph created. Expert will answer using general knowledge only.[/dim]")


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
        "extract_total_chunks": 0,
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
        state["extract_total_chunks"] = total_chunks

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
        update_kwargs: dict = {"queries_ingested": updated_queries}
        if areas:
            update_kwargs["expertise_areas"] = areas
        registry.update(meta.slug, **update_kwargs)
        if areas:
            rprint(f"[dim]Expertise areas:[/dim] {', '.join(areas)}")
    elif state["extract_total_chunks"]:
        rprint(
            f"[green]Resumed extraction:[/green] {state['extract_total_chunks']} existing chunks"
        )
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


def _run_ingest_folder_for_expert(meta, folder: "Path", registry, experts_dir, embed_cfg, llm_cfg) -> None:
    import questionary
    from pathlib import Path
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel as _Panel

    from core.embeddings.client import EmbeddingsClient
    from services.experts.ingest import ingest_folder_for_expert
    from services.ingest.metadata import extract_pdf_metadata, infer_missing_metadata

    folder = Path(folder)
    _EXCEL_EXTS = {".xlsx", ".xls"}
    _ALL_EXTS = ("pdf", "pptx", "docx", "html", "htm", "md", "adoc", "xlsx", "xls")
    all_files = sorted(p for ext in _ALL_EXTS for p in folder.glob(f"*.{ext}"))
    if not all_files:
        rprint(f"[red]No supported files found in {folder}[/red]")
        return

    rprint(f"[dim]Found {len(all_files)} file(s) in {folder}[/dim]")

    if llm_cfg is None:
        llm_cfg = _resolve_llm_cfg(Path(experts_dir))
    embed_client = EmbeddingsClient(embed_cfg)

    rprint("[dim]Extracting metadata...[/dim]")
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    def _load_meta(path):
        if path.suffix.lower() in _EXCEL_EXTS:
            from services.ingest.metadata import extract_excel_metadata
            return path, extract_excel_metadata(path)
        return path, extract_pdf_metadata(path)

    all_meta = {}
    with ThreadPoolExecutor(max_workers=min(4, len(all_files))) as executor:
        for future in _as_completed({executor.submit(_load_meta, p): p for p in all_files}):
            path, m = future.result()
            all_meta[path] = m

    needs_inference = [
        f for f in all_files
        if f.suffix.lower() not in _EXCEL_EXTS and (not all_meta[f].get("title") or not all_meta[f].get("authors"))
    ]
    skip_metadata = False
    if needs_inference:
        run_inference = questionary.confirm(
            f"{len(needs_inference)} document(s) have missing metadata. Run LLM inference to fill them in?",
            default=True,
        ).ask()
        if run_inference:
            rprint(f"[dim]Running LLM inference on {len(needs_inference)} document(s)...[/dim]")
            from core.llm.factory import create_backend
            llm = create_backend(llm_cfg)
            for f in needs_inference:
                all_meta[f] = infer_missing_metadata(all_meta[f], llm)
        else:
            skip_metadata = True

    confirmed_docs: list = []
    if skip_metadata:
        for path in all_files:
            m = all_meta[path]
            confirmed_docs.append((path, {
                "title": m.get("title") or path.stem,
                "authors": m.get("authors", ""),
                "year": m.get("year", ""),
            }))
    else:
        for path in all_files:
            m = all_meta[path]
            rprint(_Panel(f"[bold]{path.name}[/bold]", expand=False))
            title = questionary.text("Title:", default=m.get("title") or path.stem).ask()
            if title is None:
                rprint("[yellow]Skipping.[/yellow]")
                continue
            authors = questionary.text("Authors:", default=m.get("authors", "")).ask() or ""
            year = questionary.text("Year:", default=m.get("year", "")).ask() or ""
            confirmed_docs.append((path, {"title": title, "authors": authors, "year": year}))

    if not confirmed_docs:
        rprint("[yellow]No documents to ingest.[/yellow]")
        return

    tbl = Table(title="Documents to Ingest")
    tbl.add_column("File")
    tbl.add_column("Title", ratio=1)
    tbl.add_column("Authors")
    tbl.add_column("Year", justify="right")
    for p, m in confirmed_docs:
        tbl.add_row(p.name, m.get("title", ""), m.get("authors", ""), m.get("year", ""))
    rprint(tbl)

    if not questionary.confirm("Proceed with ingestion?", default=True).ask():
        rprint("[yellow]Aborted.[/yellow]")
        return

    extract_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    )
    state: dict = {"extract_task": None, "extract_started": False, "chunk_task": None}
    chunk_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    chunk_progress.start()
    state["chunk_task"] = chunk_progress.add_task("Chunking files...", total=len(confirmed_docs))

    def on_chunk(title, chunk_count):
        chunk_progress.advance(state["chunk_task"])

    def on_extract_start(total_chunks):
        chunk_progress.stop()
        extract_progress.start()
        state["extract_task"] = extract_progress.add_task("Extracting triples...", total=total_chunks)
        state["extract_started"] = True

    def on_chunk_extracted(done, total, triples):
        if state["extract_task"] is not None:
            extract_progress.update(state["extract_task"], completed=done, description=f"Extracting triples... (+{triples})")

    result = ingest_folder_for_expert(
        meta,
        folder,
        experts_dir,
        llm_cfg,
        embed_client,
        confirmed_docs,
        on_chunk=on_chunk,
        on_extract_start=on_extract_start,
        on_chunk_extracted=on_chunk_extracted,
    )

    chunk_progress.stop()
    if state["extract_started"]:
        extract_progress.stop()

    if result.docs_added > 0:
        rprint(f"[green]Ingested:[/green] {result.docs_added} docs, {result.chunks_added} chunks")
        from services.experts.paths import build_expert_paths
        from services.experts.expertise import compute_expertise_areas
        db_path = build_expert_paths(experts_dir, meta.slug).db_path
        areas = compute_expertise_areas(db_path)
        if areas:
            registry.update(meta.slug, expertise_areas=areas)
            rprint(f"[dim]Expertise areas:[/dim] {', '.join(areas)}")
    else:
        rprint("[yellow]No new documents ingested.[/yellow]")


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
                "Add local documents",
                "Refresh expertise areas",
                "Add alias (ingest under alternate name)",
                "Edit personality",
                "Edit personality strength",
                "Edit allowed users",
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
        elif action == "Add local documents":
            folder_raw = questionary.text("Path to folder containing documents:").ask()
            if folder_raw:
                from pathlib import Path as _Path
                folder = _Path(folder_raw).expanduser().resolve()
                if not folder.is_dir():
                    rprint(f"[red]Not a directory: {folder}[/red]")
                else:
                    _run_ingest_folder_for_expert(meta, folder, registry, experts_dir, embed_cfg, llm_cfg)
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
        elif action == "Edit allowed users":
            current = ", ".join(meta.allowed_users) if meta.allowed_users else ""
            rprint("[dim]Leave blank to make this expert accessible to all users.[/dim]")
            new_raw = questionary.text(
                "Allowed users (comma-separated NASA emails):",
                default=current,
            ).ask()
            if new_raw is not None:
                new_users = [e.strip().lower() for e in new_raw.split(",") if e.strip()]
                registry.update(meta.slug, allowed_users=new_users)
                meta = registry.get(meta.slug) or meta
                if new_users:
                    rprint(f"[green]Access restricted to:[/green] {', '.join(new_users)}")
                else:
                    rprint("[green]Expert is now accessible to all users.[/green]")
        elif action == "Edit personality strength":
            current_pct = int(meta.personality_strength * 100)
            rprint(
                "[dim]0% = no personality applied  "
                "1–30% = subtle  "
                "31–60% = moderate  "
                "61–90% = strong  "
                "91–100% = full[/dim]"
            )
            raw = questionary.text(
                "Personality strength (0–100):",
                default=str(current_pct),
            ).ask()
            if raw is not None:
                try:
                    pct = max(0, min(100, int(raw)))
                except ValueError:
                    rprint("[red]Invalid value — enter a number between 0 and 100.[/red]")
                else:
                    registry.update(meta.slug, personality_strength=pct / 100)
                    meta = registry.get(meta.slug) or meta
                    rprint(f"[green]Personality strength set to {pct}%.[/green]")
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
    tbl.add_row("Personality Strength", f"{int(meta.personality_strength * 100)}%")
    tbl.add_row("Allowed Users", ", ".join(meta.allowed_users) if meta.allowed_users else "(all users)")
    tbl.add_row("Queries Ingested", ", ".join(meta.queries_ingested) or "(none)")
    tbl.add_row("Max Docs", str(meta.max_docs))
    tbl.add_row("Created At", meta.created_at)
    tbl.add_row("DB Exists", "✓" if db_exists else "✗")
    tbl.add_row("Embeddings Exist", "✓" if emb_exists else "✗")
    rprint(tbl)
