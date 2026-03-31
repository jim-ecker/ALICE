from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn

from services.chat.config import load_chat_config
from core.llm.config import LLMConfig
from core.embeddings.client import EmbeddingsClient


@dataclass
class IngestResult:
    docs_added: int
    chunks_added: int
    index_size: int


@dataclass
class ServiceState:
    retriever: Any          # services.chat.retriever.Retriever
    chat_store: Any         # core.graph.chat_store.ChatStore
    llm: Any                # LLMBackend
    llm_cfg: Any            # LLMConfig
    embed_client: Any       # EmbeddingsClient
    db: Any                 # kuzu.Database
    active_expert: str | None = None   # slug, or None = standard chat mode
    expert_name: str | None = None     # display name
    expert_persona: str | None = None  # personality prompt


class Chat:
    def __init__(
        self,
        db_path: Path = Path("chat.db"),
        *,
        llm_cfg: LLMConfig | None = None,
        embed_client: EmbeddingsClient | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._chat_cfg, self._embed_cfg, self._scoring_cfg, chat_llm_cfg = load_chat_config(
            self._db_path.parent
        )
        self._default_chat_db_path = self._chat_cfg.db_path
        self._chat_cfg.db_path = self._db_path
        # [chat_llm] in alice.toml overrides the caller-supplied llm_cfg for chat responses
        if chat_llm_cfg is not None and llm_cfg is None:
            llm_cfg = chat_llm_cfg
        if host is not None:
            self._chat_cfg.host = host
        if port is not None:
            self._chat_cfg.port = port

        self._llm_cfg = llm_cfg
        self._embed_client = embed_client
        self._state: ServiceState | None = None
        self._db: Any = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_state(self, db_path: Path, embeddings_path: Path) -> ServiceState:
        """Construct a ServiceState from the given DB and embeddings paths."""
        from core.llm.config import resolve_config
        from core.llm.factory import create_backend
        from core.graph import KuzuStore
        from core.graph.kuzu_store import _BUFFER_POOL_SIZE
        from core.graph.chat_store import ChatStore
        from core.embeddings.builder import build_index
        from core.embeddings.index import EmbeddingIndex
        from core.scoring import (
            EmbeddingRelevanceScorer,
            GroundingScorer,
            ProvenanceScorer,
            WeightedCompositeScorer,
        )
        from services.chat.retriever import Retriever
        import kuzu

        if self._llm_cfg is None:
            self._llm_cfg = resolve_config(
                cli_model=None,
                cli_backend=None,
                cli_base_url=None,
                cli_api_key=None,
                cli_workers=None,
                start_dir=self._db_path.parent,
            )
        if self._embed_client is None:
            self._embed_client = EmbeddingsClient(self._embed_cfg)

        db = kuzu.Database(str(db_path), buffer_pool_size=_BUFFER_POOL_SIZE)
        store = KuzuStore(db)
        chat_store = ChatStore(db)
        llm = create_backend(self._llm_cfg)

        if embeddings_path.exists():
            index = EmbeddingIndex.load(embeddings_path)
        else:
            index = build_index(store, self._embed_client, show_progress=True)
            index.save(embeddings_path)

        relevance_scorer = EmbeddingRelevanceScorer(self._embed_client)
        provenance_scorer = ProvenanceScorer(kuzu.Connection(db))
        grounding_scorer = (
            GroundingScorer(llm) if self._scoring_cfg.grounding_enabled else None
        )
        scorer = WeightedCompositeScorer(
            cfg=self._scoring_cfg,
            relevance_scorer=relevance_scorer,
            provenance_scorer=provenance_scorer,
            grounding_scorer=grounding_scorer,
        )
        retriever = Retriever(
            index,
            self._embed_client,
            kuzu.Connection(db),
            scorer,
            top_k=self._chat_cfg.top_k_chunks,
            hop_depth=self._chat_cfg.entity_hop_depth,
            max_hop2_entities=self._chat_cfg.max_hop2_entities,
        )

        return ServiceState(
            retriever=retriever,
            chat_store=chat_store,
            llm=llm,
            llm_cfg=self._llm_cfg,
            embed_client=self._embed_client,
            db=db,
        )

    def _setup(self) -> None:
        """Eagerly initialise all service objects (called by create_app / serve)."""
        db_path = self._chat_cfg.db_path
        emb_path = self._resolve_embeddings_path(db_path)
        state = self._build_state(db_path, emb_path)
        self._db = state.db
        self._state = state

    def _resolve_embeddings_path(self, db_path: Path) -> Path:
        """Return the embeddings file associated with a database path.

        If the caller is using the configured chat database name, preserve the
        configured embeddings filename. For custom database paths, derive a
        sibling <stem>.embeddings.npz file so alternate graphs stay isolated.
        """
        configured_db = self._default_chat_db_path
        if db_path == configured_db:
            return db_path.parent / self._chat_cfg.embeddings_path
        return db_path.with_name(f"{db_path.stem}.embeddings.npz")

    def _update_state_from(self, new_state: ServiceState) -> None:
        """Update self._state fields in-place so FastAPI route closures see new values."""
        self._state.retriever = new_state.retriever
        self._state.chat_store = new_state.chat_store
        self._state.llm = new_state.llm
        self._state.llm_cfg = new_state.llm_cfg
        self._state.embed_client = new_state.embed_client
        self._state.db = new_state.db
        self._db = new_state.db

    # ── Public API ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the current KuzuDB connection."""
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None

    def switch_to_chat(self) -> None:
        """Hot-swap back to the standard chat.db without restarting the server."""
        self.close()
        db_path = self._chat_cfg.db_path
        emb_path = self._resolve_embeddings_path(db_path)
        new_state = self._build_state(db_path, emb_path)
        self._update_state_from(new_state)
        self._state.active_expert = None
        self._state.expert_name = None
        self._state.expert_persona = None

    def switch_expert(self, slug: str) -> None:
        """Hot-swap to a virtual expert's database without restarting the server.

        Raises FileNotFoundError if the expert metadata or database is missing.
        """
        from services.experts.manager import ExpertRegistry

        experts_dir = self._chat_cfg.experts_dir
        registry = ExpertRegistry(experts_dir)
        meta = registry.get(slug)
        if meta is None:
            raise FileNotFoundError(f"Expert not found: {slug}")
        db_path = Path(experts_dir) / slug / f"{slug}.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Expert database not found: {db_path}")
        emb_path = Path(experts_dir) / slug / f"{slug}.embeddings.npz"

        self.close()
        new_state = self._build_state(db_path, emb_path)
        self._update_state_from(new_state)
        self._state.active_expert = slug
        self._state.expert_name = meta.name
        self._state.expert_persona = meta.personality or None

    def ingest(
        self,
        query: str,
        *,
        center: str | None = None,
        max_docs: int = 20,
        offset: int | None = None,
        download_workers: int = 10,
        chunk_workers: int = 4,
        dashboard_port: int = 8765,
        on_download=None,
        on_downloads_complete=None,
        on_chunk=None,
        on_extract_start=None,
        on_chunk_extracted=None,
    ) -> IngestResult:
        """Run the ingest pipeline into this chat's own db."""
        from services.ingest.service import Ingest

        db_path = self._chat_cfg.db_path
        emb_path = self._resolve_embeddings_path(db_path)

        if self._llm_cfg is None:
            from core.llm.config import resolve_config
            self._llm_cfg = resolve_config(start_dir=db_path.parent)
        if self._embed_client is None:
            self._embed_client = EmbeddingsClient(self._embed_cfg)

        ingest_svc = Ingest(
            db_path=db_path,
            llm_cfg=self._llm_cfg,
            embed_client=self._embed_client,
            embeddings_path=emb_path,
            downloads_dir=db_path.parent / "downloads",
            download_workers=download_workers,
            chunk_workers=chunk_workers,
        )
        result, new_index = ingest_svc.run(
            query,
            center=center,
            max_docs=max_docs,
            offset=offset,
            dashboard_port=dashboard_port,
            on_download=on_download,
            on_downloads_complete=on_downloads_complete,
            on_chunk=on_chunk,
            on_extract_start=on_extract_start,
            on_chunk_extracted=on_chunk_extracted,
        )
        if self._state is not None:
            self._state.retriever.update_index(new_index)
        return IngestResult(
            docs_added=result.docs_added,
            chunks_added=result.chunks_added,
            index_size=result.index_size,
        )

    def ingest_local_files(
        self,
        confirmed_docs: list,
        *,
        chunk_workers: int = 4,
        dashboard_port: int = 8765,
        on_chunk=None,
        on_extract_start=None,
        on_chunk_extracted=None,
    ) -> IngestResult:
        """Ingest a list of (pdf_path, meta_dict) tuples into the chat knowledge graph."""
        from services.ingest.service import Ingest

        db_path = self._chat_cfg.db_path
        emb_path = self._resolve_embeddings_path(db_path)

        if self._llm_cfg is None:
            from core.llm.config import resolve_config
            self._llm_cfg = resolve_config(start_dir=db_path.parent)
        if self._embed_client is None:
            self._embed_client = EmbeddingsClient(self._embed_cfg)

        ingest_svc = Ingest(
            db_path=db_path,
            llm_cfg=self._llm_cfg,
            embed_client=self._embed_client,
            embeddings_path=emb_path,
            chunk_workers=chunk_workers,
        )
        result, new_index = ingest_svc.ingest_local_files(
            confirmed_docs,
            dashboard_port=dashboard_port,
            on_chunk=on_chunk,
            on_extract_start=on_extract_start,
            on_chunk_extracted=on_chunk_extracted,
        )
        if self._state is not None:
            self._state.retriever.update_index(new_index)
        return IngestResult(
            docs_added=result.docs_added,
            chunks_added=result.chunks_added,
            index_size=result.index_size,
        )

    def create_app(self):
        """Return a configured FastAPI app, triggering _setup() if needed."""
        if self._state is None:
            self._setup()
        from services.chat.app import create_app

        return create_app(
            state=self._state,
            chat=self,
            cfg=self._chat_cfg,
        )

    def serve(self, host: str | None = None, port: int | None = None) -> None:
        """Start the uvicorn server (blocking)."""
        app = self.create_app()
        uvicorn.run(
            app,
            host=host or self._chat_cfg.host,
            port=port or self._chat_cfg.port,
        )
