from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.scoring.base import ScoredRetrievalResult
from services.chat.ui import CHAT_HTML


# ── Pydantic models ──────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    backend: str
    model: str
    index_size: int


class ConversationItem(BaseModel):
    id: str
    title: str
    created_at: str


class ListConversationsResponse(BaseModel):
    conversations: list[ConversationItem]


class CreateConversationRequest(BaseModel):
    title: str = "New Conversation"


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    citations: list[Any] = []


class GetMessagesResponse(BaseModel):
    messages: list[MessageItem]


class SendMessageRequest(BaseModel):
    content: str


class CitationTriple(BaseModel):
    fact_index: int
    subject: str
    subject_type: str
    relation: str
    object_: str
    object_type: str
    # ── Trust signals ────────────────────────────────────────────────────────
    ingest_certainty: float
    relevance_score: float | None
    grounding_score: float | None
    provenance_count: int
    composite_trust: float


class Citation(BaseModel):
    chunk_id: str
    content: str
    document_title: str
    document_url: str | None
    page_number: int | None
    section_heading: str | None
    triples: list[CitationTriple]


class SendMessageResponse(BaseModel):
    message_id: str
    content: str
    citations: list[Citation]
    new_title: str


class ExpertItem(BaseModel):
    slug: str
    name: str
    personality: str
    expertise_areas: list[str]
    db_exists: bool


class ListExpertsResponse(BaseModel):
    experts: list[ExpertItem]


class ActiveExpertResponse(BaseModel):
    slug: str | None
    name: str | None


class SwitchExpertRequest(BaseModel):
    slug: str


# ── Identity ─────────────────────────────────────────────────────────────────

def get_current_user(request: Request) -> str:
    """Read user identity from the X-Remote-User header set by Apache/mod_auth_mellon.

    Returns an empty string when running without a proxy (local dev), which maps
    all requests to the anonymous owner namespace. Normalised to lowercase to
    prevent duplicate history buckets if the IdP varies email case between sessions.
    """
    return request.headers.get("X-Remote-User", "").strip().lower()


# ── App factory ──────────────────────────────────────────────────────────────

def create_app(state, chat, cfg) -> FastAPI:
    """Create the FastAPI application.

    state: ServiceState (mutable — routes read it on each request for hot-swap support)
    chat: Chat service instance (provides switch_expert / switch_to_chat)
    cfg: ChatConfig
    """
    app = FastAPI(title="ALICE Chat", version="0.1.0")
    _images_dir = Path(__file__).parent / "images"
    if _images_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_images_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(content=CHAT_HTML)

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        return StatusResponse(
            backend=state.llm_cfg.backend,
            model=state.llm_cfg.model,
            index_size=len(state.retriever._index),
        )

    @app.get("/api/conversations", response_model=ListConversationsResponse)
    async def list_conversations(owner: str = Depends(get_current_user)):
        convs = state.chat_store.list_conversations(owner=owner)
        return ListConversationsResponse(
            conversations=[
                ConversationItem(id=c.id, title=c.title, created_at=c.created_at)
                for c in convs
            ]
        )

    @app.post("/api/conversations", response_model=ConversationResponse, status_code=201)
    async def create_conversation(req: CreateConversationRequest, owner: str = Depends(get_current_user)):
        conv = state.chat_store.create_conversation(req.title, owner=owner)
        return ConversationResponse(id=conv.id, title=conv.title, created_at=conv.created_at)

    @app.delete("/api/conversations/{conv_id}", status_code=204)
    async def delete_conversation(conv_id: str, owner: str = Depends(get_current_user)):
        conv = state.chat_store.get_conversation(conv_id, owner=owner)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        state.chat_store.delete_conversation(conv_id, owner=owner)

    @app.get("/api/conversations/{conv_id}/messages", response_model=GetMessagesResponse)
    async def get_messages(conv_id: str, owner: str = Depends(get_current_user)):
        conv = state.chat_store.get_conversation(conv_id, owner=owner)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = state.chat_store.get_messages(conv_id)
        items = [
            MessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                citations=json.loads(m.citations_json) if m.citations_json else [],
            )
            for m in messages
        ]
        return GetMessagesResponse(messages=items)

    @app.post("/api/conversations/{conv_id}/messages", response_model=SendMessageResponse)
    async def send_message(conv_id: str, req: SendMessageRequest, owner: str = Depends(get_current_user)):
        conv = state.chat_store.get_conversation(conv_id, owner=owner)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # 1. Retrieve context + score triples
        retrieval: ScoredRetrievalResult = await asyncio.to_thread(
            state.retriever.retrieve, req.content
        )

        # 2. Fetch history
        history = state.chat_store.get_messages(conv_id)

        # 3. Build prompt (pass expert persona if active)
        from services.chat.prompt import build_prompt
        messages, fact_index_to_chunk_id = build_prompt(
            req.content,
            retrieval,
            history,
            history_turns=cfg.history_turns,
            max_context_chunks=cfg.max_context_chunks,
            expert_name=state.expert_name,
            expert_persona=state.expert_persona,
        )

        # 4. Generate answer
        answer: str = await asyncio.to_thread(state.llm.chat, messages, cfg.max_tokens)

        # 4a. Collect only the chunks the LLM actually cited, in citation order
        cited_fact_indices = list(
            dict.fromkeys(int(m) for m in re.findall(r'Fact_(\d+)', answer))
        )
        cited_chunk_ids_ordered = list(dict.fromkeys(
            fact_index_to_chunk_id[i]
            for i in cited_fact_indices
            if i in fact_index_to_chunk_id
        ))
        citation_chunk_ids = cited_chunk_ids_ordered

        # 5. Save user message
        state.chat_store.add_message(conv_id, "user", req.content, [])

        # 6. Build citation objects from trust bundles
        chunk_map = {c.chunk_id: c for c in retrieval.chunks}
        bundle_map: dict[str, list[tuple[int, object]]] = {}
        for fact_idx, b in enumerate(retrieval.trust_bundles, start=1):
            bundle_map.setdefault(b.triple.chunk_id, []).append((fact_idx, b))

        cited_set = set(cited_fact_indices)
        citations: list[Citation] = []
        for chunk_id in citation_chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            triples = [
                CitationTriple(
                    fact_index=fact_idx,
                    subject=b.triple.subject,
                    subject_type=b.triple.subject_type,
                    relation=b.triple.relation,
                    object_=b.triple.object_,
                    object_type=b.triple.object_type,
                    ingest_certainty=b.ingest_certainty,
                    relevance_score=b.relevance_score,
                    grounding_score=b.grounding_score,
                    provenance_count=b.provenance_count,
                    composite_trust=b.composite_trust,
                )
                for fact_idx, b in bundle_map.get(chunk_id, [])
                if fact_idx in cited_set
            ]
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    document_title=chunk.document_title,
                    document_url=chunk.document_url,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    triples=triples,
                )
            )

        # 7. Save assistant message with serialized citations
        citations_json = json.dumps([c.model_dump() for c in citations])
        asst_msg = state.chat_store.add_message(
            conv_id, "assistant", answer, citation_chunk_ids, citations_json
        )

        # 8. Generate and persist a short conversation title
        _title_messages = [
            {
                "role": "system",
                "content": (
                    "Generate a short title (5–8 words, no punctuation, no quotes) that captures "
                    "the main topic of this conversation exchange. Reply with only the title."
                ),
            },
            {
                "role": "user",
                "content": f"User asked: {req.content}\n\nAssistant answered: {answer[:400]}",
            },
        ]
        new_title: str = await asyncio.to_thread(state.llm.chat, _title_messages, 20)
        new_title = new_title.strip().strip('"').strip("'")
        state.chat_store.update_conversation_title(conv_id, new_title, owner=owner)

        return SendMessageResponse(
            message_id=asst_msg.id,
            content=answer,
            citations=citations,
            new_title=new_title,
        )

    # ── Expert endpoints ─────────────────────────────────────────────────────

    @app.get("/api/experts", response_model=ListExpertsResponse)
    async def list_experts():
        from services.experts.manager import ExpertRegistry

        experts_dir = cfg.experts_dir
        registry = ExpertRegistry(experts_dir)
        metas = registry.list()
        items = [
            ExpertItem(
                slug=m.slug,
                name=m.name,
                personality=m.personality,
                expertise_areas=m.expertise_areas,
                db_exists=(Path(experts_dir) / f"{m.slug}.db").exists(),
            )
            for m in metas
        ]
        return ListExpertsResponse(experts=items)

    @app.get("/api/experts/active", response_model=ActiveExpertResponse)
    async def get_active_expert():
        return ActiveExpertResponse(
            slug=state.active_expert,
            name=state.expert_name,
        )

    @app.post("/api/experts/switch", status_code=200)
    async def switch_expert(req: SwitchExpertRequest):
        try:
            await asyncio.to_thread(chat.switch_expert, req.slug)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Generate a personality-driven intro message
        from services.experts.manager import ExpertRegistry
        meta = ExpertRegistry(cfg.experts_dir).get(req.slug)
        areas_str = ", ".join(meta.expertise_areas) if meta and meta.expertise_areas else "their research"
        intro_messages = [
            {
                "role": "system",
                "content": (
                    f"You are {state.expert_name}, a NASA researcher and subject matter expert.\n"
                    f"{state.expert_persona or ''}\n\n"
                    f"Write a short introduction (2-4 sentences) for a user who just activated your AI avatar "
                    f"in the ALICE research assistant. You MUST: "
                    f"(1) say your full name ({state.expert_name}), "
                    f"(2) clearly state you are an AI avatar — not the real {state.expert_name}, "
                    f"(3) name your specific research areas: {areas_str}. "
                    f"Let your personality come through naturally — do NOT describe or announce your own personality, humor, or communication style. Just be it."
                ),
            },
            {"role": "user", "content": "Introduce yourself."},
        ]
        intro = await asyncio.to_thread(state.llm.chat, intro_messages, 200)

        return {"slug": state.active_expert, "name": state.expert_name, "intro": intro}

    @app.post("/api/experts/unload", status_code=200)
    async def unload_expert():
        await asyncio.to_thread(chat.switch_to_chat)
        return {"slug": None, "name": None}

    @app.get("/api/ntrs-pdf-url")
    async def ntrs_pdf_url(id: str):
        """Resolve the PDF download URL for an NTRS citation ID (server-side, bypasses CORS)."""
        import requests as _req
        from fastapi.responses import JSONResponse
        if not id.isdigit():
            raise HTTPException(status_code=400, detail="Invalid NTRS ID")
        def _fetch():
            r = _req.get(f"https://ntrs.nasa.gov/api/citations/{id}", timeout=15)
            r.raise_for_status()
            return r.json()
        data = await asyncio.to_thread(_fetch)
        downloads = data.get("downloads") or []
        pdf = next((d for d in downloads if d.get("links", {}).get("pdf")), None)
        url = ("https://ntrs.nasa.gov" + pdf["links"]["pdf"]) if pdf else None
        return JSONResponse({"url": url})

    @app.get("/api/pdf-proxy")
    async def pdf_proxy(url: str):
        """Proxy NTRS PDFs so the hosted PDF.js viewer can load them (CORS bypass)."""
        import requests as _req
        from fastapi.responses import Response as _Response
        if not (url.startswith("https://ntrs.nasa.gov/api/citations/") and ".pdf" in url):
            raise HTTPException(status_code=400, detail="Only NTRS PDF URLs are supported")
        def _fetch():
            r = _req.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        content = await asyncio.to_thread(_fetch)
        return _Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Content-Disposition": "inline",
            },
        )

    return app
