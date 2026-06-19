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
from services.experiment.ui import EXPERIMENT_HTML


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
    evidence_text: str | None = None
    evidence_char_start: int | None = None
    evidence_char_end: int | None = None
    extractor_certainty: float | None = None


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
    abstain: float = 0.0


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


# ── Text sanitization helpers ────────────────────────────────────────────────

def _clean_model_text(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^```[a-z]*\n?([\s\S]*?)\n?```$', r'\1', text.strip())
    return text.strip()


_REASONING_PREFIXES = re.compile(
    r'^(we need to|i need to|let me|okay|so |the user|the instruction|we are|i should|i will|first|alright)',
    re.IGNORECASE,
)

def _sanitize_title(raw_title: str, fallback: str) -> str:
    text = _clean_model_text(raw_title)
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    else:
        return fallback
    # Reject lines that look like reasoning internal monologue
    if _REASONING_PREFIXES.match(text) or len(text.split()) > 15:
        return fallback
    text = re.sub(r'^(title|conversation title)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = text.strip('"\'`')
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return fallback
    return text[:77] + '…' if len(text) > 80 else text


def _fallback_title_from_query(query: str) -> str:
    q = query.strip()
    if len(q) <= 60:
        return q
    cut = q.rfind(' ', 0, 60)
    return (q[:cut] if cut > 20 else q[:60]) + '…'


def _fallback_expert_intro(name: str, areas: list[str]) -> str:
    areas_str = ', '.join(areas) if areas else 'my research'
    return (
        f"I am an AI avatar for {name} in ALICE, not the real person. "
        f"My knowledge is based on {name}'s published work in {areas_str}. "
        f"How can I help you today?"
    )


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

    from services.experiment.api import make_experiment_router
    app.include_router(make_experiment_router(cfg, state))

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(content=CHAT_HTML)

    @app.get("/experiment", response_class=HTMLResponse)
    async def experiment_ui():
        return HTMLResponse(content=EXPERIMENT_HTML)

    @app.get("/api/me")
    async def get_me(owner: str = Depends(get_current_user)):
        return {"email": owner}

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        return StatusResponse(
            backend=state.llm_cfg.backend,
            model=state.llm_cfg.model,
            index_size=len(state.retriever._index),
        )

    @app.get("/api/debug/tppr")
    async def debug_tppr(q: str):
        """Probe TPPR contribution for a query against the live KG."""
        import math
        from core.graph.retrieval import _match_anchor_entities, find_entity_chunks, find_trust_paths

        conn = state.retriever._conn

        anchors = _match_anchor_entities(conn, q)

        paths = find_trust_paths(conn, anchors, max_paths=state.retriever._max_trust_paths)
        all_paths = find_trust_paths(conn, anchors, max_paths=10_000)
        uncapped_1hop = sum(1 for p in all_paths if len(p.edges) == 1)
        uncapped_2hop = sum(1 for p in all_paths if len(p.edges) == 2)
        capped_1hop = sum(1 for p in paths if len(p.edges) == 1)
        capped_2hop = sum(1 for p in paths if len(p.edges) == 2)

        entity_ids = find_entity_chunks(
            conn, q,
            hop_depth=state.retriever._hop_depth,
            max_hop2_entities=state.retriever._max_hop2_entities,
        )
        baseline: set[str] = set(entity_ids)

        tppr_ids: set[str] = set()
        for path in paths:
            for cid in path.chunk_ids:
                tppr_ids.add(cid)

        tppr_only = tppr_ids - baseline

        return {
            "query": q,
            "path_retrieval_enabled": state.retriever._path_retrieval,
            "anchors": anchors,
            "paths_found": len(paths),
            "path_breakdown": {
                "uncapped_total": len(all_paths),
                "uncapped_1hop": uncapped_1hop,
                "uncapped_2hop": uncapped_2hop,
                "capped_1hop": capped_1hop,
                "capped_2hop": capped_2hop,
            },
            "sample_1hop_paths": [
                {
                    "anchor": p.anchor,
                    "trust": round(p.path_trust, 4),
                    "edges": [
                        {"relation": rel, "target": tgt, "certainty": round(cert, 4)}
                        for rel, tgt, cert, _ in p.edges
                    ],
                    "chunk_ids": p.chunk_ids,
                }
                for p in paths if len(p.edges) == 1
            ][:5],
            "sample_2hop_paths": [
                {
                    "anchor": p.anchor,
                    "trust": round(p.path_trust, 4),
                    "edges": [
                        {"relation": rel, "target": tgt, "certainty": round(cert, 4)}
                        for rel, tgt, cert, _ in p.edges
                    ],
                    "chunk_ids": p.chunk_ids,
                }
                for p in paths if len(p.edges) == 2
            ][:10],
            "baseline_chunk_count": len(baseline),
            "tppr_chunk_count": len(tppr_ids),
            "tppr_overlap_count": len(tppr_ids & baseline),
            "tppr_new_count": len(tppr_only),
            "tppr_new_chunk_ids": list(tppr_only)[:20],
        }

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
            expert_persona_strength=state.expert_persona_strength,
        )

        # 4. Generate answer
        answer: str = await asyncio.to_thread(state.llm.chat, messages, cfg.max_tokens)

        # Strip any LLM-generated general-knowledge warning — the app adds its own below
        _WARNING_PREFIX = "⚠️ The knowledge graph does not contain information to answer this question. The following answer is based on general knowledge and should be verified for accuracy."
        answer = answer.removeprefix(_WARNING_PREFIX).lstrip()

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
                    evidence_text=b.triple.evidence_text,
                    evidence_char_start=b.triple.evidence_char_start,
                    evidence_char_end=b.triple.evidence_char_end,
                    extractor_certainty=b.triple.raw_certainty_score,
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

        # 7. Prepend grounding warning when no KG facts were cited despite context being available
        if not citations and (retrieval.chunks or retrieval.trust_bundles):
            answer = (
                "⚠️ The knowledge graph does not contain information to answer this question. "
                "The following answer is based on general knowledge and should be verified for accuracy.\n\n"
                + answer
            )

        # 8. Save assistant message with serialized citations
        citations_json = json.dumps([c.model_dump() for c in citations])
        asst_msg = state.chat_store.add_message(
            conv_id, "assistant", answer, citation_chunk_ids, citations_json
        )

        # 9. Generate and persist a short conversation title
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
        try:
            raw_title: str = await asyncio.to_thread(state.llm.chat, _title_messages, 512)
            new_title = _sanitize_title(raw_title, _fallback_title_from_query(req.content))
        except Exception:
            new_title = _fallback_title_from_query(req.content)
        state.chat_store.update_conversation_title(conv_id, new_title, owner=owner)

        return SendMessageResponse(
            message_id=asst_msg.id,
            content=answer,
            citations=citations,
            new_title=new_title,
            abstain=retrieval.abstain,
        )

    # ── Expert endpoints ─────────────────────────────────────────────────────

    @app.get("/api/experts", response_model=ListExpertsResponse)
    async def list_experts(owner: str = Depends(get_current_user)):
        from services.experts.manager import ExpertRegistry
        from services.experts.paths import build_expert_paths

        experts_dir = cfg.experts_dir
        registry = ExpertRegistry(experts_dir)
        metas = registry.list()
        items = [
            ExpertItem(
                slug=m.slug,
                name=m.name,
                personality=m.personality,
                expertise_areas=m.expertise_areas,
                db_exists=build_expert_paths(experts_dir, m.slug).db_path.exists(),
            )
            for m in metas
            if not m.allowed_users or owner in m.allowed_users
        ]
        return ListExpertsResponse(experts=items)

    @app.get("/api/experts/active", response_model=ActiveExpertResponse)
    async def get_active_expert():
        return ActiveExpertResponse(
            slug=state.active_expert,
            name=state.expert_name,
        )

    @app.post("/api/experts/switch", status_code=200)
    async def switch_expert(req: SwitchExpertRequest, owner: str = Depends(get_current_user)):
        from services.experts.manager import ExpertRegistry
        meta = ExpertRegistry(cfg.experts_dir).get(req.slug)
        if meta is None:
            raise HTTPException(status_code=404, detail="Expert not found")
        if meta.allowed_users and owner not in meta.allowed_users:
            raise HTTPException(status_code=403, detail="Access to this expert is restricted")
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
        areas_list = meta.expertise_areas if meta and meta.expertise_areas else []
        try:
            raw_intro: str = await asyncio.to_thread(state.llm.chat, intro_messages, 1024)
            intro = _clean_model_text(raw_intro)
        except Exception:
            intro = ""
        if not intro:
            intro = _fallback_expert_intro(state.expert_name, areas_list)
        elif not re.search(r'\b(ai|avatar|artificial|not the real)\b', intro, re.IGNORECASE):
            intro += f"\n\nPlease note: I am an AI avatar for {state.expert_name} in ALICE, not the real person."

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
        _headers = {"User-Agent": "Mozilla/5.0 (compatible; ALICE-bot/1.0)"}
        def _fetch():
            r = _req.get(f"https://ntrs.nasa.gov/api/citations/{id}", timeout=15, headers=_headers)
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
        _headers = {"User-Agent": "Mozilla/5.0 (compatible; ALICE-bot/1.0)"}
        def _fetch():
            r = _req.get(url, timeout=60, headers=_headers)
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
