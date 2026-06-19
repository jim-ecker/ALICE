"""FastAPI router for the experiment evaluation workbench."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from services.experiment.models import (
    AskLiveRequest,
    ProfileData,
    ResumeSessionRequest,
    SaveRatingsRequest,
    StartSessionRequest,
)
from services.experiment.session import (
    create_session,
    find_session_for_participant,
    load_agents,
    load_profile,
    load_questions,
    load_questionnaire_schema,
    load_session,
    public_session,
    save_profile,
    save_session,
    session_response,
    submit_session,
)


def _user_from_request(request: Request) -> str:
    return request.headers.get("X-Remote-User", "").strip().lower()


def make_experiment_router(cfg: Any, state: Any) -> APIRouter:
    """Build the experiment APIRouter.

    cfg:   ChatConfig  (experts_dir, top_k_chunks, max_tokens, etc.)
    state: ServiceState (shared llm + embed_client + scoring_cfg)
    """
    router = APIRouter(prefix="/api/experiment", tags=["experiment"])

    # ── Meta ─────────────────────────────────────────────────────────────────

    @router.get("/meta")
    async def get_meta():
        questions = load_questions()
        agents = load_agents()
        schema = load_questionnaire_schema()
        virtual_experts = [a for a in agents if a.get("class") == "virtual_expert"]
        return {
            "questions_available": len(questions),
            "agents_available": len(agents),
            "virtual_experts_available": len(virtual_experts),
            "default_questions_per_session": 5,
            "default_responses_per_question": len(agents),
            "response_modes": ["live", "static_bank"],
            "rating_scale_bounds": {"min": 1, "max": 5},
            "questionnaire_schema": schema,
            "questions": questions,
            "agents": [
                {"agent_id": a["agent_id"], "label": a["label"], "class": a.get("class")}
                for a in agents
            ],
        }

    # ── Profile ───────────────────────────────────────────────────────────────

    @router.get("/profile")
    async def get_profile(request: Request):
        email = _user_from_request(request)
        if not email:
            return {"profile": None}
        profile = await asyncio.to_thread(load_profile, email)
        return {"email": email, "profile": profile}

    @router.put("/profile")
    async def put_profile(request: Request, data: ProfileData):
        email = _user_from_request(request) or "anonymous"
        payload = data.model_dump()
        await asyncio.to_thread(save_profile, email, payload)
        return {"status": "saved", "email": email}

    # ── Session lifecycle ─────────────────────────────────────────────────────

    @router.post("/sessions/start")
    async def start_session(request: Request, req: StartSessionRequest):
        # Prefer email from auth header over request body participant_id
        email = _user_from_request(request)
        participant_id = email or req.participant_id or "anonymous"

        if not req.force_new_session:
            existing = await asyncio.to_thread(find_session_for_participant, participant_id)
            if existing:
                return {"status": "resumed", "session": session_response(existing)}

        session = await asyncio.to_thread(
            _create_session_with_participant, req, participant_id, cfg.experts_dir
        )
        return {"status": "created", "session": session_response(session)}

    @router.post("/sessions/resume")
    async def resume_session(request: Request, req: ResumeSessionRequest):
        email = _user_from_request(request)
        participant_id = email or req.participant_id or "anonymous"
        existing = await asyncio.to_thread(find_session_for_participant, participant_id)
        if not existing:
            raise HTTPException(status_code=404, detail="No in-progress session found for this participant")
        return {"status": "resumed", "session": session_response(existing)}

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok", "session": session_response(session)}

    # ── Per-agent live query ──────────────────────────────────────────────────

    @router.post("/sessions/{session_id}/questions/{q_id}/responses/{response_id}/query")
    async def query_single_response(session_id: str, q_id: str, response_id: str):
        """Query a single agent for one response slot, returning text and server-side elapsed time."""
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["status"] != "in_progress":
            raise HTTPException(status_code=400, detail="Session is not in progress")

        block = next((b for b in session["question_blocks"] if b["q_id"] == q_id), None)
        if not block:
            raise HTTPException(status_code=404, detail=f"Question block {q_id} not found")

        resp = next((r for r in block["responses"] if r["response_id"] == response_id), None)
        if not resp:
            raise HTTPException(status_code=404, detail=f"Response {response_id} not found")

        # Return cached result if already answered
        if resp.get("text"):
            return {"response_id": response_id, "alias": resp["alias"], "text": resp["text"], "cached": True, "elapsed_ms": 0}

        question_text = block["question_text"]
        scoring_cfg = _extract_scoring_cfg(state)

        from services.experiment.live import query_commercial_llm, query_expert, query_generic_llm

        def _run() -> str:
            if resp.get("class") == "virtual_expert" and resp.get("expert_slug"):
                return query_expert(
                    expert_slug=resp["expert_slug"],
                    question=question_text,
                    cfg=cfg,
                    embed_client=state.embed_client,
                    llm=state.llm,
                    scoring_cfg=scoring_cfg,
                )
            elif resp.get("class") == "commercial_llm":
                agents = load_agents()
                agent_cfg = next((a for a in agents if a["agent_id"] == resp["agent_id"]), {})
                return query_commercial_llm(agent_cfg, question_text, cfg.max_tokens)
            else:
                return query_generic_llm(question_text, state.llm, max_tokens=cfg.max_tokens)

        t0 = time.monotonic()
        try:
            text = await asyncio.to_thread(_run)
        except Exception as exc:
            text = f"[Error: {exc}]"
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        resp["text"] = text
        resp["elapsed_ms"] = elapsed_ms
        await asyncio.to_thread(save_session, session)

        return {"response_id": response_id, "alias": resp["alias"], "text": text, "cached": False, "elapsed_ms": elapsed_ms}

    # ── Bulk live query (kept for script use) ─────────────────────────────────

    @router.post("/sessions/{session_id}/questions/{q_id}/ask")
    async def ask_question(session_id: str, q_id: str, req: AskLiveRequest):
        """Query all agents for this question block in sequence."""
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["status"] != "in_progress":
            raise HTTPException(status_code=400, detail="Session is not in progress")

        block = next((b for b in session["question_blocks"] if b["q_id"] == q_id), None)
        if not block:
            raise HTTPException(status_code=404, detail=f"Question block {q_id} not found")

        target_ids = set(req.response_ids) if req.response_ids else None
        question_text = req.question_text or block["question_text"]
        scoring_cfg = _extract_scoring_cfg(state)

        from services.experiment.live import query_commercial_llm, query_expert, query_generic_llm

        def _run_queries():
            agents = load_agents()
            for resp in block["responses"]:
                if target_ids and resp["response_id"] not in target_ids:
                    continue
                if resp.get("text"):
                    continue
                try:
                    if resp.get("class") == "virtual_expert" and resp.get("expert_slug"):
                        text = query_expert(
                            expert_slug=resp["expert_slug"],
                            question=question_text,
                            cfg=cfg,
                            embed_client=state.embed_client,
                            llm=state.llm,
                            scoring_cfg=scoring_cfg,
                        )
                    elif resp.get("class") == "commercial_llm":
                        agent_cfg = next((a for a in agents if a["agent_id"] == resp["agent_id"]), {})
                        text = query_commercial_llm(agent_cfg, question_text, cfg.max_tokens)
                    else:
                        text = query_generic_llm(question_text, state.llm, max_tokens=cfg.max_tokens)
                    resp["text"] = text
                except Exception as exc:
                    resp["text"] = f"[Error querying this response: {exc}]"

        await asyncio.to_thread(_run_queries)
        await asyncio.to_thread(save_session, session)
        return {"status": "answered", "session": session_response(session)}

    # ── Ratings & identification ──────────────────────────────────────────────

    @router.put("/sessions/{session_id}/questions/{q_id}")
    async def save_ratings(session_id: str, q_id: str, req: SaveRatingsRequest):
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["status"] != "in_progress":
            raise HTTPException(status_code=400, detail="Session is not in progress")

        block = next((b for b in session["question_blocks"] if b["q_id"] == q_id), None)
        if not block:
            raise HTTPException(status_code=404, detail=f"Question block {q_id} not found")

        from services.experiment.session import _now
        session["evaluations"][q_id] = {
            "q_id": q_id,
            "saved_at": _now(),
            "ratings": [r.model_dump() for r in req.ratings],
            "identification": req.identification.model_dump(),
        }

        await asyncio.to_thread(save_session, session)
        return {"status": "saved", "session": session_response(session)}

    # ── Navigation ────────────────────────────────────────────────────────────

    @router.post("/sessions/{session_id}/next")
    async def advance_question(session_id: str):
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["status"] != "in_progress":
            raise HTTPException(status_code=400, detail="Session is not in progress")

        total = len(session["question_blocks"])
        if session["current_index"] < total - 1:
            session["current_index"] += 1
        await asyncio.to_thread(save_session, session)
        return {"status": "advanced", "session": session_response(session)}

    # ── Submit ────────────────────────────────────────────────────────────────

    @router.post("/sessions/{session_id}/submit")
    async def submit(session_id: str):
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        export_path = await asyncio.to_thread(submit_session, session_id)
        session = await asyncio.to_thread(load_session, session_id)
        return {"status": "submitted", "session": session_response(session, blind=False), "export_path": export_path}

    @router.get("/sessions/{session_id}/export")
    async def export_session(session_id: str):
        """Download the complete (unblinded) session as JSON."""
        from fastapi.responses import JSONResponse
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return JSONResponse(content=session)

    return router


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_session_with_participant(req, participant_id: str, experts_dir):
    """Wrapper to create a session with an externally-supplied participant_id."""
    import copy
    req_copy = copy.copy(req)
    # We pass participant_id explicitly to session.create_session via the request
    # object; since pydantic models are immutable we set it here via a dict shim.
    from services.experiment.session import (
        load_agents,
        load_questions,
        _build_question_blocks,
        _session_id,
        _now,
        save_session,
    )
    import random

    questions = load_questions()
    agents = load_agents()
    rng = random.Random(req.seed)

    question_blocks = _build_question_blocks(
        questions=questions,
        agents=agents,
        n_questions=req.questions_per_session,
        n_responses=req.responses_per_question,
        response_mode=req.response_mode,
        rng=rng,
        experts_dir=experts_dir,
    )

    session = {
        "session_id": _session_id(),
        "participant_id": participant_id,
        "status": "in_progress",
        "created_at": _now(),
        "updated_at": _now(),
        "completed_at": None,
        "questionnaire": req.questionnaire,
        "settings": {
            "questions_per_session": req.questions_per_session,
            "responses_per_question": req.responses_per_question,
            "response_mode": req.response_mode,
            "seed": req.seed,
        },
        "question_blocks": question_blocks,
        "current_index": 0,
        "evaluations": {},
    }
    save_session(session)
    return session


def _extract_scoring_cfg(state: Any):
    """Pull the ScoringConfig from the WeightedCompositeScorer inside the retriever."""
    try:
        return state.retriever._scorer._cfg
    except AttributeError:
        from core.scoring.composite import ScoringConfig
        return ScoringConfig()
