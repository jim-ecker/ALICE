"""FastAPI router for the experiment evaluation workbench."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

from services.experiment.models import (
    AskLiveRequest,
    ResumeSessionRequest,
    SaveRatingsRequest,
    StartSessionRequest,
)
from services.experiment.session import (
    create_session,
    find_session_for_participant,
    load_questions,
    load_agents,
    load_questionnaire_schema,
    load_session,
    public_session,
    save_session,
    session_response,
    submit_session,
)


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
            "default_questions_per_session": 3,
            "default_responses_per_question": min(3, len(agents)),
            "response_modes": ["live_virtual_experts", "static_bank"],
            "rating_scale_bounds": {"min": 1, "max": 5},
            "questionnaire_schema": schema,
            "questions": questions,
            "agents": [
                {"agent_id": a["agent_id"], "label": a["label"], "class": a.get("class")}
                for a in agents
            ],
        }

    # ── Session lifecycle ─────────────────────────────────────────────────────

    @router.post("/sessions/start")
    async def start_session(req: StartSessionRequest):
        # Check for existing in-progress session unless force_new
        if not req.force_new_session:
            existing = await asyncio.to_thread(find_session_for_participant, req.participant_id)
            if existing:
                return {"status": "resumed", "session": session_response(existing)}

        session = await asyncio.to_thread(
            create_session, req, cfg.experts_dir
        )
        return {"status": "created", "session": session_response(session)}

    @router.post("/sessions/resume")
    async def resume_session(req: ResumeSessionRequest):
        existing = await asyncio.to_thread(find_session_for_participant, req.participant_id)
        if not existing:
            raise HTTPException(status_code=404, detail="No in-progress session found for this participant")
        return {"status": "resumed", "session": session_response(existing)}

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = await asyncio.to_thread(load_session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok", "session": session_response(session)}

    # ── Live querying ─────────────────────────────────────────────────────────

    @router.post("/sessions/{session_id}/questions/{q_id}/ask")
    async def ask_question(session_id: str, q_id: str, req: AskLiveRequest):
        """Live-query all selected experts for this question block."""
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

        from services.experiment.live import query_expert, query_generic_llm

        # Fetch scoring_cfg from the state (already built by Chat._build_state)
        # We pull it from the scorer inside the retriever
        scoring_cfg = _extract_scoring_cfg(state)

        def _run_queries():
            for resp in block["responses"]:
                if target_ids and resp["response_id"] not in target_ids:
                    continue
                if resp.get("text"):
                    continue  # already answered
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

def _extract_scoring_cfg(state: Any):
    """Pull the ScoringConfig from the WeightedCompositeScorer inside the retriever."""
    try:
        return state.retriever._scorer._cfg
    except AttributeError:
        from core.scoring.composite import ScoringConfig
        return ScoringConfig()
