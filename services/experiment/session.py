"""Session management for the experiment evaluation workbench.

Sessions are stored as JSON files in study/data/sessions/.
Completed sessions are appended to study/data/evaluations.jsonl.
"""
from __future__ import annotations

import json
import random
import secrets
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = _REPO_ROOT / "study"
SESSIONS_DIR = STUDY_DIR / "data" / "sessions"
EVALUATIONS_FILE = STUDY_DIR / "data" / "evaluations.jsonl"
CONFIG_DIR = STUDY_DIR / "config"


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_questions() -> list[dict]:
    path = CONFIG_DIR / "questions.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


def load_agents() -> list[dict]:
    path = CONFIG_DIR / "agents.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("agents", [])


def load_questionnaire_schema() -> list[dict]:
    path = CONFIG_DIR / "questionnaire.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_response_bank() -> list[dict]:
    path = CONFIG_DIR / "response_bank.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("responses", [])


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def _session_id() -> str:
    return "exp-" + secrets.token_hex(6)


def _response_id() -> str:
    return "resp-" + secrets.token_hex(6)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_question_blocks(
    questions: list[dict],
    agents: list[dict],
    n_questions: int,
    n_responses: int,
    response_mode: str,
    rng: random.Random,
    experts_dir: Path | None = None,
) -> list[dict]:
    """Build randomized, aliased question blocks."""
    from services.experts.manager import ExpertRegistry

    # Filter agents: virtual_expert requires an existing expert DB
    available_agents: list[dict] = []
    for agent in agents:
        if agent.get("class") == "virtual_expert":
            if experts_dir is not None:
                registry = ExpertRegistry(experts_dir)
                meta = registry.get(agent["expert_slug"])
                from services.experts.paths import build_expert_paths
                if meta is None:
                    continue
                paths = build_expert_paths(experts_dir, agent["expert_slug"])
                if not paths.db_path.exists():
                    continue
            available_agents.append(agent)
        else:
            available_agents.append(agent)

    if not available_agents:
        return []

    bank = load_response_bank()

    selected_questions = rng.sample(questions, min(n_questions, len(questions)))
    aliases = list(string.ascii_uppercase)

    blocks: list[dict] = []
    for q in selected_questions:
        selected_agents = rng.sample(available_agents, min(n_responses, len(available_agents)))
        rng.shuffle(selected_agents)

        responses: list[dict] = []
        for i, agent in enumerate(selected_agents):
            alias = aliases[i]
            text = ""
            if response_mode == "static_bank":
                match = next(
                    (r for r in bank if r["q_id"] == q["q_id"] and r["agent_id"] == agent["agent_id"]),
                    None,
                )
                text = match["response_text"] if match else ""

            responses.append({
                "response_id": _response_id(),
                "alias": alias,
                "agent_id": agent["agent_id"],
                "agent_label": agent["label"],
                "class": agent.get("class", "generic_llm"),
                "expert_slug": agent.get("expert_slug"),
                "text": text,
                "chat_history": [],
            })

        blocks.append({
            "q_id": q["q_id"],
            "question_text": q["canonical"],
            "responses": responses,
        })

    return blocks


def create_session(
    req,  # StartSessionRequest
    experts_dir: Path | None = None,
) -> dict:
    """Create a new session with randomized question blocks."""
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

    session: dict[str, Any] = {
        "session_id": _session_id(),
        "participant_id": req.participant_id,
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


def load_session(session_id: str) -> Optional[dict]:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_session_for_participant(participant_id: str) -> Optional[dict]:
    """Find the most recent in-progress session for a participant."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if s.get("participant_id") == participant_id and s.get("status") == "in_progress":
            candidates.append(s)
    if not candidates:
        return None
    # Return most recently updated
    return max(candidates, key=lambda s: s.get("updated_at", ""))


def save_session(session: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session["updated_at"] = _now()
    path = SESSIONS_DIR / f"{session['session_id']}.json"
    path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")


def submit_session(session_id: str) -> str:
    """Finalize a session and append to evaluations.jsonl."""
    session = load_session(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    session["status"] = "completed"
    session["completed_at"] = _now()
    save_session(session)

    EVALUATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(session, ensure_ascii=False) + "\n")

    return str(EVALUATIONS_FILE)


# ---------------------------------------------------------------------------
# Blinding: strip internal identifiers before returning to frontend
# ---------------------------------------------------------------------------

def public_session(session: dict) -> dict:
    """Return a copy with agent_id and expert_slug stripped (blinding)."""
    import copy
    s = copy.deepcopy(session)
    for block in s.get("question_blocks", []):
        for resp in block.get("responses", []):
            resp.pop("agent_id", None)
            resp.pop("agent_label", None)
            resp.pop("expert_slug", None)
    return s


def _progress(session: dict) -> dict:
    total = len(session.get("question_blocks", []))
    current = session.get("current_index", 0)
    completed = len(session.get("evaluations", {}))
    return {
        "current_index": current,
        "total_questions": total,
        "completed_questions": completed,
        "percent_complete": round(completed / total * 100, 1) if total else 0.0,
    }


def session_response(session: dict, blind: bool = True) -> dict:
    """Build the full session payload for API responses."""
    s = public_session(session) if blind else session
    blocks = s.get("question_blocks", [])
    idx = s.get("current_index", 0)
    current_block = blocks[idx] if idx < len(blocks) else None

    return {
        "session_id": s["session_id"],
        "participant_id": s["participant_id"],
        "status": s["status"],
        "created_at": s["created_at"],
        "updated_at": s["updated_at"],
        "completed_at": s.get("completed_at"),
        "questionnaire": s.get("questionnaire"),
        "settings": s.get("settings"),
        "progress": _progress(s),
        "current_question": current_block,
        "evaluations": s.get("evaluations", {}),
        "ready_to_submit": len(s.get("evaluations", {})) >= len(blocks) and len(blocks) > 0,
    }
