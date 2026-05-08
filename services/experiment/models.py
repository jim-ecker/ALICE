"""Pydantic models for the experiment evaluation API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class StartSessionRequest(BaseModel):
    participant_id: str
    questionnaire: dict[str, Any]
    questions_per_session: int = 3
    responses_per_question: int = 3
    response_mode: str = "live_virtual_experts"   # or "static_bank"
    force_new_session: bool = False
    seed: Optional[int] = None


class ResumeSessionRequest(BaseModel):
    participant_id: str


class RatingEntry(BaseModel):
    response_id: str
    accuracy: int    # 1–5
    humanness: int   # 1–5
    comments: Optional[str] = None


class IdentificationEntry(BaseModel):
    # Maps expert label (as shown in UI) → alias (A, B, C...) the participant thinks it is
    guesses: dict[str, str]
    confidence: int   # 1–5
    comments: Optional[str] = None


class SaveRatingsRequest(BaseModel):
    ratings: list[RatingEntry]
    identification: IdentificationEntry


class AskLiveRequest(BaseModel):
    question_text: str
    response_ids: Optional[list[str]] = None
