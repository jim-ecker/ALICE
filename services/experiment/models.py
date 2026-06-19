"""Pydantic models for the experiment evaluation API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class StartSessionRequest(BaseModel):
    participant_id: str = "anonymous"
    questionnaire: dict[str, Any] = {}
    questions_per_session: int = 5
    responses_per_question: int = 6
    response_mode: str = "live"
    force_new_session: bool = False
    seed: Optional[int] = None


class ResumeSessionRequest(BaseModel):
    participant_id: str = "anonymous"


class RatingEntry(BaseModel):
    response_id: str
    accuracy: int    # 1–5
    humanness: int   # 1–5
    comments: Optional[str] = None


class IdentificationEntry(BaseModel):
    virtual_d_alias: str    # alias (A–F) the participant believes is VirtualD (B. Danette Allen)
    virtual_n_alias: str    # alias (A–F) the participant believes is VirtualN (Natalia Alexandrov)
    confidence: int         # 1–5
    comments: Optional[str] = None


class SaveRatingsRequest(BaseModel):
    ratings: list[RatingEntry]
    identification: IdentificationEntry


class AskLiveRequest(BaseModel):
    question_text: Optional[str] = None
    response_ids: Optional[list[str]] = None


class ProfileData(BaseModel):
    # Professional background
    role_title: str = ""
    organization: str = ""
    highest_degree: str = ""
    degree_field: str = ""
    years_at_nasa: Optional[int] = None
    years_in_domain: Optional[int] = None
    years_in_aerospace: Optional[int] = None
    # Familiarity with key personnel
    years_known_d: Optional[int] = None
    years_known_n: Optional[int] = None
    interaction_freq_d: str = ""
    interaction_freq_n: str = ""
    papers_coauthored_d: Optional[int] = None
    papers_coauthored_n: Optional[int] = None
    projects_with_d_or_n: str = ""
    # Domain familiarity
    familiarity_alice: Optional[int] = None      # 1–5
    familiarity_d_research: Optional[int] = None  # 1–5
    familiarity_n_research: Optional[int] = None  # 1–5
    papers_read: str = ""
    # Optional demographics
    age_range: str = ""
    gender: str = ""
    # Communication style
    familiarity_d_comms: Optional[int] = None    # 1–5
    familiarity_n_comms: Optional[int] = None    # 1–5
