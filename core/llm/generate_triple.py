from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .ingest_confidence import clip_score, score_ingest_confidence
from .base import LLMBackend
from .ner import Entity

_SYSTEM = """\
You are a knowledge graph extraction assistant. Given a passage of scientific or \
technical text, extract all meaningful relationships as triples.

Return a JSON array of objects with exactly these fields:
  - "subject": subject entity name (string)
  - "subject_type": concise type label (string, e.g. "PERSON", "TECHNOLOGY", "ORGANIZATION", "CONCEPT", "LOCATION")
  - "relation": relationship as a concise verb phrase (string)
  - "object": object entity name (string)
  - "object_type": concise type label (string)
  - "certainty": confidence 0.0-1.0 that this relation is stated or strongly implied in the text (float)
  - "evidence_text": minimal continuous quote from the text that supports the relation (string)

Only extract relations grounded in the provided text.
Use the exact wording from the passage for evidence_text. Do not paraphrase it.
Return only the JSON array. No explanation, no markdown fences."""


def _parse_json(response: str) -> list:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Response was likely truncated — salvage everything up to the last complete object
        last_close = cleaned.rfind("}")
        if last_close > 0:
            try:
                return json.loads(cleaned[: last_close + 1] + "]")
            except json.JSONDecodeError:
                pass
    return []


@dataclass
class EvaluatedTriple:
    subject: Entity
    relation: str
    object_: Entity
    certainty_score: float
    raw_certainty_score: float
    chunk_id: str
    evidence_text: str
    evidence_char_start: int
    evidence_char_end: int
    evidence_alignment_score: float
    entity_anchor_score: float
    evidence_scope_score: float
    confidence_version: str = "v2_evidence_hybrid"


def generate_triples(chunk_id: str, content: str, backend: LLMBackend) -> list[EvaluatedTriple]:
    # Single-pass extraction: one LLM call returns subject+type, relation, object+type,
    # raw certainty, and supporting evidence text.
    # Previously two calls (NER then relation extraction) — halves per-chunk LLM overhead.
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Extract all knowledge graph triples from the following text:\n\n{content}"},
    ]
    response = backend.chat(messages, max_tokens=4096)
    raw = _parse_json(response)

    best_by_key: dict[tuple[str, str, str], EvaluatedTriple] = {}
    for r in raw:
        subj = r.get("subject")
        subj_type = r.get("subject_type")
        rel = r.get("relation")
        obj = r.get("object")
        obj_type = r.get("object_type")
        if not all([subj, subj_type, rel, obj, obj_type]):
            continue  # skip malformed rows
        try:
            raw_certainty = float(r.get("certainty", 0.0))
        except (TypeError, ValueError):
            raw_certainty = 0.0
        evidence_text = str(r.get("evidence_text", "") or "").strip()
        signals = score_ingest_confidence(
            subject=str(subj),
            relation=str(rel),
            object_=str(obj),
            raw_certainty_score=raw_certainty,
            evidence_text=evidence_text,
            chunk_text=content,
        )
        if signals is None:
            continue
        key = (subj, rel, obj)
        candidate = EvaluatedTriple(
            subject=Entity(name=subj, type=subj_type),
            relation=rel,
            object_=Entity(name=obj, type=obj_type),
            certainty_score=signals.ingest_confidence,
            raw_certainty_score=clip_score(raw_certainty),
            chunk_id=chunk_id,
            evidence_text=signals.evidence_text,
            evidence_char_start=signals.evidence_char_start,
            evidence_char_end=signals.evidence_char_end,
            evidence_alignment_score=signals.evidence_alignment_score,
            entity_anchor_score=signals.entity_anchor_score,
            evidence_scope_score=signals.evidence_scope_score,
            confidence_version=signals.confidence_version,
        )
        existing = best_by_key.get(key)
        if existing is None or candidate.certainty_score > existing.certainty_score:
            best_by_key[key] = candidate
    return list(best_by_key.values())
