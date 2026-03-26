from __future__ import annotations

import json
import re
from dataclasses import dataclass

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

Only extract relations grounded in the provided text.
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
    chunk_id: str


def generate_triples(chunk_id: str, content: str, backend: LLMBackend) -> list[EvaluatedTriple]:
    # Single-pass extraction: one LLM call returns subject+type, relation, object+type, certainty.
    # Previously two calls (NER then relation extraction) — halves per-chunk LLM overhead.
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Extract all knowledge graph triples from the following text:\n\n{content}"},
    ]
    response = backend.chat(messages, max_tokens=4096)
    raw = _parse_json(response)

    seen: set[tuple[str, str, str]] = set()
    triples: list[EvaluatedTriple] = []
    for r in raw:
        subj = r.get("subject")
        subj_type = r.get("subject_type")
        rel = r.get("relation")
        obj = r.get("object")
        obj_type = r.get("object_type")
        if not all([subj, subj_type, rel, obj, obj_type]):
            continue  # skip malformed rows
        try:
            certainty = float(r.get("certainty", 0.0))
        except (TypeError, ValueError):
            certainty = 0.0
        key = (subj, rel, obj)
        if key in seen:
            continue  # deduplicate repeated triples
        seen.add(key)
        triples.append(EvaluatedTriple(
            subject=Entity(name=subj, type=subj_type),
            relation=rel,
            object_=Entity(name=obj, type=obj_type),
            certainty_score=certainty,
            chunk_id=chunk_id,
        ))
    return triples
