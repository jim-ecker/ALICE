import json
import re
from dataclasses import dataclass

from .base import LLMBackend
from .ner import Entity

_SYSTEM = """\
You are an information extraction assistant specialising in relation extraction.
Given a list of entities and source text, extract relations between pairs of entities.
Return a JSON array of objects with exactly these fields:
  - "subject": entity name (string)
  - "relation": the relationship expressed as a concise verb phrase (string)
  - "object": entity name (string)
  - "certainty": your confidence that this relation is stated or strongly implied in the text, between 0.0 and 1.0 (float)

Only extract relations that are grounded in the provided text. Return only the JSON array. No explanation, no markdown fences."""


@dataclass
class Triple:
    subject: Entity
    relation: str
    object_: Entity
    certainty_score: float


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


def extract_relations(text: str, entities: list[Entity], backend: LLMBackend) -> list[Triple]:
    entity_list = ", ".join(f"{e.name} ({e.type})" for e in entities)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Entities: {entity_list}\n\nText:\n{text}"},
    ]
    response = backend.chat(messages, max_tokens=4096)
    try:
        raw = _parse_json(response)
    except (json.JSONDecodeError, ValueError):
        return []
    entity_map = {e.name: e for e in entities}
    triples = []
    for r in raw:
        subj = r.get("subject")
        rel = r.get("relation")
        obj = r.get("object")
        if not subj or not rel or not obj:
            continue
        try:
            certainty = float(r.get("certainty", 0.0))
        except (TypeError, ValueError):
            certainty = 0.0
        subject = entity_map.get(subj, Entity(name=subj, type="UNKNOWN"))
        object_ = entity_map.get(obj, Entity(name=obj, type="UNKNOWN"))
        triples.append(Triple(
            subject=subject,
            relation=rel,
            object_=object_,
            certainty_score=certainty,
        ))
    return triples
