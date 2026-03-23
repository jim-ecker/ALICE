import json
import re
from dataclasses import dataclass

from .base import LLMBackend

_SYSTEM = """\
You are an information extraction assistant. Extract all named entities from scientific and technical text.
Always include every person's full name mentioned in the text.
For organizations and research centers, always extract the most specific and complete name
available in the text (e.g. 'NASA Langley Research Center' rather than 'NASA' or 'LaRC').
Return a JSON array of objects with exactly these fields:
  - "name": the entity as it appears in the text (string)
  - "type": a concise entity type label (string, e.g. "PERSON", "TECHNOLOGY", "ORGANIZATION", "CONCEPT", "LOCATION")

Return only the JSON array. No explanation, no markdown fences."""


@dataclass
class Entity:
    name: str
    type: str


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


def extract_entities(text: str, backend: LLMBackend) -> list[Entity]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Extract all named entities from the following text:\n\n{text}"},
    ]
    response = backend.chat(messages, max_tokens=4096)
    try:
        raw = _parse_json(response)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[ner] JSON parse error ({exc}); response preview: {response[:300]!r}")
        return []
    return [Entity(name=e["name"], type=e["type"]) for e in raw if e.get("name") and e.get("type")]
