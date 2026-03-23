from dataclasses import dataclass

from .base import LLMBackend
from .ner import Entity, extract_entities
from .relations import extract_relations


@dataclass
class EvaluatedTriple:
    subject: Entity
    relation: str
    object_: Entity
    certainty_score: float
    chunk_id: str


def generate_triples(chunk_id: str, content: str, backend: LLMBackend) -> list[EvaluatedTriple]:
    entities = extract_entities(content, backend)
    if not entities:
        return []
    triples = extract_relations(content, entities, backend)
    return [
        EvaluatedTriple(
            subject=t.subject,
            relation=t.relation,
            object_=t.object_,
            certainty_score=t.certainty_score,
            chunk_id=chunk_id,
        )
        for t in triples
    ]
