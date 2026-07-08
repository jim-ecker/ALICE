from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ingest_confidence import clip_score, score_ingest_confidence
from .base import LLMBackend
from .ner import Entity

if TYPE_CHECKING:
    from core.trust.config import TrustConfig

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

Academic paper conventions: In paper headers, an author name followed by an institution \
name on the next line implies an "affiliated with" relation. A footnote of the form \
"* Role Title, Institution" implies the author "has role" Role Title and "affiliated with" \
Institution. Extract these implicit relations.

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
    # v3 measurable trust signals — None on legacy rows and when [trust] is disabled
    ptrue_score: float | None = None
    ptrue_method: str | None = None
    ptrue_samples: int | None = None
    nli_entailment_score: float | None = None
    nli_contradiction_score: float | None = None
    nli_model: str | None = None
    self_consistency_freq: float | None = None
    self_consistency_count: int | None = None
    self_consistency_runs: int | None = None


def _build_messages(content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Extract all knowledge graph triples from the following text:\n\n{content}"},
    ]


def generate_triples(
    chunk_id: str,
    content: str,
    backend: LLMBackend,
    trust_cfg: TrustConfig | None = None,
) -> list[EvaluatedTriple]:
    # Single-pass extraction: one LLM call returns subject+type, relation, object+type,
    # raw certainty, and supporting evidence text.
    # Previously two calls (NER then relation extraction) — halves per-chunk LLM overhead.
    messages = _build_messages(content)
    response = backend.chat(messages, max_tokens=4096)
    raw = _parse_json(response)

    best_by_key: dict[tuple[str, str, str], EvaluatedTriple] = {}
    sentence_by_key: dict[tuple[str, str, str], str] = {}
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
            sentence_by_key[key] = signals.sentence_text

    triples = list(best_by_key.values())
    if trust_cfg is not None and trust_cfg.enabled and triples:
        _attach_trust_signals(
            triples,
            sentence_by_key=sentence_by_key,
            messages=messages,
            backend=backend,
            trust_cfg=trust_cfg,
        )
    return triples


def _attach_trust_signals(
    triples: list[EvaluatedTriple],
    *,
    sentence_by_key: dict[tuple[str, str, str], str],
    messages: list[dict[str, str]],
    backend: LLMBackend,
    trust_cfg: TrustConfig,
) -> None:
    from core.trust.consistency import canonical_key, count_across_runs
    from core.trust.signals import compute_evidence_signals

    # Self-consistency: k rerun extractions, parse-only (no scoring/dedup) —
    # we just need which canonical triples reappear.
    consistency = None
    if trust_cfg.self_consistency_enabled and trust_cfg.self_consistency_k > 0:
        reruns = backend.sample_n(
            messages,
            n=trust_cfg.self_consistency_k,
            temperature=trust_cfg.self_consistency_temperature,
            max_tokens=4096,
        )
        rerun_key_sets: list[set[tuple[str, str, str]]] = []
        for rerun in reruns:
            keys: set[tuple[str, str, str]] = set()
            for row in _parse_json(rerun):
                subj, rel, obj = row.get("subject"), row.get("relation"), row.get("object")
                if subj and rel and obj:
                    keys.add(canonical_key(str(subj), str(rel), str(obj)))
            rerun_key_sets.append(keys)
        base_keys = {
            canonical_key(t.subject.name, t.relation, t.object_.name) for t in triples
        }
        consistency = count_across_runs(base_keys, rerun_key_sets)

    for triple in triples:
        key = (triple.subject.name, triple.relation, triple.object_.name)
        evidence_sentence = sentence_by_key.get(key) or triple.evidence_text
        signals = compute_evidence_signals(
            backend,
            subject=triple.subject.name,
            relation=triple.relation,
            object_=triple.object_.name,
            evidence_sentence=evidence_sentence,
            cfg=trust_cfg,
        )
        triple.ptrue_score = signals.ptrue_score
        triple.ptrue_method = signals.ptrue_method
        triple.ptrue_samples = signals.ptrue_samples
        triple.nli_entailment_score = signals.nli_entailment_score
        triple.nli_contradiction_score = signals.nli_contradiction_score
        triple.nli_model = signals.nli_model
        if consistency is not None:
            result = consistency[canonical_key(triple.subject.name, triple.relation, triple.object_.name)]
            triple.self_consistency_freq = result.freq
            triple.self_consistency_count = result.count
            triple.self_consistency_runs = result.runs
        triple.confidence_version = "v3_signals"
