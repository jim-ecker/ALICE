from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"\S.*?(?:(?<! [A-Z])[.!?](?=\s|$)|\n{2,}|$)", re.S)
_HEDGE_RE = re.compile(
    r"\b(?:may|might|could|suggest|suggests|appears|appear|possible|possibly|likely|approximately)\b",
    re.I,
)


@dataclass(frozen=True)
class EvidenceAlignment:
    evidence_text: str
    evidence_char_start: int
    evidence_char_end: int
    evidence_alignment_score: float
    alignment_method: str
    sentence_text: str


@dataclass(frozen=True)
class IngestConfidenceSignals:
    ingest_confidence: float
    evidence_text: str
    evidence_char_start: int
    evidence_char_end: int
    evidence_alignment_score: float
    entity_anchor_score: float
    evidence_scope_score: float
    raw_certainty_score: float
    confidence_version: str = "v2_evidence_hybrid"


def clip_score(value: float | int | None) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def score_ingest_confidence(
    *,
    subject: str,
    relation: str,
    object_: str,
    raw_certainty_score: float,
    evidence_text: str,
    chunk_text: str,
) -> IngestConfidenceSignals | None:
    del relation  # reserved for future relation-aware heuristics

    evidence = str(evidence_text or "").strip()
    if not evidence:
        return None

    alignment = _align_evidence(
        chunk_text=chunk_text,
        evidence_text=evidence,
        subject=subject,
        object_=object_,
    )
    if alignment is None:
        return None

    entity_anchor_score = _score_entity_anchors(
        aligned_evidence=alignment.evidence_text,
        sentence_text=alignment.sentence_text,
        subject=subject,
        object_=object_,
    )
    if entity_anchor_score < 0.75:
        return None

    scope_score = _score_evidence_scope(alignment.evidence_text)
    raw_score = clip_score(raw_certainty_score)
    ingest_confidence = (
        0.45 * alignment.evidence_alignment_score
        + 0.25 * entity_anchor_score
        + 0.20 * scope_score
        + 0.10 * raw_score
    )

    return IngestConfidenceSignals(
        ingest_confidence=round(ingest_confidence, 6),
        evidence_text=alignment.evidence_text,
        evidence_char_start=alignment.evidence_char_start,
        evidence_char_end=alignment.evidence_char_end,
        evidence_alignment_score=alignment.evidence_alignment_score,
        entity_anchor_score=entity_anchor_score,
        evidence_scope_score=scope_score,
        raw_certainty_score=raw_score,
    )


def _align_evidence(
    *,
    chunk_text: str,
    evidence_text: str,
    subject: str,
    object_: str,
) -> EvidenceAlignment | None:
    exact_matches = _find_exact_matches(chunk_text, evidence_text)
    if exact_matches:
        best = _pick_best_span(chunk_text, exact_matches, subject=subject, object_=object_)
        return _make_alignment(
            chunk_text=chunk_text,
            start=best[0],
            end=best[1],
            score=1.0,
            method="exact",
        )

    normalized_matches = _find_normalized_matches(chunk_text, evidence_text)
    if normalized_matches:
        best = _pick_best_span(chunk_text, normalized_matches, subject=subject, object_=object_)
        return _make_alignment(
            chunk_text=chunk_text,
            start=best[0],
            end=best[1],
            score=1.0,
            method="normalized",
        )

    sentence_spans = _split_sentence_spans(chunk_text)
    fallback = _pick_sentence_fallback(
        chunk_text=chunk_text,
        sentence_spans=sentence_spans,
        evidence_text=evidence_text,
        subject=subject,
        object_=object_,
    )
    if fallback is None:
        return None
    return _make_alignment(
        chunk_text=chunk_text,
        start=fallback[0],
        end=fallback[1],
        score=0.8,
        method="sentence_fallback",
    )


def _find_exact_matches(text: str, target: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(target, start)
        if idx < 0:
            break
        matches.append((idx, idx + len(target)))
        start = idx + 1
    return matches


def _find_normalized_matches(text: str, target: str) -> list[tuple[int, int]]:
    norm_text, index_map = _normalize_with_map(text)
    norm_target, _ = _normalize_with_map(target)
    if not norm_text or not norm_target:
        return []

    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = norm_text.find(norm_target, start)
        if idx < 0:
            break
        orig_start = index_map[idx]
        orig_end = index_map[idx + len(norm_target) - 1] + 1
        matches.append((orig_start, orig_end))
        start = idx + 1
    return matches


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    pending_space = False
    pending_idx = 0

    for idx, ch in enumerate(text):
        if ch.isspace():
            if chars:
                pending_space = True
                pending_idx = idx
            continue
        if pending_space:
            chars.append(" ")
            index_map.append(pending_idx)
            pending_space = False
        chars.append(ch.lower())
        index_map.append(idx)

    return "".join(chars), index_map


def _pick_best_span(
    chunk_text: str,
    spans: list[tuple[int, int]],
    *,
    subject: str,
    object_: str,
) -> tuple[int, int]:
    def _score(span: tuple[int, int]) -> tuple[int, int, int]:
        start, end = span
        aligned = chunk_text[start:end]
        sentence_text = _sentence_covering_span(chunk_text, start, end)
        anchor_rank = _anchor_rank(
            aligned_evidence=aligned,
            sentence_text=sentence_text,
            subject=subject,
            object_=object_,
        )
        return (anchor_rank, -(end - start), -start)

    return max(spans, key=_score)


def _make_alignment(
    *,
    chunk_text: str,
    start: int,
    end: int,
    score: float,
    method: str,
) -> EvidenceAlignment:
    aligned = chunk_text[start:end].strip()
    sentence_text = _sentence_covering_span(chunk_text, start, end)
    return EvidenceAlignment(
        evidence_text=aligned,
        evidence_char_start=start,
        evidence_char_end=end,
        evidence_alignment_score=score,
        alignment_method=method,
        sentence_text=sentence_text,
    )


def _pick_sentence_fallback(
    *,
    chunk_text: str,
    sentence_spans: list[tuple[int, int]],
    evidence_text: str,
    subject: str,
    object_: str,
) -> tuple[int, int] | None:
    evidence_tokens = _token_set(evidence_text)
    candidates: list[tuple[float, int, int, int, int]] = []

    for idx, (start, end) in enumerate(sentence_spans):
        spans = [(start, end)]
        if idx + 1 < len(sentence_spans):
            spans.append((start, sentence_spans[idx + 1][1]))

        for cand_start, cand_end in spans:
            candidate_text = chunk_text[cand_start:cand_end].strip()
            if not (
                _contains_entity(candidate_text, subject)
                and _contains_entity(candidate_text, object_)
            ):
                continue
            overlap = _token_overlap_ratio(evidence_tokens, _token_set(candidate_text))
            candidates.append((overlap, -(cand_end - cand_start), -cand_start, cand_start, cand_end))

    if not candidates:
        return None

    best = max(candidates)
    return best[3], best[4]


def _split_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _sentence_covering_span(text: str, start: int, end: int) -> str:
    for sent_start, sent_end in _split_sentence_spans(text):
        if sent_start <= start < sent_end or sent_start < end <= sent_end:
            return text[sent_start:sent_end].strip()
        if start <= sent_start and sent_end <= end:
            return text[sent_start:sent_end].strip()
    return text[start:end].strip()


def _score_entity_anchors(
    *,
    aligned_evidence: str,
    sentence_text: str,
    subject: str,
    object_: str,
) -> float:
    return float(
        _anchor_rank(
            aligned_evidence=aligned_evidence,
            sentence_text=sentence_text,
            subject=subject,
            object_=object_,
        )
    )


def _anchor_rank(
    *,
    aligned_evidence: str,
    sentence_text: str,
    subject: str,
    object_: str,
) -> float:
    subj_in_evidence = _contains_entity(aligned_evidence, subject)
    obj_in_evidence = _contains_entity(aligned_evidence, object_)
    if subj_in_evidence and obj_in_evidence:
        return 1.0

    subj_in_sentence = _contains_entity(sentence_text, subject)
    obj_in_sentence = _contains_entity(sentence_text, object_)
    if (subj_in_evidence and obj_in_sentence) or (obj_in_evidence and subj_in_sentence):
        return 0.75
    return 0.0


def _score_evidence_scope(evidence_text: str) -> float:
    text = evidence_text.strip()
    sentence_count = len(_split_sentence_spans(text)) or 1
    if sentence_count == 1 and len(text) <= 320:
        score = 1.0
    elif sentence_count <= 2 or len(text) <= 500:
        score = 0.8
    else:
        score = 0.6
    if _HEDGE_RE.search(text):
        score *= 0.85
    return round(score, 6)


def _contains_entity(text: str, entity: str) -> bool:
    if not text or not entity:
        return False
    norm_text, _ = _normalize_with_map(text)
    norm_entity, _ = _normalize_with_map(entity)
    return bool(norm_entity) and norm_entity in norm_text


def _token_set(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


def _token_overlap_ratio(reference: set[str], candidate: set[str]) -> float:
    if not reference:
        return 0.0
    return len(reference & candidate) / len(reference)
