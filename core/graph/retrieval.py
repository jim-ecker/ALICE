from __future__ import annotations

import re
from dataclasses import dataclass

import kuzu


@dataclass
class CitationChunk:
    chunk_id: str
    content: str
    section_heading: str | None
    page_number: int | None
    document_title: str
    document_id: str
    document_url: str | None


@dataclass
class RetrievedTriple:
    subject: str
    subject_type: str
    relation: str
    object_: str
    object_type: str
    certainty_score: float
    chunk_id: str
    raw_certainty_score: float | None = None
    evidence_text: str | None = None
    evidence_char_start: int | None = None
    evidence_char_end: int | None = None
    evidence_alignment_score: float | None = None
    entity_anchor_score: float | None = None
    evidence_scope_score: float | None = None
    confidence_version: str | None = None


@dataclass
class RetrievalResult:
    chunks: list[CitationChunk]
    triples: list[RetrievedTriple]


def _normalize_entity_name(name: str) -> str:
    """Collapse OCR artifact spaces: 'A ssemblers' → 'Assemblers'."""
    return re.sub(r'\b([A-Za-z]) ([a-z])', r'\1\2', name)


def find_entity_chunks(
    conn: kuzu.Connection,
    query_text: str,
    min_name_len: int = 10,
    hop_depth: int = 1,
    max_hop2_entities: int = 20,
) -> list[str]:
    """Find chunk IDs whose triples involve entities mentioned in the query text.

    Walks the entity table and returns chunk_ids for any (s, r, o) edge where
    the entity name appears as a substring of the query (case-insensitive), OR
    any word (≥4 chars) of the entity name appears as a whole word in the query.
    Short names (< min_name_len chars) are skipped to avoid false positives.

    The word-level fallback lets short queries like "where is Langley located"
    match multi-word entities like "NASA Langley Research Center" via the word
    "langley", without lowering min_name_len (which would re-admit single-word
    noise entities like 'Robot', 'Research', 'Center').
    """
    query_lower = query_text.lower()
    query_words = set(query_lower.split())

    # Collect entity names that appear in the query
    entity_result = conn.execute("MATCH (e:Entity) RETURN e.name")
    all_long_names: list[str] = []
    while entity_result.has_next():
        row = entity_result.get_next()
        name: str = row[0] or ""
        if len(name) >= min_name_len:
            all_long_names.append(name)

    # Primary pass: full entity name is a substring of the query
    matched_names = [n for n in all_long_names if n.lower() in query_lower]

    # Fallback: if no substring matches, try word-level — any significant word
    # (≥4 chars) of the entity name appears as an exact word in the query.
    # This lets short queries like "where is Langley located" match multi-word
    # entities like "NASA Langley Research Center" via the word "langley", without
    # polluting queries that already have substring matches (e.g. queries that
    # explicitly name "Langley Research Center" would trigger "research" and
    # "center" as word matches, inflating the candidate set).
    if not matched_names:
        matched_names = [
            n for n in all_long_names
            if any(w in query_words for w in n.lower().split() if len(w) >= 4)
        ]

    if not matched_names:
        return []

    # For each matched entity, collect chunk_ids from outgoing and incoming edges
    chunk_ids: set[str] = set()
    for name in matched_names:
        out = conn.execute(
            "MATCH (e:Entity {name: $n})-[r:RELATES_TO]->() RETURN r.chunk_id",
            parameters={"n": name},
        )
        while out.has_next():
            cid = out.get_next()[0]
            if cid:
                chunk_ids.add(cid)

        inc = conn.execute(
            "MATCH ()-[r:RELATES_TO]->(e:Entity {name: $n}) RETURN r.chunk_id",
            parameters={"n": name},
        )
        while inc.has_next():
            cid = inc.get_next()[0]
            if cid:
                chunk_ids.add(cid)

    if hop_depth < 2 or not chunk_ids:
        return list(chunk_ids)

    # Hop-2: find frequently-appearing new entities in hop-1 chunks' triples
    hop1_ids = list(chunk_ids)
    triple_result = conn.execute(
        "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) WHERE r.chunk_id IN $ids "
        "RETURN e1.name, e2.name",
        parameters={"ids": hop1_ids},
    )
    already_matched = {n.lower() for n in matched_names}
    entity_freq: dict[str, int] = {}
    while triple_result.has_next():
        row = triple_result.get_next()
        for name in (row[0], row[1]):
            if name and name.lower() not in already_matched:
                entity_freq[name] = entity_freq.get(name, 0) + 1

    top_new = sorted(entity_freq, key=entity_freq.__getitem__, reverse=True)[:max_hop2_entities]
    for name in top_new:
        out = conn.execute(
            "MATCH (e:Entity {name: $n})-[r:RELATES_TO]->() RETURN r.chunk_id",
            parameters={"n": name},
        )
        while out.has_next():
            cid = out.get_next()[0]
            if cid:
                chunk_ids.add(cid)
        inc = conn.execute(
            "MATCH ()-[r:RELATES_TO]->(e:Entity {name: $n}) RETURN r.chunk_id",
            parameters={"n": name},
        )
        while inc.has_next():
            cid = inc.get_next()[0]
            if cid:
                chunk_ids.add(cid)

    return list(chunk_ids)


def retrieve_context(conn: kuzu.Connection, chunk_ids: list[str]) -> RetrievalResult:
    """Fetch CitationChunk and RetrievedTriple records for the given chunk IDs."""
    if not chunk_ids:
        return RetrievalResult(chunks=[], triples=[])

    # Chunks + document title
    chunk_result = conn.execute(
        """
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
        WHERE c.id IN $ids
        RETURN c.id, c.content, c.section_heading, c.page_number, d.title, d.id, d.source_url
        """,
        parameters={"ids": chunk_ids},
    )
    chunks: list[CitationChunk] = []
    while chunk_result.has_next():
        row = chunk_result.get_next()
        chunks.append(
            CitationChunk(
                chunk_id=row[0],
                content=row[1],
                section_heading=row[2],
                page_number=row[3],
                document_title=row[4] or "",
                document_id=row[5],
                document_url=row[6] or None,
            )
        )

    # Triples whose chunk_id falls in the set
    triple_result = conn.execute(
        """
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        WHERE r.chunk_id IN $ids
        RETURN e1.name, e1.type, r.relation, e2.name, e2.type, r.certainty_score, r.chunk_id,
               r.raw_certainty_score, r.evidence_text, r.evidence_char_start, r.evidence_char_end,
               r.evidence_alignment_score, r.entity_anchor_score, r.evidence_scope_score, r.confidence_version
        """,
        parameters={"ids": chunk_ids},
    )
    triples: list[RetrievedTriple] = []
    while triple_result.has_next():
        row = triple_result.get_next()
        triples.append(
            RetrievedTriple(
                subject=_normalize_entity_name(row[0]),
                subject_type=row[1],
                relation=row[2],
                object_=_normalize_entity_name(row[3]),
                object_type=row[4],
                certainty_score=float(row[5]),
                chunk_id=row[6],
                raw_certainty_score=float(row[7]) if row[7] is not None else None,
                evidence_text=row[8] or None,
                evidence_char_start=int(row[9]) if row[9] is not None else None,
                evidence_char_end=int(row[10]) if row[10] is not None else None,
                evidence_alignment_score=float(row[11]) if row[11] is not None else None,
                entity_anchor_score=float(row[12]) if row[12] is not None else None,
                evidence_scope_score=float(row[13]) if row[13] is not None else None,
                confidence_version=row[14] or None,
            )
        )

    return RetrievalResult(chunks=chunks, triples=triples)
