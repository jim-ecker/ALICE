from __future__ import annotations

import math
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


@dataclass
class EvidencePath:
    """A 1- or 2-hop KG path scored by geometric mean of edge trust.

    Each edge in `edges` is (relation, target_entity, certainty_score, chunk_id).
    `path_trust` = geometric mean of the certainty scores — a measure of how
    well every step in the chain is grounded in source text evidence.
    """
    anchor: str
    edges: list[tuple[str, str, float, str]]  # (relation, target, certainty, chunk_id)
    path_trust: float

    @property
    def chunk_ids(self) -> list[str]:
        return [chunk_id for _, _, _, chunk_id in self.edges]


def _normalize_entity_name(name: str) -> str:
    """Collapse OCR artifact spaces: 'A ssemblers' → 'Assemblers'."""
    return re.sub(r'\b([A-Za-z]) ([a-z])', r'\1\2', name)


def _strip_middle_initials(name: str) -> str:
    """Remove middle initials for fuzzy name matching.

    'James E. Ecker' → 'James Ecker'
    'C. L. Taylor'   → 'C. L. Taylor' (not touched — all parts are initials)
    """
    tokens = name.split()
    filtered = [t for t in tokens if not re.match(r'^[A-Za-z]\.$', t)]
    # Only return stripped version if we removed something and at least two tokens remain
    if len(filtered) >= 2 and len(filtered) < len(tokens):
        return " ".join(filtered)
    return name


def _match_anchor_entities(
    conn: kuzu.Connection,
    query_text: str,
    min_name_len: int = 10,
) -> list[str]:
    """Return entity names from the KG that appear in the query text.

    Uses a four-pass fuzzy matching strategy: exact-word → long-name substring
    → word-level fallback → prefix fallback.
    """
    query_lower = query_text.lower()
    query_words = set(re.sub(r'[^\w\s]', '', query_lower).split())

    entity_result = conn.execute("MATCH (e:Entity) RETURN e.name")
    all_long_names: list[str] = []
    all_names: list[str] = []
    while entity_result.has_next():
        row = entity_result.get_next()
        name: str = row[0] or ""
        all_names.append(name)
        if len(name) >= min_name_len:
            all_long_names.append(name)

    _MIN_EXACT_WORD_LEN = 8
    seen: set[str] = set()
    matched_names: list[str] = []
    for n in all_names:
        if len(n) >= _MIN_EXACT_WORD_LEN and n.lower() in query_words:
            matched_names.append(n)
            seen.add(n)

    for n in all_long_names:
        if n not in seen and (
            n.lower() in query_lower
            or _strip_middle_initials(n).lower() in query_lower
        ):
            matched_names.append(n)
            seen.add(n)

    if not matched_names:
        matched_names = [
            n for n in all_long_names
            if any(w in query_words for w in n.lower().split() if len(w) >= 4)
        ]

    if not matched_names:
        matched_names = [
            n for n in all_long_names
            if n not in seen and any(
                re.sub(r"[^\w]", "", tok).startswith(qw)
                for tok in n.lower().split()
                for qw in query_words
                if len(qw) >= 5 and len(tok) > len(qw)
            )
        ]

    return matched_names


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
    matched_names = _match_anchor_entities(conn, query_text, min_name_len)

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


def find_trust_paths(
    conn: kuzu.Connection,
    anchor_entities: list[str],
    max_paths: int = 50,
) -> list[EvidencePath]:
    """Find 1- and 2-hop KG paths from anchor entities, scored by geometric mean of edge trust.

    This is Trust-Propagated Path Retrieval (TPPR): a chain is ranked by the
    weakest evidentiary link, not by graph centrality or semantic proximity.
    """
    if not anchor_entities:
        return []

    anchors = anchor_entities[:5]
    one_hop: list[EvidencePath] = []
    two_hop: list[EvidencePath] = []

    res = conn.execute(
        "MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity) WHERE s.name IN $anchors "
        "RETURN s.name, r.relation, r.certainty_score, r.chunk_id, e.name",
        parameters={"anchors": anchors},
    )
    while res.has_next():
        row = res.get_next()
        anchor, rel, cert, cid, target = row[0], row[1], float(row[2] or 0.0), row[3], row[4]
        if cid:
            one_hop.append(EvidencePath(anchor=anchor, edges=[(rel, target, cert, cid)], path_trust=cert))

    res = conn.execute(
        "MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity) WHERE e.name IN $anchors "
        "RETURN s.name, r.relation, r.certainty_score, r.chunk_id, e.name",
        parameters={"anchors": anchors},
    )
    while res.has_next():
        row = res.get_next()
        src, rel, cert, cid, anchor = row[0], row[1], float(row[2] or 0.0), row[3], row[4]
        if cid:
            one_hop.append(EvidencePath(anchor=anchor, edges=[(rel, src, cert, cid)], path_trust=cert))

    res = conn.execute(
        "MATCH (s:Entity)-[r1:RELATES_TO]->(m:Entity)-[r2:RELATES_TO]->(e:Entity) "
        "WHERE s.name IN $anchors AND s.name <> e.name "
        "RETURN s.name, r1.relation, r1.certainty_score, r1.chunk_id, "
        "m.name, r2.relation, r2.certainty_score, r2.chunk_id, e.name",
        parameters={"anchors": anchors},
    )
    while res.has_next():
        row = res.get_next()
        anchor = row[0]
        rel1, cert1, cid1 = row[1], float(row[2] or 0.0), row[3]
        mid = row[4]
        rel2, cert2, cid2 = row[5], float(row[6] or 0.0), row[7]
        if cid1 and cid2:
            two_hop.append(EvidencePath(
                anchor=anchor,
                edges=[(rel1, mid, cert1, cid1), (rel2, row[8], cert2, cid2)],
                path_trust=math.sqrt(cert1 * cert2),
            ))

    # Cap each hop tier separately so 2-hop paths aren't crowded out by 1-hop
    # paths that happen to share the same trust score (common when certainty=1.0).
    half = max_paths // 2
    one_hop.sort(key=lambda p: p.path_trust, reverse=True)
    two_hop.sort(key=lambda p: p.path_trust, reverse=True)
    paths = one_hop[:half] + two_hop[:max_paths - half]
    paths.sort(key=lambda p: p.path_trust, reverse=True)
    return paths


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
        subj = _normalize_entity_name(row[0])
        obj = _normalize_entity_name(row[3])
        # Skip tautological triples (subject == object) — these are extraction artifacts
        if subj.lower() == obj.lower():
            continue
        triples.append(
            RetrievedTriple(
                subject=subj,
                subject_type=row[1],
                relation=row[2],
                object_=obj,
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
