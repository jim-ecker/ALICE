#!/usr/bin/env python3
"""Run LLM extraction on all unextracted chunks for a document already committed to the graph.

Usage:
    uv run python scripts/extract_doc.py "Goal Detection"
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import kuzu
from core.llm.config import resolve_config
from core.llm.factory import create_backend
from core.llm.generate_triple import generate_triples

DB_PATH = REPO_ROOT / "services/chat/data/chat.db"

if len(sys.argv) < 2:
    print("Usage: uv run python scripts/extract_doc.py <title_fragment>")
    sys.exit(1)

title_fragment = sys.argv[1]

db = kuzu.Database(str(DB_PATH))
conn = kuzu.Connection(db)

r = conn.execute(
    "MATCH (d:Document) WHERE d.title CONTAINS $frag RETURN d.id, d.title",
    parameters={"frag": title_fragment},
)
if not r.has_next():
    print(f"No document found containing '{title_fragment}'")
    sys.exit(1)
row = r.get_next()
doc_id, doc_title = row[0], row[1]
print(f"Document: {doc_title}")

r2 = conn.execute(
    "MATCH (c:Chunk) WHERE c.document_id = $id RETURN c.id, c.content",
    parameters={"id": doc_id},
)
all_chunks = []
while r2.has_next():
    row = r2.get_next()
    all_chunks.append((row[0], row[1]))
print(f"{len(all_chunks)} total chunks")

r3 = conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN DISTINCT r.chunk_id")
extracted_ids = set()
while r3.has_next():
    extracted_ids.add(r3.get_next()[0])

to_extract = [(cid, content) for cid, content in all_chunks if cid not in extracted_ids]
print(f"{len(to_extract)} unextracted chunks\n")

cfg = resolve_config(
    cli_model=None, cli_backend=None, cli_base_url=None,
    cli_api_key=None, cli_workers=None, start_dir=REPO_ROOT,
)
backend = create_backend(cfg)

total_triples = 0
for i, (chunk_id, content) in enumerate(to_extract, 1):
    triples = generate_triples(chunk_id, content, backend)
    n = len(triples)
    total_triples += n

    if triples:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        # Write each triple
        for t in triples:
            try:
                conn.execute(
                    "MERGE (e:Entity {name: $name}) SET e.type = $type",
                    parameters={"name": t.subject.name, "type": t.subject.type},
                )
                conn.execute(
                    "MERGE (e:Entity {name: $name}) SET e.type = $type",
                    parameters={"name": t.object_.name, "type": t.object_.type},
                )
                conn.execute(
                    """
                    MATCH (e1:Entity {name: $s}), (e2:Entity {name: $o})
                    CREATE (e1)-[r:RELATES_TO {
                        relation: $rel,
                        certainty_score: $cert,
                        chunk_id: $cid,
                        raw_certainty_score: $raw,
                        evidence_text: $ev,
                        evidence_char_start: $cs,
                        evidence_char_end: $ce,
                        evidence_alignment_score: $ea,
                        entity_anchor_score: $anc,
                        evidence_scope_score: $sc,
                        confidence_version: $ver
                    }]->(e2)
                    """,
                    parameters={
                        "s": t.subject.name, "o": t.object_.name,
                        "rel": t.relation,
                        "cert": t.certainty_score,
                        "cid": chunk_id,
                        "raw": t.raw_certainty_score or 0.0,
                        "ev": t.evidence_text or "",
                        "cs": t.evidence_char_start or 0,
                        "ce": t.evidence_char_end or 0,
                        "ea": t.evidence_alignment_score or 0.0,
                        "anc": t.entity_anchor_score or 0.0,
                        "sc": t.evidence_scope_score or 0.0,
                        "ver": t.confidence_version or "v2_evidence_hybrid",
                    },
                )
            except Exception as e:
                print(f"    write error: {e}")
        # Mark chunk extracted
        conn.execute(
            "MATCH (c:Chunk {id: $cid}) SET c.extracted_at = $ts",
            parameters={"cid": chunk_id, "ts": ts},
        )

    print(f"[{i}/{len(to_extract)}] {n:2d} triples | {repr(content[:70])}")

print(f"\nDone. {total_triples} triples written.")
