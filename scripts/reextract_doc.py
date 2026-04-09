#!/usr/bin/env python3
"""Delete a document from the graph and re-ingest it from NTRS.

Usage:
    uv run python3 scripts/reextract_doc.py "Toward Trustworthy" "ATTRACTOR" langley
"""
import sys
import kuzu
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "services/chat/data/chat.db"


def delete_document(title_fragment: str) -> str:
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
    print(f"Deleting: {doc_title}")
    print(f"      ID: {doc_id}")

    # Collect chunk IDs so we can delete their triples too
    r2 = conn.execute(
        "MATCH (d:Document {id: $id})-[:CONTAINS]->(c:Chunk) RETURN c.id",
        parameters={"id": doc_id},
    )
    chunk_ids = []
    while r2.has_next():
        chunk_ids.append(r2.get_next()[0])
    print(f"  {len(chunk_ids)} chunks found")

    # Delete triples referencing these chunks (orphaned edges block re-extraction)
    if chunk_ids:
        conn.execute(
            "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) WHERE r.chunk_id IN $ids DELETE r",
            parameters={"ids": chunk_ids},
        )
        print(f"  Triples deleted")

    conn.execute(
        "MATCH (d:Document {id: $id})-[:CONTAINS]->(c:Chunk) DETACH DELETE c",
        parameters={"id": doc_id},
    )
    conn.execute(
        "MATCH (d:Document {id: $id}) DETACH DELETE d",
        parameters={"id": doc_id},
    )
    print("Deleted.")
    return doc_id


def reset_offset(query: str, location: str) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from core.graph import KuzuStore

    query_key = f"{query}|{location}|"
    with KuzuStore(DB_PATH) as store:
        store.set_ingest_offset(query_key, 0)
        print(f"Reset offset for '{query_key}' to 0")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: uv run python3 scripts/reextract_doc.py <title_fragment> <query> <location>")
        print('Example: uv run python3 scripts/reextract_doc.py "Toward Trustworthy" "ATTRACTOR" langley')
        sys.exit(1)

    title_fragment = sys.argv[1]
    query = sys.argv[2]
    location = sys.argv[3]

    delete_document(title_fragment)
    reset_offset(query, location)

    print()
    print("Now run:")
    print(f'  uv run alice chat ingest "{query}" --location {location} --max-docs 1')
