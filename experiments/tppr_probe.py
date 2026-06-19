#!/usr/bin/env python3
"""
TPPR diagnostic probe.

Runs a query through the retrieval pipeline with and without Trust-Propagated
Path Retrieval, and shows exactly what TPPR contributes at each stage.

Usage:
    uv run python experiments/tppr_probe.py "your query here"
    uv run python experiments/tppr_probe.py --list-entities   # show KG entity sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default=None, help="Query to probe")
    p.add_argument("--list-entities", action="store_true", help="Print a sample of KG entities and exit")
    p.add_argument("--max-paths", type=int, default=50)
    p.add_argument("--db-path", type=Path, default=None)
    p.add_argument("--embeddings-path", type=Path, default=None)
    args = p.parse_args()

    from services.chat.config import load_chat_config
    chat_cfg, _embed_cfg, _scoring_cfg, _llm_cfg = load_chat_config()

    db_path = args.db_path or chat_cfg.db_path
    if not db_path.exists():
        print(f"[error] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    import kuzu
    from core.graph.kuzu_store import _BUFFER_POOL_SIZE
    db = kuzu.Database(str(db_path), buffer_pool_size=_BUFFER_POOL_SIZE, read_only=True)
    conn = kuzu.Connection(db)

    # -------------------------------------------------------------------------
    # --list-entities: show a sample so the user can pick a good test query
    # -------------------------------------------------------------------------
    if args.list_entities:
        res = conn.execute(
            "MATCH (e:Entity)-[r:RELATES_TO]->(e2:Entity) "
            "RETURN e.name, r.relation, e2.name, r.certainty_score "
            "ORDER BY r.certainty_score DESC LIMIT 40"
        )
        print("\nTop 40 KG triples by certainty score:\n")
        print(f"  {'Subject':<35} {'Relation':<25} {'Object':<35} {'Cert':>5}")
        print(f"  {'-'*35} {'-'*25} {'-'*35} {'-'*5}")
        while res.has_next():
            s, r, o, c = res.get_next()
            print(f"  {(s or ''):<35} {(r or ''):<25} {(o or ''):<35} {float(c or 0):.3f}")
        db.close()
        return

    if not args.query:
        p.print_help()
        sys.exit(1)

    query = args.query

    from core.graph.retrieval import _match_anchor_entities, find_entity_chunks, find_trust_paths

    # -------------------------------------------------------------------------
    # Stage 1: anchor entity matching
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Query: {query!r}")
    print(f"{'='*70}")

    anchors = _match_anchor_entities(conn, query)
    print(f"\n[Stage 1] Anchor entities matched: {len(anchors)}")
    for a in anchors:
        print(f"  · {a}")

    if not anchors:
        print("\n  ⚠  No anchors found — TPPR will contribute nothing for this query.")
        print("     Try --list-entities to find entity names present in the KG.")

    # -------------------------------------------------------------------------
    # Stage 2: trust paths
    # -------------------------------------------------------------------------
    paths = find_trust_paths(conn, anchors, max_paths=args.max_paths)
    print(f"\n[Stage 2] Trust paths found: {len(paths)}")
    for i, path in enumerate(paths[:10]):
        edges_str = " → ".join(
            f"{rel}({tgt})" for rel, tgt, _c, _cid in path.edges
        )
        chunk_ids = ", ".join(path.chunk_ids[:2])
        print(f"  [{i+1}] anchor={path.anchor!r}  trust={path.path_trust:.3f}")
        print(f"       {edges_str}")
        print(f"       chunk_ids: {chunk_ids}")
    if len(paths) > 10:
        print(f"  ... {len(paths) - 10} more paths")

    # -------------------------------------------------------------------------
    # Stage 3: compare chunk sets
    # -------------------------------------------------------------------------
    entity_ids = find_entity_chunks(conn, query, hop_depth=chat_cfg.entity_hop_depth,
                                    max_hop2_entities=chat_cfg.max_hop2_entities)
    baseline_ids: set[str] = set(entity_ids)

    tppr_ids: set[str] = set()
    for path in paths:
        for cid in path.chunk_ids:
            tppr_ids.add(cid)

    tppr_only = tppr_ids - baseline_ids
    overlap = tppr_ids & baseline_ids

    print(f"\n[Stage 3] Chunk ID comparison")
    print(f"  Entity-hop baseline chunks : {len(baseline_ids)}")
    print(f"  TPPR path chunks (total)   : {len(tppr_ids)}")
    print(f"  Already in baseline        : {len(overlap)}")
    print(f"  NEW chunks from TPPR       : {len(tppr_only)}")

    if tppr_only:
        print(f"\n  Sample of TPPR-only chunk IDs:")
        for cid in list(tppr_only)[:5]:
            print(f"    · {cid}")

        # fetch content of a couple
        sample_ids = list(tppr_only)[:3]
        res = conn.execute(
            """
            MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
            WHERE c.id IN $ids
            RETURN c.id, c.content, d.title
            """,
            parameters={"ids": sample_ids},
        )
        print(f"\n  Content of up to 3 TPPR-only chunks:")
        while res.has_next():
            cid, content, title = res.get_next()
            snippet = (content or "")[:200].replace("\n", " ")
            print(f"\n  ── chunk {cid} (from: {title})")
            print(f"     {snippet}...")
    else:
        if tppr_ids:
            print("\n  All TPPR chunks were already in the baseline — TPPR is not adding new coverage.")
        else:
            print("\n  TPPR found no paths → no chunk IDs contributed.")

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    if len(anchors) == 0:
        verdict = "TPPR NOT TRIGGERED — no anchor entities matched the query."
    elif len(paths) == 0:
        verdict = "TPPR TRIGGERED but no paths found — KG may lack edges for these anchors."
    elif len(tppr_only) == 0:
        verdict = "TPPR TRIGGERED, paths found, but no new chunks added (all overlapped baseline)."
    else:
        verdict = f"TPPR WORKING — added {len(tppr_only)} new chunk(s) beyond entity-hop baseline."
    print(f"Verdict: {verdict}")
    print(f"{'='*70}\n")

    db.close()


if __name__ == "__main__":
    main()
