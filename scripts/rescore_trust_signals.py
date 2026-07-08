#!/usr/bin/env python3
"""Backfill v3 trust signals (P(True) + NLI) onto existing RELATES_TO edges.

Only edges with evidence_text and missing signal columns are processed, so the
script is idempotent and resumable — rerun it and it picks up where it stopped.
Self-consistency cannot be backfilled (it requires re-extracting the chunk);
use scripts/reextract_doc.py for that.

Usage:
    uv run python3 scripts/rescore_trust_signals.py [--db PATH] [--limit N]
        [--only {ptrue,nli,all}] [--force] [--dry-run] [--no-backup]
        [--model NAME] [--backend {mlx,vllm,openai-compatible}] [--base-url URL]

The verifier defaults to alice.toml [llm]; --model/--backend/--base-url override
it, e.g. a fast non-reasoning model for the P(True) vote path or a vLLM endpoint
for the cheaper logprob path.

Note: the default pre-run backup rolls the DB back on interrupt, losing that
run's updates (they are safe to recompute). For very long runs, --no-backup
keeps partial progress at the (small) risk of corruption on a hard kill.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = REPO_ROOT / "services/chat/data/chat.db"


@dataclass
class Stats:
    scored: int = 0
    skipped: int = 0
    failed: int = 0


def _select_edges(conn, only: str, force: bool, limit: int | None):
    if force:
        missing = ""
    elif only == "ptrue":
        missing = "AND r.ptrue_score IS NULL"
    elif only == "nli":
        missing = "AND r.nli_entailment_score IS NULL"
    else:
        missing = "AND (r.ptrue_score IS NULL OR r.nli_entailment_score IS NULL)"
    query = (
        "MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity) "
        f"WHERE r.evidence_text IS NOT NULL {missing} "
        "RETURN s.name, r.relation, o.name, r.evidence_text, r.evidence_char_start, "
        "r.evidence_char_end, r.chunk_id, r.fact_id, r.ptrue_score, r.nli_entailment_score"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    result = conn.execute(query)
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


def _premise_for(conn, chunk_cache: dict, chunk_id: str, start, end, evidence_text: str) -> str:
    from core.llm.ingest_confidence import _sentence_covering_span

    if chunk_id not in chunk_cache:
        if len(chunk_cache) > 256:
            chunk_cache.clear()
        result = conn.execute(
            "MATCH (c:Chunk {id: $cid}) RETURN c.content", parameters={"cid": chunk_id}
        )
        chunk_cache[chunk_id] = result.get_next()[0] if result.has_next() else None
    content = chunk_cache[chunk_id]
    if content is None or start is None or end is None:
        return evidence_text
    return _sentence_covering_span(content, int(start), int(end))


def rescore(conn, backend, trust_cfg, *, limit: int | None, only: str, force: bool) -> Stats:
    from core.trust.signals import compute_evidence_signals

    stats = Stats()
    chunk_cache: dict[str, str | None] = {}
    rows = _select_edges(conn, only, force, limit)
    print(f"{len(rows)} edges to score (only={only}, force={force})")

    for i, row in enumerate(rows):
        subj, rel, obj, evidence, start, end, chunk_id, fact_id, has_ptrue, has_nli = row
        # Per-edge gating: don't burn LLM calls recomputing signals that exist
        cfg = dataclasses.replace(
            trust_cfg,
            ptrue_enabled=trust_cfg.ptrue_enabled
            and only in ("ptrue", "all")
            and (force or has_ptrue is None),
            nli_enabled=trust_cfg.nli_enabled
            and only in ("nli", "all")
            and (force or has_nli is None),
        )
        if not (cfg.ptrue_enabled or cfg.nli_enabled):
            stats.skipped += 1
            continue
        try:
            premise = _premise_for(conn, chunk_cache, chunk_id, start, end, evidence)
            signals = compute_evidence_signals(
                backend, subject=subj, relation=rel, object_=obj,
                evidence_sentence=premise, cfg=cfg,
            )
        except Exception as exc:
            print(f"[rescore] edge fact_id={fact_id} failed: {exc}")
            stats.failed += 1
            continue

        updates = {
            "ptrue_score": signals.ptrue_score,
            "ptrue_method": signals.ptrue_method,
            "ptrue_samples": signals.ptrue_samples,
            "nli_entailment_score": signals.nli_entailment_score,
            "nli_contradiction_score": signals.nli_contradiction_score,
            "nli_model": signals.nli_model,
        }
        updates = {k: v for k, v in updates.items() if v is not None}
        if not updates:
            stats.skipped += 1
            continue

        set_clause = ", ".join(f"r.{k} = ${k}" for k in updates)
        conn.execute(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.fact_id = $fid AND r.chunk_id = $cid "
            f"SET {set_clause}",
            parameters={"fid": fact_id, "cid": chunk_id, **updates},
        )
        stats.scored += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} edges processed")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", choices=["ptrue", "nli", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="recompute even if signals exist")
    parser.add_argument("--dry-run", action="store_true", help="count matching edges and exit")
    parser.add_argument("--no-backup", action="store_true", help="skip the safe_db backup (keeps partial progress on interrupt)")
    parser.add_argument("--model", default=None,
                        help="verification model override, e.g. a fast non-reasoning model (default: alice.toml [llm])")
    parser.add_argument("--backend", default=None, choices=["mlx", "vllm", "openai-compatible"],
                        help="LLM backend override (default: alice.toml [llm])")
    parser.add_argument("--base-url", default=None,
                        help="LLM server URL override, e.g. a vLLM endpoint for the cheaper logprob path")
    args = parser.parse_args()

    from core.graph import KuzuStore
    from core.llm.config import resolve_config
    from core.llm.factory import create_backend
    from core.trust.config import load_trust_config
    from services.ingest.service import safe_db

    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    trust_cfg = load_trust_config(REPO_ROOT)

    if args.dry_run:
        # KuzuStore (not raw kuzu) so the additive column migration runs first
        with KuzuStore(args.db) as store:
            rows = _select_edges(store._conn, args.only, args.force, args.limit)
        print(f"dry run: {len(rows)} edges would be scored")
        return

    print("Note: self-consistency cannot be backfilled — it requires re-extraction "
          "(scripts/reextract_doc.py). This script populates ptrue/nli columns only.")

    llm_cfg = resolve_config(
        cli_model=args.model, cli_backend=args.backend, cli_base_url=args.base_url,
        cli_api_key=None, cli_workers=None,
        start_dir=REPO_ROOT,
    )
    print(f"Verifier: {llm_cfg.model} ({llm_cfg.backend} @ {llm_cfg.base_url or 'local'})")
    backend = create_backend(llm_cfg)

    def _run():
        # KuzuStore (not raw kuzu) so the additive column migration runs first
        with KuzuStore(args.db) as store:
            stats = rescore(store._conn, backend, trust_cfg,
                            limit=args.limit, only=args.only, force=args.force)
        print(f"Done: {stats.scored} scored, {stats.skipped} skipped, {stats.failed} failed")

    if args.no_backup:
        _run()
    else:
        with safe_db(args.db):
            _run()


if __name__ == "__main__":
    main()
