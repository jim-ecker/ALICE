#!/usr/bin/env python3
"""Mint per-triple calibration labels from any ALICE knowledge graph.

Corpus-agnostic: point --db at the database whose distribution you deploy on
(for ALICE that means an NTRS-ingested DB, not a public benchmark — calibration
does not survive domain shift). Public-benchmark DBs remain useful for
developing this pipeline and for external-validity reporting.

Two stages:
  1. judge  — stratified random sample of triples (by document), each labeled
              Supported/NotSupported/Unclear by the configured LLM judge.
  2. audit  — `--audit N` interactively collects human labels on a uniform
              random subsample of judged rows; PPI uses these to correct the
              judge's bias (core/trust/ppi.py).

Usage:
    uv run python3 scripts/mint_calibration_labels.py --db <path> --sample 500 \
        [--judge-model llama3.3:70b] [--out labels.jsonl] [--seed 42]
    uv run python3 scripts/mint_calibration_labels.py --audit 50 [--out labels.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = REPO_ROOT / "services/chat/data/chat.db"
DEFAULT_OUT = REPO_ROOT / "experiments/data/calibration/labels.jsonl"

_SIGNAL_FIELDS = [
    "ptrue_score", "nli_entailment_score", "nli_contradiction_score",
    "self_consistency_freq", "evidence_alignment_score", "entity_anchor_score",
    "evidence_scope_score", "raw_certainty_score",
]


def sample_triples(conn, n: int, seed: int) -> list[dict]:
    """Stratified random sample of judgeable triples, round-robin by document
    so a few large documents can't dominate the calibration distribution."""
    result = conn.execute(
        """
        MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
        WHERE c.id = r.chunk_id AND r.evidence_text IS NOT NULL
        RETURN r.fact_id, r.chunk_id, d.id, s.name, r.relation, o.name,
               r.evidence_text, c.content,
               r.ptrue_score, r.nli_entailment_score, r.nli_contradiction_score,
               r.self_consistency_freq, r.evidence_alignment_score,
               r.entity_anchor_score, r.evidence_scope_score, r.raw_certainty_score,
               r.confidence_version, r.certainty_score
        """
    )
    by_document: dict[str, list[dict]] = defaultdict(list)
    while result.has_next():
        row = result.get_next()
        record = {
            "fact_id": row[0],
            "chunk_id": row[1],
            "document_id": row[2],
            "subject": row[3],
            "relation": row[4],
            "object": row[5],
            "evidence_text": row[6],
            "chunk_text": row[7],
            **{field: row[8 + i] for i, field in enumerate(_SIGNAL_FIELDS)},
            "confidence_version": row[16],
            "certainty_score": row[17],
        }
        by_document[record["document_id"]].append(record)

    rng = random.Random(seed)
    for records in by_document.values():
        rng.shuffle(records)
    documents = sorted(by_document)
    rng.shuffle(documents)

    sampled: list[dict] = []
    while len(sampled) < n and any(by_document[d] for d in documents):
        for document in documents:
            if by_document[document]:
                sampled.append(by_document[document].pop())
                if len(sampled) >= n:
                    break
    return sampled


def _load_existing(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def _write_all(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run_judge(args) -> None:
    import kuzu

    from core.trust.config import load_trust_config
    from core.trust.judge import Judge

    cfg = load_trust_config(REPO_ROOT)
    judge = Judge(cfg, model=args.judge_model, backend=args.judge_backend,
                  base_url=args.judge_base_url)

    existing = _load_existing(args.out)
    seen = {(r["fact_id"], r["chunk_id"]) for r in existing}

    db = kuzu.Database(str(args.db))
    conn = kuzu.Connection(db)
    sampled = sample_triples(conn, args.sample, args.seed)
    conn.close()
    db.close()

    todo = [r for r in sampled if (r["fact_id"], r["chunk_id"]) not in seen]
    print(f"{len(sampled)} sampled, {len(sampled) - len(todo)} already labeled, {len(todo)} to judge")

    rows = list(existing)
    for i, record in enumerate(todo):
        verdict = judge.judge_triple(
            record["chunk_text"], record["subject"], record["relation"],
            record["object"], evidence_text=record["evidence_text"],
        )
        record["judge_verdict"] = verdict.verdict
        record["judge_model"] = verdict.judge_model
        record["label"] = {"supported": 1, "not_supported": 0}.get(verdict.verdict)
        record["human_label"] = None
        rows.append(record)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            _write_all(args.out, rows)  # checkpoint — judging is the slow part
            print(f"  {i + 1}/{len(todo)} judged")

    _write_all(args.out, rows)
    verdicts = defaultdict(int)
    for r in rows:
        verdicts[r.get("judge_verdict", "?")] += 1
    print(f"Done: {dict(verdicts)} → {args.out}")


def run_audit(args) -> None:
    rows = _load_existing(args.out)
    if not rows:
        print(f"No labels file at {args.out} — run the judge stage first")
        sys.exit(1)

    unaudited = [i for i, r in enumerate(rows) if r.get("human_label") is None]
    rng = random.Random(args.seed)
    rng.shuffle(unaudited)
    picks = unaudited[: args.audit]
    print(f"Auditing {len(picks)} of {len(rows)} rows "
          f"({len(rows) - len(unaudited)} already audited).")
    print("For each fact: is it supported by the passage, in the correct context?")
    print("  [s]upported  [n]ot supported  [u]nclear  [q]uit (progress is saved)\n")

    for count, idx in enumerate(picks, start=1):
        row = rows[idx]
        print("=" * 78)
        print(f"[{count}/{len(picks)}] PASSAGE:\n{row['chunk_text']}\n")
        print(f"EVIDENCE CITED: {row['evidence_text']}\n")
        print(f"FACT: {row['subject']} — {row['relation']} — {row['object']}")
        while True:
            answer = input("verdict [s/n/u/q]: ").strip().lower()
            if answer in {"s", "n", "u", "q"}:
                break
        if answer == "q":
            break
        row["human_label"] = {"s": 1, "n": 0, "u": None}[answer]
        row["human_audited"] = True  # distinguishes audited-unclear from unaudited
        _write_all(args.out, rows)

    audited = sum(1 for r in rows if r.get("human_audited"))
    print(f"\nSaved. {audited} rows carry a human audit → {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-backend", default=None, choices=[None, "openai-compatible", "anthropic"])
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--audit", type=int, default=None,
                        help="interactively human-label N random judged rows instead of judging")
    args = parser.parse_args()

    if args.audit is not None:
        run_audit(args)
    else:
        if not args.db.exists():
            print(f"Database not found: {args.db}")
            sys.exit(1)
        run_judge(args)


if __name__ == "__main__":
    main()
