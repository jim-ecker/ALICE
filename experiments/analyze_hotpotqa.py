#!/usr/bin/env python3
"""
Oracle and failure analysis for HotpotQA evaluation runs.

For each question, re-runs retrieval (no LLM) and checks whether the gold
answer appears in:
  1. The original HotpotQA context paragraphs  → context ceiling
  2. The retrieved chunks from ALICE            → retrieval ceiling

This separates three distinct failure modes:
  A. Answer not in original context  — dataset limitation, nothing to fix
  B. Answer in context but not retrieved — retrieval failure, fixable
  C. Answer retrieved but LLM got it wrong — reasoning failure, fixable

Usage:
    uv run python experiments/analyze_hotpotqa.py \\
        --results experiments/results/hotpotqa_20260515_222554/hotpotqa__base.jsonl \\
        --db-path experiments/data/hotpotqa/hotpotqa.db \\
        --embeddings-path experiments/data/hotpotqa/hotpotqa.embeddings.npz

    # compare two profiles side-by-side
    uv run python experiments/analyze_hotpotqa.py \\
        --results experiments/results/.../hotpotqa__high_recall.jsonl \\
        --results experiments/results/.../hotpotqa__base.jsonl \\
        --db-path experiments/data/hotpotqa/hotpotqa.db \\
        --embeddings-path experiments/data/hotpotqa/hotpotqa.embeddings.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._core import answer_f1, build_profile_opts, close_session, load_profiles, normalize_text, open_session

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_DEFAULT_DB = _EXPERIMENTS_DIR / "data" / "hotpotqa" / "hotpotqa.db"
_DEFAULT_EMB = _EXPERIMENTS_DIR / "data" / "hotpotqa" / "hotpotqa.embeddings.npz"
_DEFAULT_PROFILES = _EXPERIMENTS_DIR / "configs" / "profiles.toml"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle and failure analysis for HotpotQA runs.")
    p.add_argument("--results", action="append", required=True,
                   help="Path(s) to hotpotqa__<profile>.jsonl result file(s).")
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--hf-subset", default="fullwiki")
    p.add_argument("--max-examples", type=int, default=0,
                   help="Re-load this many examples from HF (0 = infer from results file).")
    p.add_argument("--db-path", type=Path, default=_DEFAULT_DB)
    p.add_argument("--embeddings-path", type=Path, default=_DEFAULT_EMB)
    p.add_argument("--profiles", type=Path, default=_DEFAULT_PROFILES)
    p.add_argument("--retrieval-profile", default="high_recall",
                   help="Profile to use when re-running retrieval for oracle analysis.")
    p.add_argument("--output", type=Path, default=None,
                   help="Write full per-question oracle table to this JSON file.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# HotpotQA context loader
# ---------------------------------------------------------------------------

def _load_hf_contexts(split: str, subset: str, sample_ids: set[str]) -> dict[str, list[str]]:
    """Return {question_id: [paragraph_text, ...]} for the requested IDs."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: uv add datasets") from exc
    ds = load_dataset("hotpot_qa", subset, split=split)
    contexts: dict[str, list[str]] = {}
    for row in ds:
        qid = row.get("id") or row.get("_id", "")
        if qid not in sample_ids:
            continue
        ctx = row.get("context") or {}
        paragraphs: list[str] = []
        if isinstance(ctx, dict):
            for sents in (ctx.get("sentences") or []):
                paragraphs.append(" ".join(s.strip() for s in sents if s.strip()))
        else:
            for _, sents in ctx:
                paragraphs.append(" ".join(s.strip() for s in sents if s.strip()))
        contexts[qid] = paragraphs
        if len(contexts) >= len(sample_ids):
            break
    return contexts


# ---------------------------------------------------------------------------
# Oracle check helpers
# ---------------------------------------------------------------------------

def _answer_in_text(gold: str, text: str) -> bool:
    return normalize_text(gold) in normalize_text(text)


def _oracle_retrieved(gold: str, chunks) -> bool:
    return any(_answer_in_text(gold, c.content) for c in chunks)


def _oracle_context(gold: str, paragraphs: list[str]) -> bool:
    return any(_answer_in_text(gold, p) for p in paragraphs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    from services.chat.config import load_chat_config
    chat_cfg, embed_cfg, scoring_cfg, llm_cfg = load_chat_config()

    profiles_payload = load_profiles(args.profiles)
    global_cfg = dict(profiles_payload.get("global") or {})
    profile_map = dict(profiles_payload.get("profiles") or {})
    raw_profile = dict(profile_map.get(args.retrieval_profile) or {})
    opts = build_profile_opts(global_cfg, raw_profile)

    # Load all result files
    all_results: dict[str, list[dict]] = {}
    all_ids: set[str] = set()
    for rpath in args.results:
        rpath = Path(rpath)
        profile_name = rpath.stem.replace("hotpotqa__", "")
        rows = [json.loads(l) for l in rpath.read_text().splitlines() if l.strip()]
        all_results[profile_name] = rows
        all_ids.update(r["id"] for r in rows)

    print(f"Loaded {len(all_ids)} unique questions across {len(all_results)} profile(s).")

    # Load HotpotQA contexts for oracle ceiling
    print(f"Loading HotpotQA contexts from HuggingFace ({args.hf_subset}/{args.hf_split})...")
    hf_contexts = _load_hf_contexts(args.hf_split, args.hf_subset, all_ids)
    print(f"  Loaded contexts for {len(hf_contexts)}/{len(all_ids)} questions.")

    # Open retrieval session for oracle retrieval check
    print(f"Opening retrieval session (profile: {args.retrieval_profile})...")
    session = open_session(
        args.db_path, args.embeddings_path, opts,
        embed_cfg, scoring_cfg, llm_cfg,
    )

    # Build a unified question map from the first result file
    first_rows = next(iter(all_results.values()))
    question_map = {r["id"]: r for r in first_rows}

    oracle_rows: list[dict] = []
    total = len(question_map)

    try:
        for i, (qid, base_row) in enumerate(question_map.items(), 1):
            question = base_row["question"]
            gold = base_row["gold_answer"]
            q_type = base_row.get("type", "unknown")

            retrieval = session.retriever.retrieve(question)
            in_context = _oracle_context(gold, hf_contexts.get(qid, []))
            in_retrieved = _oracle_retrieved(gold, retrieval.chunks)

            # Failure mode classification
            if in_context and in_retrieved:
                failure_mode = "reasoning"   # LLM had the answer, got it wrong
            elif in_context and not in_retrieved:
                failure_mode = "retrieval"   # Answer was there, retrieval missed it
            elif not in_context:
                failure_mode = "context_gap" # Not in provided paragraphs at all
            else:
                failure_mode = "unknown"

            row: dict[str, Any] = {
                "id": qid,
                "question": question,
                "type": q_type,
                "gold_answer": gold,
                "in_original_context": in_context,
                "in_retrieved_chunks": in_retrieved,
                "failure_mode": failure_mode,
                "chunks_retrieved": len(retrieval.chunks),
            }
            # Attach per-profile EM/F1 from result files
            for profile_name, rows in all_results.items():
                row_map = {r["id"]: r for r in rows}
                pr = row_map.get(qid, {})
                row[f"{profile_name}_em"] = pr.get("answer_em")
                row[f"{profile_name}_f1"] = pr.get("answer_f1")
                row[f"{profile_name}_extracted"] = pr.get("extracted_answer")

            oracle_rows.append(row)
            print(f"  [{i}/{total}] {failure_mode:12s}  in_ctx={int(in_context)}  in_ret={int(in_retrieved)}  {question[:60]}")
    finally:
        close_session(session)

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ORACLE ANALYSIS SUMMARY")
    print("=" * 70)

    total_n = len(oracle_rows)
    ctx_yes = [r for r in oracle_rows if r["in_original_context"]]
    ret_yes = [r for r in oracle_rows if r["in_retrieved_chunks"]]

    print(f"\nContext ceiling  (answer in original paragraphs): {len(ctx_yes)}/{total_n} = {len(ctx_yes)/total_n:.1%}")
    print(f"Retrieval ceiling (answer in retrieved chunks):   {len(ret_yes)}/{total_n} = {len(ret_yes)/total_n:.1%}")

    print("\nFailure mode breakdown:")
    for mode in ("reasoning", "retrieval", "context_gap", "unknown"):
        subset = [r for r in oracle_rows if r["failure_mode"] == mode]
        print(f"  {mode:12s}: {len(subset):3d}  ({len(subset)/total_n:.1%})")

    print("\nFailure modes by question type:")
    for q_type in ("bridge", "comparison"):
        subset = [r for r in oracle_rows if r.get("type") == q_type]
        if not subset:
            continue
        print(f"\n  {q_type} ({len(subset)} questions):")
        for mode in ("reasoning", "retrieval", "context_gap"):
            ms = [r for r in subset if r["failure_mode"] == mode]
            print(f"    {mode:12s}: {len(ms):3d}  ({len(ms)/len(subset):.1%})")

    # Per-profile EM given oracle conditions
    print("\nEM by retrieval oracle condition (does having the answer in context help?):")
    for profile_name in all_results:
        em_key = f"{profile_name}_em"
        has_ctx_em = [r[em_key] for r in oracle_rows if r["in_original_context"] and r.get(em_key) is not None]
        no_ctx_em  = [r[em_key] for r in oracle_rows if not r["in_original_context"] and r.get(em_key) is not None]
        has_ret_em = [r[em_key] for r in oracle_rows if r["in_retrieved_chunks"] and r.get(em_key) is not None]
        no_ret_em  = [r[em_key] for r in oracle_rows if not r["in_retrieved_chunks"] and r.get(em_key) is not None]
        print(f"\n  [{profile_name}]")
        if has_ctx_em:
            print(f"    EM when gold in original context:     {sum(has_ctx_em)/len(has_ctx_em):.1%}  (n={len(has_ctx_em)})")
        if no_ctx_em:
            print(f"    EM when gold NOT in original context: {sum(no_ctx_em)/len(no_ctx_em):.1%}  (n={len(no_ctx_em)})")
        if has_ret_em:
            print(f"    EM when gold in retrieved chunks:     {sum(has_ret_em)/len(has_ret_em):.1%}  (n={len(has_ret_em)})")
        if no_ret_em:
            print(f"    EM when gold NOT in retrieved chunks: {sum(no_ret_em)/len(no_ret_em):.1%}  (n={len(no_ret_em)})")

    print("\nRetrieval failures (answer in context but not retrieved):")
    for r in oracle_rows:
        if r["failure_mode"] == "retrieval":
            print(f"  [{r['type']:10s}] {r['question'][:70]}")
            print(f"             gold: {r['gold_answer']}")

    print("\nReasoning failures (answer retrieved but LLM wrong) — sample:")
    reasoning = [r for r in oracle_rows if r["failure_mode"] == "reasoning"]
    for r in reasoning[:10]:
        profile_name = next(iter(all_results))
        extracted = r.get(f"{profile_name}_extracted", "?")
        print(f"  [{r['type']:10s}] {r['question'][:70]}")
        print(f"             gold: {r['gold_answer']}  |  extracted: {extracted}")

    if args.output:
        args.output.write_text(json.dumps(oracle_rows, indent=2))
        print(f"\nFull table written to {args.output}")


if __name__ == "__main__":
    main()
