#!/usr/bin/env python3
"""
Oracle and failure analysis for 2WikiMultihopQA evaluation runs.

For each question, re-runs retrieval (no LLM) and checks whether the gold
answer appears in:
  1. The original 2Wiki context paragraphs  → context ceiling
  2. The retrieved chunks from ALICE         → retrieval ceiling

Failure modes:
  reasoning   — answer in context AND retrieved, LLM wrong
  retrieval   — answer in context but not retrieved
  context_gap — answer not in provided context at all

Usage:
    uv run python experiments/analyze_2wiki.py \\
        --results experiments/results/multihop_2wikimultihopqa_20260520_044806/multihop_2wikimultihopqa__high_recall.jsonl \\
        --results experiments/results/multihop_2wikimultihopqa_20260520_044806/multihop_2wikimultihopqa__trust_filtered.jsonl \\
        --local-data-file experiments/data/2wikimultihopqa/dev.json \\
        --output experiments/results/wiki_oracle_20260521.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._core import build_profile_opts, close_session, load_profiles, normalize_text, open_session

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_DEFAULT_DB = _EXPERIMENTS_DIR / "data" / "2wikimultihopqa" / "2wikimultihopqa.db"
_DEFAULT_EMB = _EXPERIMENTS_DIR / "data" / "2wikimultihopqa" / "2wikimultihopqa.embeddings.npz"
_DEFAULT_PROFILES = _EXPERIMENTS_DIR / "configs" / "profiles.toml"
_DEFAULT_DATA = _EXPERIMENTS_DIR / "data" / "2wikimultihopqa" / "dev.json"
_QUESTION_TYPES = ("bridge", "comparison", "inference", "compositional")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle and failure analysis for 2WikiMultihopQA runs.")
    p.add_argument("--results", action="append", required=True,
                   help="Path(s) to multihop_2wikimultihopqa__<profile>.jsonl result file(s).")
    p.add_argument("--local-data-file", type=Path, default=_DEFAULT_DATA,
                   help="Path to dev.json from the official 2Wiki release.")
    p.add_argument("--db-path", type=Path, default=_DEFAULT_DB)
    p.add_argument("--embeddings-path", type=Path, default=_DEFAULT_EMB)
    p.add_argument("--profiles", type=Path, default=_DEFAULT_PROFILES)
    p.add_argument("--retrieval-profile", default="high_recall")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Context loader from local dev.json
# ---------------------------------------------------------------------------

def _load_2wiki_contexts(data_file: Path, sample_ids: set[str]) -> dict[str, list[str]]:
    """Return {question_id: [paragraph_text, ...]} from local dev.json."""
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else next(v for v in raw.values() if isinstance(v, list))
    contexts: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row.get("_id") or row.get("id", ""))
        if qid not in sample_ids:
            continue
        paragraphs: list[str] = []
        for item in row.get("context") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sents = item
                text = " ".join(str(s).strip() for s in sents if s)
                if text:
                    paragraphs.append(text)
            elif isinstance(item, dict):
                text = " ".join(str(s).strip() for s in (item.get("sentences") or []) if s)
                if text:
                    paragraphs.append(text)
        contexts[qid] = paragraphs
        if len(contexts) >= len(sample_ids):
            break
    return contexts


# ---------------------------------------------------------------------------
# Oracle helpers
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

    all_results: dict[str, list[dict]] = {}
    all_ids: set[str] = set()
    for rpath in args.results:
        rpath = Path(rpath)
        profile_name = rpath.stem.replace("multihop_2wikimultihopqa__", "")
        rows = [json.loads(l) for l in rpath.read_text().splitlines() if l.strip()]
        all_results[profile_name] = rows
        all_ids.update(r["id"] for r in rows)

    print(f"Loaded {len(all_ids)} unique questions across {len(all_results)} profile(s).")

    print(f"Loading 2Wiki contexts from {args.local_data_file}...")
    hf_contexts = _load_2wiki_contexts(args.local_data_file, all_ids)
    print(f"  Loaded contexts for {len(hf_contexts)}/{len(all_ids)} questions.")

    print(f"Opening retrieval session (profile: {args.retrieval_profile})...")
    session = open_session(
        args.db_path, args.embeddings_path, opts,
        embed_cfg, scoring_cfg, llm_cfg,
    )

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
            paragraphs = hf_contexts.get(qid, [])
            in_context = _oracle_context(gold, paragraphs)
            in_retrieved = _oracle_retrieved(gold, retrieval.chunks)

            if in_context and in_retrieved:
                failure_mode = "reasoning"
            elif in_context and not in_retrieved:
                failure_mode = "retrieval"
            elif not in_context:
                failure_mode = "context_gap"
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
            for profile_name, rows in all_results.items():
                row_map = {r["id"]: r for r in rows}
                pr = row_map.get(qid, {})
                row[f"{profile_name}_em"] = pr.get("answer_em")
                row[f"{profile_name}_f1"] = pr.get("answer_f1")
                row[f"{profile_name}_extracted"] = pr.get("extracted_answer")

            oracle_rows.append(row)
            print(f"  [{i}/{total}] {failure_mode:12s}  in_ctx={int(in_context)}  in_ret={int(in_retrieved)}  [{q_type:14s}]  {question[:55]}")
    finally:
        close_session(session)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ORACLE ANALYSIS SUMMARY — 2WikiMultihopQA")
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
    for q_type in _QUESTION_TYPES:
        subset = [r for r in oracle_rows if r.get("type") == q_type]
        if not subset:
            continue
        print(f"\n  {q_type} ({len(subset)} questions):")
        ctx_sub = [r for r in subset if r["in_original_context"]]
        ret_sub = [r for r in subset if r["in_retrieved_chunks"]]
        print(f"    context ceiling:   {len(ctx_sub)}/{len(subset)} = {len(ctx_sub)/len(subset):.1%}")
        print(f"    retrieval ceiling: {len(ret_sub)}/{len(subset)} = {len(ret_sub)/len(subset):.1%}")
        for mode in ("reasoning", "retrieval", "context_gap"):
            ms = [r for r in subset if r["failure_mode"] == mode]
            print(f"    {mode:12s}: {len(ms):3d}  ({len(ms)/len(subset):.1%})")

    print("\nEM by retrieval oracle condition:")
    for profile_name in all_results:
        em_key = f"{profile_name}_em"
        has_ret_em  = [r[em_key] for r in oracle_rows if r["in_retrieved_chunks"] and r.get(em_key) is not None]
        no_ret_em   = [r[em_key] for r in oracle_rows if not r["in_retrieved_chunks"] and r.get(em_key) is not None]
        has_ctx_em  = [r[em_key] for r in oracle_rows if r["in_original_context"] and r.get(em_key) is not None]
        no_ctx_em   = [r[em_key] for r in oracle_rows if not r["in_original_context"] and r.get(em_key) is not None]
        print(f"\n  [{profile_name}]")
        if has_ctx_em:
            print(f"    EM when gold in original context:     {sum(has_ctx_em)/len(has_ctx_em):.1%}  (n={len(has_ctx_em)})")
        if no_ctx_em:
            print(f"    EM when gold NOT in original context: {sum(no_ctx_em)/len(no_ctx_em):.1%}  (n={len(no_ctx_em)})")
        if has_ret_em:
            print(f"    EM when gold in retrieved chunks:     {sum(has_ret_em)/len(has_ret_em):.1%}  (n={len(has_ret_em)})")
        if no_ret_em:
            print(f"    EM when gold NOT in retrieved chunks: {sum(no_ret_em)/len(no_ret_em):.1%}  (n={len(no_ret_em)})")

    print("\nEM by question type and oracle condition (high_recall profile):")
    profile_name = next(iter(all_results))
    em_key = f"{profile_name}_em"
    for q_type in _QUESTION_TYPES:
        subset = [r for r in oracle_rows if r.get("type") == q_type]
        if not subset:
            continue
        ret_em = [r[em_key] for r in subset if r["in_retrieved_chunks"] and r.get(em_key) is not None]
        no_ret_em = [r[em_key] for r in subset if not r["in_retrieved_chunks"] and r.get(em_key) is not None]
        overall_em = [r[em_key] for r in subset if r.get(em_key) is not None]
        print(f"  {q_type:14s}: overall={sum(overall_em)/len(overall_em):.1%}", end="")
        if ret_em:
            print(f"  retrieved={sum(ret_em)/len(ret_em):.1%}(n={len(ret_em)})", end="")
        if no_ret_em:
            print(f"  not-retrieved={sum(no_ret_em)/len(no_ret_em):.1%}(n={len(no_ret_em)})", end="")
        print()

    if args.output:
        args.output.write_text(json.dumps(oracle_rows, indent=2))
        print(f"\nFull table written to {args.output}")


if __name__ == "__main__":
    main()
