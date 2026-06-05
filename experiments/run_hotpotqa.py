#!/usr/bin/env python3
"""
Evaluate ALICE against HotpotQA.

Ingests HotpotQA context paragraphs into a fresh isolated knowledge graph, then
runs each question through the full ALICE retrieval pipeline and scores EM/F1.

Usage:
    uv run python experiments/run_hotpotqa.py
    uv run python experiments/run_hotpotqa.py --max-examples 50 --profile base
    uv run python experiments/run_hotpotqa.py --reuse-db   # skip ingestion, re-run eval
    uv run python experiments/run_hotpotqa.py --profile base --profile high_recall

Requires: uv add datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._core import (
    answer_em,
    answer_f1,
    build_profile_opts,
    close_session,
    extract_metrics,
    git_head,
    load_profiles,
    open_session,
    run_query,
    summarize,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_DEFAULT_DB = _EXPERIMENTS_DIR / "data" / "hotpotqa" / "hotpotqa.db"
_DEFAULT_EMB = _EXPERIMENTS_DIR / "data" / "hotpotqa" / "hotpotqa.embeddings.npz"
_DEFAULT_PROFILES = _EXPERIMENTS_DIR / "configs" / "profiles.toml"

_SENTENCES_PER_CHUNK = 4


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ALICE against HotpotQA.")
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Number of QA examples to evaluate (default: 50).")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--subset", default="fullwiki", choices=["fullwiki", "distractor"])
    parser.add_argument("--question-type", default="all", choices=["all", "bridge", "comparison"],
                        help="Filter to a specific HotpotQA question type.")
    parser.add_argument("--profile", action="append", default=[],
                        help="Retrieval profile(s) to evaluate (default: base). Repeatable.")
    parser.add_argument("--profiles", type=Path, default=_DEFAULT_PROFILES)
    parser.add_argument("--db-path", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--embeddings-path", type=Path, default=_DEFAULT_EMB)
    parser.add_argument("--reuse-db", action="store_true",
                        help="Skip ingestion if DB and embeddings already exist.")
    parser.add_argument("--no-extract-answer", action="store_true",
                        help="Score ALICE's verbose output directly instead of extracting a short answer.")
    parser.add_argument("--min-ingest-confidence", type=float, default=0.6,
                        help="Minimum LLM certainty score to keep a triple during ingestion.")
    parser.add_argument("--output-dir", type=Path, default=_EXPERIMENTS_DIR / "results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HotpotQA loading
# ---------------------------------------------------------------------------

def _load_hotpotqa(split: str, subset: str, max_examples: int, question_type: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install it with: uv add datasets"
        ) from exc
    ds = load_dataset("hotpot_qa", subset, split=split)
    rows: list[dict] = []
    for row in ds:
        if question_type != "all" and row.get("type") != question_type:
            continue
        rows.append(dict(row))
        if len(rows) >= max_examples:
            break
    return rows


# ---------------------------------------------------------------------------
# Context ingestion (bypasses Docling/PDF)
# ---------------------------------------------------------------------------

def _collect_articles(examples: list[dict]) -> dict[str, list[str]]:
    """Collect unique Wikipedia articles across all examples.

    HotpotQA context on HuggingFace is a dict with parallel lists:
      {"title": [...], "sentences": [[sent, ...], ...]}
    """
    articles: dict[str, list[str]] = {}
    for ex in examples:
        ctx = ex.get("context") or {}
        if isinstance(ctx, dict):
            titles = ctx.get("title") or []
            sents_list = ctx.get("sentences") or []
            pairs = zip(titles, sents_list)
        else:
            # Fallback: list of [title, sentences] pairs
            pairs = ctx
        for title, sentences in pairs:
            if title not in articles:
                articles[title] = [s.strip() for s in sentences if s.strip()]
    return articles


def _sentences_to_chunks(title: str, sentences: list[str], doc_id: str, source_url: str) -> list:
    """Group sentences into Chunk objects."""
    from services.ingest.models import Chunk, ChunkProvenance

    chunks = []
    group: list[str] = []

    def flush() -> None:
        if not group:
            return
        content = " ".join(group)
        chunk_id = hashlib.sha256(f"{doc_id}:{content}".encode()).hexdigest()
        chunks.append(Chunk(
            id=chunk_id,
            content=content,
            provenance=ChunkProvenance(
                document_id=doc_id,
                source_url=source_url,
                section_heading=title,
                page_number=None,
            ),
        ))
        group.clear()

    for sent in sentences:
        group.append(sent)
        if len(group) >= _SENTENCES_PER_CHUNK:
            flush()
    flush()
    return chunks


def _ingest_contexts(
    examples: list[dict],
    db_path: Path,
    embeddings_path: Path,
    ingest_llm_cfg: Any,
    embed_cfg: Any,
    min_ingest_confidence: float,
    skip_chunk_writing: bool = False,
) -> None:
    """Write HotpotQA article chunks to KuzuStore, extract triples, build index.

    skip_chunk_writing=True skips the document/chunk writing phase but still
    runs extraction and index building. Use this when chunks are already in the
    DB but extraction previously failed (e.g. missing mlx_lm).
    """
    from core.graph import KuzuStore
    from core.embeddings.client import EmbeddingsClient
    from services.ingest.models import SourceDocument
    from services.ingest.service import Ingest

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_chunk_writing:
        articles = _collect_articles(examples)
        print(json.dumps({"event": "ingest_start", "unique_articles": len(articles)}))
        with KuzuStore(db_path) as store:
            for title, sentences in articles.items():
                doc_id = hashlib.sha256(title.encode()).hexdigest()
                if store.document_exists(doc_id):
                    continue
                source_url = f"hotpotqa://{title.replace(' ', '_')}"
                source = SourceDocument(
                    id=doc_id,
                    path=Path("/dev/null"),
                    doc_type="hotpotqa",
                    source_url=source_url,
                )
                chunks = _sentences_to_chunks(title, sentences, doc_id, source_url)
                if chunks:
                    store.write_document_with_chunks(source, title, chunks)
        print(json.dumps({"event": "chunks_written"}))
    else:
        print(json.dumps({"event": "skipping_chunk_writing", "reason": "reuse-db"}))

    embed_client = EmbeddingsClient(embed_cfg)
    ingest = Ingest(
        db_path=db_path,
        llm_cfg=ingest_llm_cfg,
        embed_client=embed_client,
        embeddings_path=embeddings_path,
    )
    total_triples = ingest.extract(min_ingest_confidence=min_ingest_confidence)
    print(json.dumps({"event": "extraction_done", "triples": total_triples}))

    index = ingest.build_index()
    print(json.dumps({"event": "index_built", "size": len(index)}))


# ---------------------------------------------------------------------------
# Short answer extraction
# ---------------------------------------------------------------------------

_SHORT_ANSWER_PROMPT = (
    "Extract the answer from the response below.\n"
    "Output ONLY the bare answer — a name, phrase, number, or yes/no. No explanation, no sentence structure.\n"
    "- For yes/no questions output only: yes  or  no\n"
    "- Include the full proper noun or place name as written — do not truncate at a comma or 'and' (e.g. 'Greenwich Village, New York City' not 'Greenwich Village').\n"
    "- Preserve exact spacing within names and titles.\n"
    "- Include a number's unit or qualifier if the response states them together (e.g. '3,677 seated').\n"
    "- Do not add words that are not in the response.\n"
    "- If the response hedges, output your best guess from any partial information present.\n\n"
    "Question: {question}\n"
    "Response: {answer}\n\n"
    "Answer:"
)


def _extract_short_answer(llm: Any, question: str, verbose_answer: str) -> str:
    prompt = _SHORT_ANSWER_PROMPT.format(question=question, answer=verbose_answer[:2000])
    try:
        return llm.chat([{"role": "user", "content": prompt}], 32).strip()
    except Exception:
        return verbose_answer


# ---------------------------------------------------------------------------
# Per-profile evaluation
# ---------------------------------------------------------------------------

def _run_profile(
    profile_name: str,
    opts: dict[str, Any],
    samples: list[dict],
    db_path: Path,
    embeddings_path: Path,
    embed_cfg: Any,
    scoring_cfg: Any,
    llm_cfg: Any,
    extract_answer: bool,
) -> list[dict]:
    session = open_session(db_path, embeddings_path, opts, embed_cfg, scoring_cfg, llm_cfg)
    rows: list[dict] = []
    try:
        total = len(samples)
        for i, sample in enumerate(samples):
            question = sample["question"]
            gold_answer = sample["answer"]
            gold = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)

            verbose_answer, retrieval, elapsed = run_query(session, question)

            if extract_answer:
                scored_answer = _extract_short_answer(session.llm, question, verbose_answer)
            else:
                scored_answer = verbose_answer

            metrics = extract_metrics(verbose_answer, retrieval)
            em = answer_em(scored_answer, gold)
            f1 = answer_f1(scored_answer, gold)

            row = {
                "profile": profile_name,
                "id": sample.get("_id", f"sample_{i}"),
                "question": question,
                "type": sample.get("type", "unknown"),
                "level": sample.get("level", "unknown"),
                "gold_answer": gold_answer,
                "alice_answer": verbose_answer,
                "answer_em": em,
                "answer_f1": round(f1, 4),
                "runtime_seconds": round(elapsed, 4),
                **metrics,
            }
            if extract_answer:
                row["extracted_answer"] = scored_answer
            rows.append(row)

            print(json.dumps({
                "event": "question_done",
                "profile": profile_name,
                "i": i + 1,
                "total": total,
                "em": em,
                "f1": round(f1, 3),
                "question": question[:80],
            }))
    finally:
        close_session(session)
    return rows


def _type_breakdown(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for q_type in ("bridge", "comparison"):
        subset = [r for r in rows if r.get("type") == q_type]
        if subset:
            out[f"{q_type}_count"] = len(subset)
            out[f"{q_type}_em"] = round(sum(r["answer_em"] for r in subset) / len(subset), 4)
            out[f"{q_type}_f1"] = round(sum(r["answer_f1"] for r in subset) / len(subset), 4)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import tomllib
    from services.chat.config import load_chat_config
    from core.llm.config import LLMConfig

    args = _parse_args()

    chat_cfg, embed_cfg, scoring_cfg, chat_llm_cfg = load_chat_config()
    if chat_llm_cfg is None:
        print("[error] No [chat_llm] in alice.toml — required for evaluation.", file=sys.stderr)
        sys.exit(1)

    # Load [llm] (extraction LLM) from alice.toml for ingestion
    for search in [Path.cwd(), *Path.cwd().parents]:
        toml_path = search / "alice.toml"
        if toml_path.exists():
            raw_toml = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            break
    else:
        print("[error] alice.toml not found.", file=sys.stderr)
        sys.exit(1)

    llm_raw = raw_toml.get("llm") or {}
    ingest_llm_cfg = LLMConfig(
        backend=llm_raw.get("backend", "auto"),
        model=llm_raw.get("model", ""),
        base_url=llm_raw.get("base_url", ""),
        api_key=llm_raw.get("api_key", "token"),
        workers=llm_raw.get("workers", 1),
    )

    profiles_payload = load_profiles(args.profiles)
    global_cfg = dict(profiles_payload.get("global") or {})
    profile_map = dict(profiles_payload.get("profiles") or {})
    profile_names = args.profile or ["base"]

    for p in profile_names:
        if p not in profile_map:
            print(f"[error] Unknown profile '{p}'. Available: {sorted(profile_map)}", file=sys.stderr)
            sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"hotpotqa_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "event": "loading_dataset",
        "split": args.split,
        "subset": args.subset,
        "max_examples": args.max_examples,
        "question_type": args.question_type,
    }))
    examples = _load_hotpotqa(args.split, args.subset, args.max_examples, args.question_type)
    samples = [
        {
            "_id": ex.get("id") or ex.get("_id", f"sample_{i}"),
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex.get("type"),
            "level": ex.get("level"),
        }
        for i, ex in enumerate(examples)
    ]
    print(json.dumps({"event": "dataset_loaded", "samples": len(samples)}))

    # Ingestion
    # --reuse-db: skip chunk writing (DB already has chunks) but still extract+index.
    # No flag: wipe DB and start fresh.
    if args.reuse_db and args.db_path.exists():
        _ingest_contexts(
            examples, args.db_path, args.embeddings_path,
            ingest_llm_cfg, embed_cfg, args.min_ingest_confidence,
            skip_chunk_writing=True,
        )
    else:
        if args.db_path.exists():
            if args.db_path.is_dir():
                shutil.rmtree(args.db_path)
            else:
                args.db_path.unlink()
        if args.embeddings_path.exists():
            args.embeddings_path.unlink()
        _ingest_contexts(
            examples, args.db_path, args.embeddings_path,
            ingest_llm_cfg, embed_cfg, args.min_ingest_confidence,
        )

    if not args.db_path.exists():
        print(f"[error] DB missing after ingestion: {args.db_path}", file=sys.stderr)
        sys.exit(1)
    if not args.embeddings_path.exists():
        print(f"[error] Embeddings missing after ingestion: {args.embeddings_path}", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(_REPO_ROOT),
        "max_examples": args.max_examples,
        "split": args.split,
        "subset": args.subset,
        "question_type": args.question_type,
        "db_path": str(args.db_path),
        "embeddings_path": str(args.embeddings_path),
        "extract_short_answer": not args.no_extract_answer,
        "profiles": profile_names,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    all_runs: list[dict] = []
    for profile_name in profile_names:
        raw_profile = dict(profile_map.get(profile_name) or {})
        opts = build_profile_opts(global_cfg, raw_profile)

        print(json.dumps({"event": "profile_start", "profile": profile_name, "samples": len(samples)}))
        rows = _run_profile(
            profile_name, opts, samples,
            args.db_path, args.embeddings_path,
            embed_cfg, scoring_cfg, chat_llm_cfg,
            extract_answer=not args.no_extract_answer,
        )

        summary = summarize(rows, include_qa_metrics=True)
        summary.update(_type_breakdown(rows))

        report = {
            "profile": profile_name,
            "profile_opts": {k: v for k, v in opts.items() if k not in ("enabled", "description")},
            "summary": summary,
            "results": rows,
        }
        file_base = f"hotpotqa__{profile_name}"
        (run_dir / f"{file_base}.json").write_text(json.dumps(report, indent=2))
        with (run_dir / f"{file_base}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        all_runs.append({"profile": profile_name, "summary": summary})
        print(json.dumps({"event": "profile_done", "profile": profile_name, "summary": summary}))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({"manifest": manifest, "runs": all_runs}, indent=2))
    print(json.dumps({"event": "done", "output": str(run_dir)}))


if __name__ == "__main__":
    main()
