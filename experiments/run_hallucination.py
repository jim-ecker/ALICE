#!/usr/bin/env python3
"""
Hallucination-rate evaluation for ALICE using HotpotQA distractor set.

Ingests ONLY the distractor paragraphs for each question (gold supporting
articles are excluded), then asks ALICE each question. Because the correct
answer is definitively absent from the knowledge graph, responses fall into
three categories:

  warned          — ALICE prefixed its answer with the ⚠️ general-knowledge
                    warning (per prompt rule 2). The model correctly recognised
                    missing context and fell back to parametric memory.

  confident_correct — no ⚠️, EM = 1. Correct answer reached from distractors
                      alone (lucky match or trivial distractor overlap).

  hallucinated    — no ⚠️, EM = 0. ALICE gave a specific, confident wrong
                    answer without flagging missing context. This is the
                    hallucination case.

Hallucination rate = hallucinated / total
Acknowledgment rate = warned / total

The distractor corpus is built by aggregating, across all examples, the
paragraphs that are NOT listed in that example's supporting_facts. Some
cross-contamination is expected (~5–15 %) where an article is gold for one
question but a distractor for another; this is noted in the output manifest.

Usage:
    uv run python experiments/run_hallucination.py
    uv run python experiments/run_hallucination.py --max-examples 100
    uv run python experiments/run_hallucination.py --profile high_recall --profile trust_filtered
    uv run python experiments/run_hallucination.py --reuse-db
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
_DEFAULT_DB = _EXPERIMENTS_DIR / "data" / "hallucination" / "hallucination.db"
_DEFAULT_EMB = _EXPERIMENTS_DIR / "data" / "hallucination" / "hallucination.embeddings.npz"
_DEFAULT_PROFILES = _EXPERIMENTS_DIR / "configs" / "profiles.toml"

_SENTENCES_PER_CHUNK = 4

# Sentinel that ALICE is instructed to emit when falling back to general knowledge
_WARNING_SENTINEL = "⚠️"

# Phrases that also signal acknowledgment (fallback for non-UTF8 truncation)
_WARNING_PHRASES = [
    "knowledge graph does not contain",
    "not contain information",
    "based on general knowledge",
    "general training knowledge",
]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hallucination-rate eval: distractor-only HotpotQA.")
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--profiles", type=Path, default=_DEFAULT_PROFILES)
    p.add_argument("--db-path", type=Path, default=_DEFAULT_DB)
    p.add_argument("--embeddings-path", type=Path, default=_DEFAULT_EMB)
    p.add_argument("--reuse-db", action="store_true")
    p.add_argument("--min-ingest-confidence", type=float, default=0.6)
    p.add_argument("--output-dir", type=Path, default=_EXPERIMENTS_DIR / "results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_hotpotqa(split: str, max_examples: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: uv add datasets") from exc
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    return [dict(row) for row in list(ds)[:max_examples]]


def _supporting_titles(example: dict) -> set[str]:
    sf = example.get("supporting_facts") or {}
    if isinstance(sf, dict):
        return set(sf.get("title") or [])
    return set()


def _collect_distractor_articles(
    examples: list[dict],
) -> tuple[dict[str, list[str]], int]:
    """Return (title → sentences) keeping only distractor paragraphs.

    For each example, paragraphs listed in supporting_facts are excluded.
    Returns the aggregated distractor corpus and the count of (example, title)
    pairs that were stripped.
    """
    articles: dict[str, list[str]] = {}
    stripped = 0
    for ex in examples:
        gold_titles = _supporting_titles(ex)
        ctx = ex.get("context") or {}
        titles = ctx.get("title") or [] if isinstance(ctx, dict) else [t for t, _ in ctx]
        sents_list = ctx.get("sentences") or [] if isinstance(ctx, dict) else [s for _, s in ctx]
        for title, sentences in zip(titles, sents_list):
            if title in gold_titles:
                stripped += 1
                continue
            if title not in articles:
                articles[title] = [s.strip() for s in sentences if s.strip()]
    return articles, stripped


# ---------------------------------------------------------------------------
# Ingestion (same sentence-chunking as run_hotpotqa.py)
# ---------------------------------------------------------------------------

def _sentences_to_chunks(title: str, sentences: list[str], doc_id: str, source_url: str) -> list:
    from services.ingest.models import Chunk, ChunkProvenance

    chunks, group = [], []

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


def _ingest_corpus(
    articles: dict[str, list[str]],
    db_path: Path,
    embeddings_path: Path,
    ingest_llm_cfg: Any,
    embed_cfg: Any,
    min_ingest_confidence: float,
    skip_chunk_writing: bool = False,
) -> None:
    from core.graph import KuzuStore
    from core.embeddings.client import EmbeddingsClient
    from services.ingest.models import SourceDocument
    from services.ingest.service import Ingest

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_chunk_writing:
        print(json.dumps({"event": "ingest_start", "distractor_articles": len(articles)}))
        with KuzuStore(db_path) as store:
            for title, sentences in articles.items():
                doc_id = hashlib.sha256(title.encode()).hexdigest()
                if store.document_exists(doc_id):
                    continue
                source_url = f"hotpotqa://{title.replace(' ', '_')}"
                source = SourceDocument(
                    id=doc_id,
                    path=Path("/dev/null"),
                    doc_type="hotpotqa_distractor",
                    source_url=source_url,
                )
                chunks = _sentences_to_chunks(title, sentences, doc_id, source_url)
                if chunks:
                    store.write_document_with_chunks(source, title, chunks)
        print(json.dumps({"event": "chunks_written"}))

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
# Short answer extraction (same prompt as run_hotpotqa.py)
# ---------------------------------------------------------------------------

_SHORT_ANSWER_PROMPT = (
    "Extract the answer from the response below.\n"
    "Output ONLY the bare answer — a name, phrase, number, or yes/no. No explanation, no sentence structure.\n"
    "- For yes/no questions output only: yes  or  no\n"
    "- Include the full proper noun or place name as written — do not truncate at a comma or 'and'.\n"
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
# Hallucination classification
# ---------------------------------------------------------------------------

def _is_acknowledged(answer: str) -> bool:
    """Return True if the model flagged its answer as general-knowledge fallback."""
    if _WARNING_SENTINEL in answer:
        return True
    lower = answer.lower()
    return any(phrase in lower for phrase in _WARNING_PHRASES)


def _classify(acknowledged: bool, em: float) -> str:
    if acknowledged:
        return "warned"
    if em == 1.0:
        return "confident_correct"
    return "hallucinated"


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
) -> list[dict]:
    session = open_session(db_path, embeddings_path, opts, embed_cfg, scoring_cfg, llm_cfg)
    rows: list[dict] = []
    try:
        for i, sample in enumerate(samples):
            question = sample["question"]
            gold_answer = sample["answer"]
            gold = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)

            verbose_answer, retrieval, elapsed = run_query(session, question)
            extracted = _extract_short_answer(session.llm, question, verbose_answer)

            metrics = extract_metrics(verbose_answer, retrieval)
            em = answer_em(extracted, gold)
            f1 = answer_f1(extracted, gold)
            acknowledged = _is_acknowledged(verbose_answer)
            category = _classify(acknowledged, em)

            row = {
                "profile": profile_name,
                "id": sample.get("_id", f"sample_{i}"),
                "question": question,
                "type": sample.get("type", "unknown"),
                "gold_answer": gold_answer,
                "gold_titles": sample.get("gold_titles", []),
                "alice_answer": verbose_answer,
                "extracted_answer": extracted,
                "answer_em": em,
                "answer_f1": round(f1, 4),
                "acknowledged": acknowledged,
                "category": category,
                "runtime_seconds": round(elapsed, 4),
                **metrics,
            }
            rows.append(row)

            print(json.dumps({
                "event": "question_done",
                "profile": profile_name,
                "i": i + 1,
                "total": len(samples),
                "category": category,
                "em": em,
                "question": question[:80],
            }))
    finally:
        close_session(session)
    return rows


def _hallucination_summary(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    counts = {"warned": 0, "confident_correct": 0, "hallucinated": 0}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        q_type = r.get("type", "unknown")
        by_type.setdefault(q_type, {"warned": 0, "confident_correct": 0, "hallucinated": 0})
        by_type[q_type][r["category"]] = by_type[q_type].get(r["category"], 0) + 1

    return {
        "total": n,
        "hallucination_rate": round(counts["hallucinated"] / n, 4),
        "acknowledgment_rate": round(counts["warned"] / n, 4),
        "confident_correct_rate": round(counts["confident_correct"] / n, 4),
        "counts": counts,
        "by_question_type": {
            qt: {
                "n": sum(c.values()),
                "hallucination_rate": round(c["hallucinated"] / max(sum(c.values()), 1), 4),
                "acknowledgment_rate": round(c["warned"] / max(sum(c.values()), 1), 4),
            }
            for qt, c in by_type.items()
        },
    }


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
        print("[error] No [chat_llm] in alice.toml.", file=sys.stderr)
        sys.exit(1)

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
    profile_names = args.profile or ["high_recall"]

    for p in profile_names:
        if p not in profile_map:
            print(f"[error] Unknown profile '{p}'. Available: {sorted(profile_map)}", file=sys.stderr)
            sys.exit(1)

    print(json.dumps({"event": "loading_dataset", "max_examples": args.max_examples}))
    examples = _load_hotpotqa(args.split, args.max_examples)

    # Build distractor corpus
    distractor_articles, stripped_count = _collect_distractor_articles(examples)
    total_context_pairs = sum(
        len((ex.get("context") or {}).get("title") or []) for ex in examples
    )
    cross_contamination_estimate = stripped_count / max(total_context_pairs, 1)
    print(json.dumps({
        "event": "corpus_built",
        "distractor_articles": len(distractor_articles),
        "gold_pairs_stripped": stripped_count,
        "cross_contamination_estimate": round(cross_contamination_estimate, 3),
    }))

    samples = [
        {
            "_id": ex.get("id") or ex.get("_id", f"s{i}"),
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex.get("type", "unknown"),
            "gold_titles": sorted(_supporting_titles(ex)),
        }
        for i, ex in enumerate(examples)
    ]

    # Ingestion
    if args.reuse_db and args.db_path.exists():
        _ingest_corpus(
            distractor_articles, args.db_path, args.embeddings_path,
            ingest_llm_cfg, embed_cfg, args.min_ingest_confidence,
            skip_chunk_writing=True,
        )
    else:
        if args.db_path.exists():
            shutil.rmtree(args.db_path) if args.db_path.is_dir() else args.db_path.unlink()
        if args.embeddings_path.exists():
            args.embeddings_path.unlink()
        _ingest_corpus(
            distractor_articles, args.db_path, args.embeddings_path,
            ingest_llm_cfg, embed_cfg, args.min_ingest_confidence,
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"hallucination_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(_REPO_ROOT),
        "max_examples": args.max_examples,
        "distractor_articles": len(distractor_articles),
        "gold_pairs_stripped": stripped_count,
        "cross_contamination_estimate": round(cross_contamination_estimate, 3),
        "profiles": profile_names,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    all_runs: list[dict] = []
    for profile_name in profile_names:
        raw_profile = dict(profile_map.get(profile_name) or {})
        opts = build_profile_opts(global_cfg, raw_profile)

        print(json.dumps({"event": "profile_start", "profile": profile_name}))
        rows = _run_profile(
            profile_name, opts, samples,
            args.db_path, args.embeddings_path,
            embed_cfg, scoring_cfg, chat_llm_cfg,
        )

        base_summary = summarize(rows, include_qa_metrics=True)
        hall_summary = _hallucination_summary(rows)
        full_summary = {**base_summary, **hall_summary}

        report = {
            "profile": profile_name,
            "profile_opts": {k: v for k, v in opts.items() if k not in ("enabled", "description")},
            "summary": full_summary,
            "results": rows,
        }
        base_name = f"hallucination__{profile_name}"
        (run_dir / f"{base_name}.json").write_text(json.dumps(report, indent=2))
        with (run_dir / f"{base_name}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        all_runs.append({"profile": profile_name, "summary": full_summary})
        print(json.dumps({"event": "profile_done", "profile": profile_name, "summary": full_summary}))

    (run_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "runs": all_runs}, indent=2)
    )
    print(json.dumps({"event": "done", "output": str(run_dir)}))


if __name__ == "__main__":
    main()
