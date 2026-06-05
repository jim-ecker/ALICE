#!/usr/bin/env python3
"""
Evaluate ALICE against multi-hop QA benchmarks for Graph RAG architecture comparison.

Supported datasets:
  2wikimultihopqa  — xanhho/2WikiMultihopQA (bridge, comparison, inference, compositional)
  musique          — dgslibisey/MuSiQue (bridge-style chains of 2–4 hops)

Published baselines for comparison:
  2WikiMultihopQA: IRCoT (GPT-3) ~71% EM, HippoRAG ~60% EM
  MuSiQue:         IRCoT (GPT-3) ~38% EM, HippoRAG ~30–35% EM

Usage:
    uv run python experiments/run_multihop.py --dataset 2wikimultihopqa --max-examples 200
    uv run python experiments/run_multihop.py --dataset musique --max-examples 200
    uv run python experiments/run_multihop.py --dataset 2wikimultihopqa --reuse-db

Requires: uv add datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
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
_DEFAULT_PROFILES = _EXPERIMENTS_DIR / "configs" / "profiles.toml"
_SENTENCES_PER_CHUNK = 4

_DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    # 2WikiMultihopQA uses a legacy HF dataset script incompatible with datasets>=3.0.
    # Download the raw JSON from the original release and point --local-data-file at it:
    #   https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip  (official release)
    # Extract dev.json and pass: --dataset 2wikimultihopqa --local-data-file path/to/dev.json
    "2wikimultihopqa": {
        "hf_name": None,
        "type_values": ("bridge", "comparison", "inference", "compositional"),
    },
    "musique": {
        "hf_name": "dgslibisey/MuSiQue",
        "type_values": (),
    },
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ALICE against multi-hop QA benchmarks."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(_DATASET_CONFIGS),
        help="Dataset to evaluate on.",
    )
    parser.add_argument("--max-examples", type=int, default=200,
                        help="Number of examples to evaluate (default: 200).")
    parser.add_argument("--split", default="validation", choices=["validation", "train", "test"])
    parser.add_argument("--profile", action="append", default=[],
                        help="Retrieval profile(s) to evaluate. Repeatable. Default: high_recall, trust_filtered.")
    parser.add_argument("--profiles", type=Path, default=_DEFAULT_PROFILES)
    parser.add_argument("--db-path", type=Path, default=None,
                        help="DB path. Default: experiments/data/<dataset>/<dataset>.db")
    parser.add_argument("--embeddings-path", type=Path, default=None,
                        help="Embeddings path. Default: experiments/data/<dataset>/<dataset>.embeddings.npz")
    parser.add_argument("--local-data-file", type=Path, default=None,
                        help="Path to a local JSON data file (required for 2wikimultihopqa).")
    parser.add_argument("--reuse-db", action="store_true",
                        help="Skip ingestion if DB already exists.")
    parser.add_argument("--no-extract-answer", action="store_true",
                        help="Score verbose ALICE output directly instead of extracting a short answer.")
    parser.add_argument("--min-ingest-confidence", type=float, default=0.6)
    parser.add_argument("--output-dir", type=Path, default=_EXPERIMENTS_DIR / "results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_local_json(path: Path, max_examples: int) -> list[dict]:
    """Load examples from a local JSON file (list or dict with a list value)."""
    import json as _json
    raw = _json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        # Some releases wrap data under a key, e.g. {"data": [...]}
        rows = next(v for v in raw.values() if isinstance(v, list))
    else:
        raise ValueError(f"Unrecognised JSON structure in {path}")
    return rows[:max_examples]


def _load_hf(hf_name: str, split: str, max_examples: int, hf_config: str | None = None) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The `datasets` package is required. Install it with: uv add datasets") from exc

    load_kwargs: dict = {"split": split}
    if hf_config:
        load_kwargs["name"] = hf_config

    ds = load_dataset(hf_name, **load_kwargs)
    rows: list[dict] = []
    for row in ds:
        rows.append(dict(row))
        if len(rows) >= max_examples:
            break
    return rows


def _examples_2wiki(rows: list[dict]) -> list[dict]:
    """Normalize 2WikiMultihopQA rows → (id, question, answer, type, context_pairs)."""
    out: list[dict] = []
    for row in rows:
        ctx = row.get("context") or []
        pairs: list[tuple[str, list[str]]] = []
        for item in ctx:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sents = item
                pairs.append((str(title), [str(s) for s in sents if s]))
            elif isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences") or []
                pairs.append((str(title), [str(s) for s in sents if s]))
        out.append({
            "id": str(row.get("_id") or row.get("id", "")),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "type": str(row.get("type", "unknown")),
            "context_pairs": pairs,
        })
    return out


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences on sentence-ending punctuation."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _examples_musique(rows: list[dict]) -> list[dict]:
    """Normalize MuSiQue rows → (id, question, answer, context_pairs)."""
    out: list[dict] = []
    for row in rows:
        paragraphs = row.get("paragraphs") or []
        pairs: list[tuple[str, list[str]]] = []
        for para in paragraphs:
            if not isinstance(para, dict):
                continue
            title = str(para.get("title", ""))
            text = str(para.get("paragraph_text", ""))
            sents = _split_sentences(text) or [text]
            pairs.append((title, sents))
        out.append({
            "id": str(row.get("id", "")),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "type": None,
            "context_pairs": pairs,
        })
    return out


# ---------------------------------------------------------------------------
# Context ingestion (mirrors run_hotpotqa.py)
# ---------------------------------------------------------------------------

def _collect_articles(examples: list[dict]) -> dict[str, list[str]]:
    articles: dict[str, list[str]] = {}
    for ex in examples:
        for title, sents in ex["context_pairs"]:
            if title not in articles:
                articles[title] = [s.strip() for s in sents if s.strip()]
    return articles


def _sentences_to_chunks(title: str, sentences: list[str], doc_id: str, source_url: str) -> list:
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
                slug = title.replace(" ", "_")
                source_url = f"multihop://{slug}"
                source = SourceDocument(
                    id=doc_id,
                    path=Path("/dev/null"),
                    doc_type="multihop",
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
# Short answer extraction (kept in sync with run_hotpotqa.py)
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
# Per-profile evaluation (mirrors run_hotpotqa.py)
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
                "id": sample["id"],
                "question": question,
                "type": sample.get("type"),
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


def _type_breakdown(rows: list[dict], type_values: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for q_type in type_values:
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
    ds_cfg = _DATASET_CONFIGS[args.dataset]

    data_dir = _EXPERIMENTS_DIR / "data" / args.dataset
    db_path = args.db_path or (data_dir / f"{args.dataset}.db")
    embeddings_path = args.embeddings_path or (data_dir / f"{args.dataset}.embeddings.npz")

    chat_cfg, embed_cfg, scoring_cfg, chat_llm_cfg = load_chat_config()
    if chat_llm_cfg is None:
        print("[error] No [chat_llm] in alice.toml — required for evaluation.", file=sys.stderr)
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
    profile_names = args.profile or ["high_recall", "trust_filtered"]

    for p in profile_names:
        if p not in profile_map:
            print(f"[error] Unknown profile '{p}'. Available: {sorted(profile_map)}", file=sys.stderr)
            sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"multihop_{args.dataset}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "event": "loading_dataset",
        "dataset": args.dataset,
        "hf_name": ds_cfg["hf_name"],
        "split": args.split,
        "max_examples": args.max_examples,
    }))
    if ds_cfg["hf_name"] is None:
        if not args.local_data_file:
            print(
                f"[error] --dataset {args.dataset} requires --local-data-file.\n"
                f"        Download dev.json from the official release and pass its path.",
                file=sys.stderr,
            )
            sys.exit(1)
        raw_rows = _load_local_json(args.local_data_file, args.max_examples)
    else:
        raw_rows = _load_hf(ds_cfg["hf_name"], args.split, args.max_examples, ds_cfg.get("hf_config"))

    if args.dataset == "2wikimultihopqa":
        examples = _examples_2wiki(raw_rows)
    else:
        examples = _examples_musique(raw_rows)

    samples = [
        {"id": ex["id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type")}
        for ex in examples
    ]
    print(json.dumps({"event": "dataset_loaded", "samples": len(samples)}))

    if args.reuse_db and db_path.exists():
        _ingest_contexts(
            examples, db_path, embeddings_path,
            ingest_llm_cfg, embed_cfg, args.min_ingest_confidence,
            skip_chunk_writing=True,
        )
    else:
        if db_path.exists():
            if db_path.is_dir():
                shutil.rmtree(db_path)
            else:
                db_path.unlink()
        if embeddings_path.exists():
            embeddings_path.unlink()
        _ingest_contexts(
            examples, db_path, embeddings_path,
            ingest_llm_cfg, embed_cfg, args.min_ingest_confidence,
        )

    if not db_path.exists():
        print(f"[error] DB missing after ingestion: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not embeddings_path.exists():
        print(f"[error] Embeddings missing after ingestion: {embeddings_path}", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(_REPO_ROOT),
        "dataset": args.dataset,
        "hf_name": ds_cfg["hf_name"],
        "split": args.split,
        "max_examples": args.max_examples,
        "db_path": str(db_path),
        "embeddings_path": str(embeddings_path),
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
            db_path, embeddings_path,
            embed_cfg, scoring_cfg, chat_llm_cfg,
            extract_answer=not args.no_extract_answer,
        )

        summary = summarize(rows, include_qa_metrics=True)
        summary.update(_type_breakdown(rows, ds_cfg["type_values"]))

        report = {
            "profile": profile_name,
            "profile_opts": {k: v for k, v in opts.items() if k not in ("enabled", "description")},
            "summary": summary,
            "results": rows,
        }
        file_base = f"multihop_{args.dataset}__{profile_name}"
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
