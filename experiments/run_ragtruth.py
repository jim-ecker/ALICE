#!/usr/bin/env python3
"""
Faithfulness evaluation for ALICE using RAGTruth QA.

RAGTruth (Niu et al. 2024) provides (context, query, output) tuples from
various LLMs with word-level hallucination annotations. We use the QA split
as a faithfulness benchmark for ALICE.

For each example:
  1. The retrieved context paragraphs are ingested into a per-question
     temporary KG (created fresh, deleted after the query).
  2. ALICE answers the query through its full retrieval pipeline.
  3. A GPT-4o faithfulness judge classifies the answer as:
       faithful      — all claims are supported by the provided context
       unfaithful    — at least one claim contradicts or is absent from context
       acknowledged  — ALICE prefixed with ⚠️ (correctly flagged missing context)

Hallucination rate = unfaithful / total (excluding acknowledged)

Results are compared against the RAGTruth paper's reported rates for
GPT-3.5-turbo and GPT-4 to establish a cross-system reference point.

Usage:
    uv run python experiments/run_ragtruth.py
    uv run python experiments/run_ragtruth.py --max-examples 100
    uv run python experiments/run_ragtruth.py --profile high_recall --profile trust_filtered
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._core import (
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
_WARNING_SENTINEL = "⚠️"
_WARNING_PHRASES = [
    "knowledge graph does not contain",
    "not contain information",
    "based on general knowledge",
    "general training knowledge",
]

_FAITHFULNESS_PROMPT = """\
You are a strict faithfulness judge. Given a CONTEXT and an ANSWER, decide
whether every factual claim in the answer is supported by the context.

Rules:
- If the answer only contains information from the context, output: faithful
- If the answer introduces facts not in the context (fabricates, contradicts,
  or adds detail not mentioned), output: unfaithful
- If the answer is a refusal or says it cannot answer, output: abstain
- Output exactly one word: faithful, unfaithful, or abstain

CONTEXT:
{context}

ANSWER:
{answer}

Verdict:"""


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGTruth faithfulness eval for ALICE.")
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--split", default="train", choices=["train"])
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--profiles", type=Path, default=_DEFAULT_PROFILES)
    p.add_argument("--min-ingest-confidence", type=float, default=0.6)
    p.add_argument("--output-dir", type=Path, default=_EXPERIMENTS_DIR / "results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_ragtruth_qa(split: str, max_examples: int) -> list[dict]:
    """Load RAGTruth QA examples, deduplicated by (query, context) pair.

    RAGTruth stores multiple model outputs per (query, context); we keep one
    row per unique pair (preferring rows with hallucination labels present).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: uv add datasets") from exc
    ds = load_dataset("wandb/RAGTruth-processed", split=split)
    seen: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        if r["task_type"] != "QA" or r["quality"] != "good":
            continue
        key = hashlib.sha256(f"{r['query']}|||{str(r['context'])[:500]}".encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(r))
        if len(rows) >= max_examples:
            break
    return rows


# ---------------------------------------------------------------------------
# Per-question KG ingestion
# ---------------------------------------------------------------------------

def _context_to_chunks(context: str, doc_id: str, source_url: str) -> list:
    from services.ingest.models import Chunk, ChunkProvenance

    sentences = [s.strip() for s in context.replace("\n\n", "\n").split("\n") if s.strip()]
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
                section_heading="context",
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


def _ingest_question(
    context: str,
    db_path: Path,
    embeddings_path: Path,
    ingest_llm_cfg: Any,
    embed_cfg: Any,
    min_ingest_confidence: float,
) -> bool:
    """Ingest one question's context into a fresh KG. Returns True on success."""
    from core.graph import KuzuStore
    from core.embeddings.client import EmbeddingsClient
    from services.ingest.models import SourceDocument
    from services.ingest.service import Ingest

    try:
        doc_id = hashlib.sha256(context[:200].encode()).hexdigest()
        source_url = f"ragtruth://{doc_id[:16]}"
        with KuzuStore(db_path) as store:
            source = SourceDocument(
                id=doc_id,
                path=Path("/dev/null"),
                doc_type="ragtruth_qa",
                source_url=source_url,
            )
            chunks = _context_to_chunks(context, doc_id, source_url)
            if chunks:
                store.write_document_with_chunks(source, "context", chunks)

        embed_client = EmbeddingsClient(embed_cfg)
        ingest = Ingest(
            db_path=db_path,
            llm_cfg=ingest_llm_cfg,
            embed_client=embed_client,
            embeddings_path=embeddings_path,
        )
        ingest.extract(min_ingest_confidence=min_ingest_confidence)
        ingest.build_index()
        return True
    except Exception as e:
        print(json.dumps({"event": "ingest_error", "error": str(e)[:200]}))
        return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _is_acknowledged(answer: str) -> bool:
    if _WARNING_SENTINEL in answer:
        return True
    lower = answer.lower()
    return any(phrase in lower for phrase in _WARNING_PHRASES)


def _faithfulness_verdict(llm: Any, context: str, answer: str) -> str:
    """Ask GPT-4o to judge whether the answer is faithful to the context."""
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context[:3000],
        answer=answer[:1500],
    )
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=8).strip().lower()
        if "unfaithful" in raw:
            return "unfaithful"
        if "faithful" in raw:
            return "faithful"
        return "abstain"
    except Exception:
        return "unknown"


def _classify(acknowledged: bool, faithfulness: str) -> str:
    if acknowledged:
        return "acknowledged"
    if faithfulness == "unfaithful":
        return "hallucinated"
    if faithfulness == "faithful":
        return "faithful"
    return "abstain"


# ---------------------------------------------------------------------------
# Per-profile evaluation
# ---------------------------------------------------------------------------

def _run_profile(
    profile_name: str,
    opts: dict[str, Any],
    samples: list[dict],
    embed_cfg: Any,
    scoring_cfg: Any,
    llm_cfg: Any,
    ingest_llm_cfg: Any,
    min_ingest_confidence: float,
) -> list[dict]:
    rows: list[dict] = []

    for i, sample in enumerate(samples):
        query = sample["query"]
        context = sample["context"]
        gold_labels = sample["hallucination_labels_processed"]

        tmp = tempfile.mkdtemp(prefix="alice_ragtruth_")
        db_path = Path(tmp) / "ragtruth.db"
        emb_path = Path(tmp) / "ragtruth.embeddings.npz"

        try:
            ok = _ingest_question(
                context, db_path, emb_path,
                ingest_llm_cfg, embed_cfg, min_ingest_confidence,
            )
            if not ok:
                continue

            session = open_session(db_path, emb_path, opts, embed_cfg, scoring_cfg, llm_cfg)
            try:
                t0 = time.perf_counter()
                verbose_answer, retrieval, elapsed = run_query(session, query)
            finally:
                close_session(session)

            metrics = extract_metrics(verbose_answer, retrieval)
            acknowledged = _is_acknowledged(verbose_answer)
            faithfulness = _faithfulness_verdict(session.llm, context, verbose_answer)
            category = _classify(acknowledged, faithfulness)

            # Gold: did the original model hallucinate on this example?
            gold_hallucinated = (
                gold_labels.get("evident_conflict", 0) > 0
                or gold_labels.get("baseless_info", 0) > 0
            )

            row = {
                "profile": profile_name,
                "id": sample.get("id", f"s{i}"),
                "query": query,
                "alice_answer": verbose_answer,
                "faithfulness": faithfulness,
                "acknowledged": acknowledged,
                "category": category,
                "gold_hallucinated": gold_hallucinated,
                "gold_labels": gold_labels,
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
                "faithfulness": faithfulness,
                "gold_hallucinated": gold_hallucinated,
                "query": query[:60],
            }))

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return rows


def _faithfulness_summary(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    # Among non-acknowledged responses, what fraction are unfaithful?
    non_ack = [r for r in rows if not r["acknowledged"]]
    halluc_rate = sum(1 for r in non_ack if r["category"] == "hallucinated") / max(len(non_ack), 1)

    # Correlation: on examples where original model hallucinated, does ALICE too?
    gold_halluc = [r for r in rows if r["gold_hallucinated"]]
    gold_clean = [r for r in rows if not r["gold_hallucinated"]]

    return {
        "total": n,
        "hallucination_rate": round(halluc_rate, 4),
        "acknowledgment_rate": round(counts.get("acknowledged", 0) / n, 4),
        "faithfulness_rate": round(counts.get("faithful", 0) / n, 4),
        "counts": counts,
        "on_gold_hallucinated": {
            "n": len(gold_halluc),
            "alice_hallucination_rate": round(
                sum(1 for r in gold_halluc if r["category"] == "hallucinated") / max(len(gold_halluc), 1), 4
            ),
        },
        "on_gold_clean": {
            "n": len(gold_clean),
            "alice_hallucination_rate": round(
                sum(1 for r in gold_clean if r["category"] == "hallucinated") / max(len(gold_clean), 1), 4
            ),
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
    samples = _load_ragtruth_qa(args.split, args.max_examples)
    print(json.dumps({"event": "dataset_loaded", "samples": len(samples)}))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"ragtruth_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(_REPO_ROOT),
        "max_examples": args.max_examples,
        "samples": len(samples),
        "profiles": profile_names,
        "note": "Per-question temp KG. Faithfulness judged by GPT-4o.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    all_runs: list[dict] = []
    for profile_name in profile_names:
        raw_profile = dict(profile_map.get(profile_name) or {})
        opts = build_profile_opts(global_cfg, raw_profile)

        print(json.dumps({"event": "profile_start", "profile": profile_name, "samples": len(samples)}))
        rows = _run_profile(
            profile_name, opts, samples,
            embed_cfg, scoring_cfg, chat_llm_cfg,
            ingest_llm_cfg, args.min_ingest_confidence,
        )

        base_summary = summarize(rows)
        faith_summary = _faithfulness_summary(rows)
        full_summary = {**base_summary, **faith_summary}

        report = {
            "profile": profile_name,
            "profile_opts": {k: v for k, v in opts.items() if k not in ("enabled", "description")},
            "summary": full_summary,
            "results": rows,
        }
        base_name = f"ragtruth__{profile_name}"
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
