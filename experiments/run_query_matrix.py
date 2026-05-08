#!/usr/bin/env python3
"""
Run a profile matrix of ALICE query experiments.

Executes the same question set across multiple retrieval profiles and writes
per-profile + aggregate reports for reproducible comparison.

Usage:
    uv run python experiments/run_query_matrix.py
    uv run python experiments/run_query_matrix.py --profile base --profile high_recall
    uv run python experiments/run_query_matrix.py --db-path services/experts/data/natalia_alexandrov/natalia_alexandrov.db
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._core import (
    answer_em, answer_f1, build_profile_opts, close_session, extract_metrics,
    git_head, load_profiles, open_session, run_query, summarize,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ALICE query experiment profile matrix.")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=_EXPERIMENTS_DIR / "configs" / "profiles.toml",
        help="TOML file defining global + per-profile query options.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=_EXPERIMENTS_DIR / "configs" / "questions.json",
        help="JSON file containing question list.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Profile name filter (repeat for multiple). Default: all enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_EXPERIMENTS_DIR / "results",
        help="Directory where run outputs are written.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to Kuzu DB. Defaults to alice.toml [chat] db_path.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Path to embeddings .npz. Defaults to alice.toml [chat] embeddings_path.",
    )
    return parser.parse_args()


def _load_questions(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("questions", payload) if isinstance(payload, dict) else payload
    rows: list[dict[str, str]] = []
    for idx, item in enumerate(source, start=1):
        if isinstance(item, dict):
            q_id = str(item.get("q_id") or f"Q{idx}")
            text = str(item.get("canonical") or item.get("text") or "").strip()
        else:
            q_id = f"Q{idx}"
            text = str(item).strip()
        if text:
            rows.append({"q_id": q_id, "text": text})
    if not rows:
        raise ValueError(f"No questions loaded from {path}")
    return rows


def _run_profile(
    profile_name: str,
    opts: dict[str, Any],
    questions: list[dict[str, str]],
    db_path: Path,
    embeddings_path: Path,
    embed_cfg,
    scoring_cfg,
    llm_cfg,
) -> dict[str, Any]:
    session = open_session(db_path, embeddings_path, opts, embed_cfg, scoring_cfg, llm_cfg)
    rows: list[dict[str, Any]] = []
    try:
        for q in questions:
            print(json.dumps({"event": "query_start", "profile": profile_name, "q_id": q["q_id"]}))
            answer, retrieval, elapsed = run_query(session, q["text"])
            metrics = extract_metrics(answer, retrieval)
            rows.append({
                "profile": profile_name,
                "q_id": q["q_id"],
                "question": q["text"],
                "runtime_seconds": round(elapsed, 4),
                "response_text": answer,
                **metrics,
            })
            print(json.dumps({
                "event": "query_complete",
                "profile": profile_name,
                "q_id": q["q_id"],
                "runtime_seconds": round(elapsed, 4),
                "facts_retrieved": metrics["facts_retrieved"],
            }))
    finally:
        close_session(session)

    return {
        "profile": profile_name,
        "profile_opts": {k: v for k, v in opts.items() if k not in ("enabled", "description")},
        "summary": summarize(rows),
        "results": rows,
    }


def main() -> None:
    args = _parse_args()

    from services.chat.config import load_chat_config
    chat_cfg, embed_cfg, scoring_cfg, llm_cfg = load_chat_config()

    db_path = args.db_path or chat_cfg.db_path
    embeddings_path = args.embeddings_path or chat_cfg.embeddings_path

    if not db_path.exists():
        print(f"[error] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not embeddings_path.exists():
        print(f"[error] Embeddings not found: {embeddings_path}", file=sys.stderr)
        sys.exit(1)

    profiles_payload = load_profiles(args.profiles)
    questions = _load_questions(args.questions)
    global_cfg = dict(profiles_payload.get("global") or {})
    profile_map = dict(profiles_payload.get("profiles") or {})
    selected = set(args.profile or [])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"query_matrix_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(_REPO_ROOT),
        "db_path": str(db_path),
        "embeddings_path": str(embeddings_path),
        "profiles_file": str(args.profiles),
        "questions_file": str(args.questions),
        "question_count": len(questions),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    all_summaries: list[dict[str, Any]] = []
    for profile_name, raw_cfg in profile_map.items():
        cfg = dict(raw_cfg or {})
        if not cfg.get("enabled", True):
            continue
        if selected and profile_name not in selected:
            continue

        description = str(cfg.get("description", "")).strip()
        print(json.dumps({"event": "profile_start", "profile": profile_name, "description": description}))

        opts = build_profile_opts(global_cfg, cfg)
        try:
            report = _run_profile(
                profile_name, opts, questions,
                db_path, embeddings_path, embed_cfg, scoring_cfg, llm_cfg,
            )
        except Exception as exc:
            print(json.dumps({"event": "profile_failed", "profile": profile_name, "error": str(exc)}))
            continue

        profile_json = run_dir / f"{profile_name}.json"
        profile_jsonl = run_dir / f"{profile_name}.jsonl"
        profile_json.write_text(json.dumps(report, indent=2))
        with profile_jsonl.open("w") as fh:
            for row in report["results"]:
                fh.write(json.dumps(row) + "\n")

        all_summaries.append({
            "profile": profile_name,
            "description": description,
            "summary": report["summary"],
            "report": str(profile_json),
        })
        print(json.dumps({
            "event": "profile_complete",
            "profile": profile_name,
            "summary": report["summary"],
            "report": str(profile_json),
        }))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({"manifest": manifest, "profiles": all_summaries}, indent=2))
    print(json.dumps({"event": "run_complete", "summary": str(summary_path)}))


if __name__ == "__main__":
    main()
