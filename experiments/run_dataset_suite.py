#!/usr/bin/env python3
"""
Run ALICE against QA datasets with answer-quality metrics (EM / F1).

Supports local JSON, local JSONL, and Hugging Face datasets (optional).

Usage:
    uv run python experiments/run_dataset_suite.py
    uv run python experiments/run_dataset_suite.py --profile base --dataset local_eval
    uv run python experiments/run_dataset_suite.py --db-path services/experts/data/natalia_alexandrov/natalia_alexandrov.db
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._core import (
    answer_em, answer_f1, build_profile_opts, close_session, extract_metrics,
    git_head, load_profiles, open_session, run_query, summarize,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ALICE dataset benchmark suite.")
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=_EXPERIMENTS_DIR / "configs" / "dataset_suite.toml",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=_EXPERIMENTS_DIR / "configs" / "profiles.toml",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Filter to named dataset(s).")
    parser.add_argument("--profile", action="append", default=[], help="Filter to named profile(s).")
    parser.add_argument("--max-examples", type=int, default=0, help="Per-dataset cap (0 = config value).")
    parser.add_argument("--output-dir", type=Path, default=_EXPERIMENTS_DIR / "results")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail on any dataset load error.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_json(cfg: dict) -> list[dict]:
    path = Path(str(cfg.get("path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [dict(r) for r in payload["data"] if isinstance(r, dict)]
    if isinstance(payload, list):
        return [dict(r) for r in payload if isinstance(r, dict)]
    raise ValueError(f"Unsupported JSON format: {path}")


def _load_jsonl(cfg: dict) -> list[dict]:
    path = Path(str(cfg.get("path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_hf(cfg: dict) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install `datasets` with: uv add datasets") from exc
    name = str(cfg.get("hf_name") or "")
    subset = cfg.get("hf_subset")
    split = str(cfg.get("split") or "validation")
    ds = load_dataset(name, split=split) if subset is None else load_dataset(name, str(subset), split=split)
    return [dict(row) for row in ds]


def _load_rows(cfg: dict) -> list[dict]:
    loader = str(cfg.get("loader") or "").lower()
    if loader == "json":
        return _load_json(cfg)
    if loader == "jsonl":
        return _load_jsonl(cfg)
    if loader == "hf":
        return _load_hf(cfg)
    raise ValueError(f"Unknown loader: {loader!r}")


def _field(obj: dict, path: str, default=None):
    current = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _ensure_answers(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_samples(rows: list[dict], cfg: dict, max_examples: int) -> list[dict]:
    id_f = str(cfg.get("id_field") or "id")
    q_f = str(cfg.get("question_field") or "question")
    a_f = str(cfg.get("answers_field") or "answers")
    out: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        question = str(_field(row, q_f, "") or "").strip()
        if not question:
            continue
        answers = _ensure_answers(_field(row, a_f))
        sample_id = str(_field(row, id_f, f"{cfg.get('name', 'sample')}_{idx}"))
        out.append({"id": sample_id, "question": question, "answers": answers})
        if max_examples > 0 and len(out) >= max_examples:
            break
    return out


# ---------------------------------------------------------------------------
# Per-combination runner
# ---------------------------------------------------------------------------

def _run_one(
    dataset_name: str,
    profile_name: str,
    opts: dict[str, Any],
    samples: list[dict],
    db_path: Path,
    embeddings_path: Path,
    embed_cfg,
    scoring_cfg,
    llm_cfg,
) -> dict[str, Any]:
    session = open_session(db_path, embeddings_path, opts, embed_cfg, scoring_cfg, llm_cfg)
    rows: list[dict[str, Any]] = []
    try:
        for sample in samples:
            answer, retrieval, elapsed = run_query(session, sample["question"])
            metrics = extract_metrics(answer, retrieval)
            gold = sample.get("answers") or []
            rows.append({
                "dataset": dataset_name,
                "profile": profile_name,
                "id": sample["id"],
                "question": sample["question"],
                "answers": gold,
                "prediction": answer,
                "answer_em": answer_em(answer, gold),
                "answer_f1": answer_f1(answer, gold),
                "runtime_seconds": round(elapsed, 4),
                **metrics,
            })
    finally:
        close_session(session)

    return {
        "dataset": dataset_name,
        "profile": profile_name,
        "profile_opts": {k: v for k, v in opts.items() if k not in ("enabled", "description")},
        "summary": summarize(rows, include_qa_metrics=True),
        "results": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import tomllib

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

    suite = tomllib.loads(args.suite_config.read_text(encoding="utf-8"))
    profiles_payload = load_profiles(args.profiles)

    global_suite = dict(suite.get("global") or {})
    dataset_entries = list(suite.get("datasets") or [])
    if not dataset_entries:
        raise ValueError(f"No [[datasets]] in {args.suite_config}")

    profile_names = list(global_suite.get("profile_names") or [])
    if args.profile:
        profile_names = [p for p in profile_names if p in set(args.profile)]
    if not profile_names:
        raise ValueError("No profiles selected.")

    global_cfg = dict(profiles_payload.get("global") or {})
    profile_map = dict(profiles_payload.get("profiles") or {})

    global_max = int(global_suite.get("max_examples_per_dataset", 0) or 0)
    max_examples = int(args.max_examples or 0) or global_max

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"dataset_suite_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(_REPO_ROOT),
        "db_path": str(db_path),
        "embeddings_path": str(embeddings_path),
        "suite_config": str(args.suite_config),
        "profiles_config": str(args.profiles),
        "profile_names": profile_names,
        "max_examples_override": max_examples,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    selected_datasets = set(args.dataset or [])
    final_rows: list[dict] = []
    failures: list[dict] = []

    for raw_cfg in dataset_entries:
        cfg = dict(raw_cfg or {})
        name = str(cfg.get("name") or "").strip()
        if not name or (selected_datasets and name not in selected_datasets):
            continue
        if not cfg.get("enabled", True):
            continue

        print(json.dumps({"event": "dataset_load_start", "dataset": name}))
        try:
            raw_rows = _load_rows(cfg)
            samples = _normalize_samples(raw_rows, cfg, max_examples)
            if not samples:
                raise ValueError(f"Dataset '{name}' yielded zero samples.")
        except Exception as exc:
            failures.append({"dataset": name, "error": str(exc)})
            print(json.dumps({"event": "dataset_load_failed", "dataset": name, "error": str(exc)}))
            if args.strict:
                raise
            continue

        print(json.dumps({"event": "dataset_loaded", "dataset": name, "samples": len(samples)}))

        for profile_name in profile_names:
            raw_profile = dict((profile_map.get(profile_name) or {}))
            opts = build_profile_opts(global_cfg, raw_profile)

            print(json.dumps({"event": "run_start", "dataset": name, "profile": profile_name, "samples": len(samples)}))
            try:
                report = _run_one(name, profile_name, opts, samples, db_path, embeddings_path, embed_cfg, scoring_cfg, llm_cfg)
            except Exception as exc:
                failures.append({"dataset": name, "profile": profile_name, "error": str(exc)})
                print(json.dumps({"event": "run_failed", "dataset": name, "profile": profile_name, "error": str(exc)}))
                if args.strict:
                    raise
                continue

            file_base = f"{name}__{profile_name}"
            (run_dir / f"{file_base}.json").write_text(json.dumps(report, indent=2))
            with (run_dir / f"{file_base}.jsonl").open("w") as fh:
                for row in report["results"]:
                    fh.write(json.dumps(row) + "\n")

            final_rows.append({
                "dataset": name,
                "profile": profile_name,
                "summary": report["summary"],
                "report": str(run_dir / f"{file_base}.json"),
            })
            print(json.dumps({"event": "run_complete", "dataset": name, "profile": profile_name, "summary": report["summary"]}))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({"manifest": manifest, "runs": final_rows, "failures": failures}, indent=2))
    print(json.dumps({"event": "suite_complete", "summary": str(summary_path)}))


if __name__ == "__main__":
    main()
