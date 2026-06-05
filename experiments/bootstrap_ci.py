#!/usr/bin/env python3
"""
Bootstrap confidence intervals for ALICE benchmark results.

Computes 95% CI for EM and F1 across all profiles for each dataset/model run,
and tests pairwise differences between profiles and between model sizes.

Usage:
    uv run python experiments/bootstrap_ci.py
    uv run python experiments/bootstrap_ci.py --n-bootstrap 20000 --alpha 0.05
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple

_EXPERIMENTS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

RUNS = {
    "musique_14b": {
        "label": "MuSiQue (14B)",
        "dir": _EXPERIMENTS_DIR / "results" / "multihop_musique_20260521_095331",
        "prefix": "multihop_musique__",
        "profiles": ["high_recall", "trust_filtered", "path_trust", "path_trust_filtered"],
    },
    "musique_32b": {
        "label": "MuSiQue (32B)",
        "dir": _EXPERIMENTS_DIR / "results" / "multihop_musique_20260519_002909",
        "prefix": "multihop_musique__",
        "profiles": ["high_recall", "trust_filtered", "path_trust", "path_trust_filtered"],
    },
    "musique_72b": {
        "label": "MuSiQue (72B)",
        "dir": _EXPERIMENTS_DIR / "results" / "multihop_musique_20260519_164737",
        "prefix": "multihop_musique__",
        "profiles": ["high_recall", "trust_filtered", "path_trust", "path_trust_filtered"],
    },
    "wiki_14b": {
        "label": "2WikiMultihopQA (14B)",
        "dir": _EXPERIMENTS_DIR / "results" / "multihop_2wikimultihopqa_20260521_180924",
        "prefix": "multihop_2wikimultihopqa__",
        "profiles": ["high_recall", "trust_filtered", "path_trust", "path_trust_filtered"],
    },
    "wiki_32b": {
        "label": "2WikiMultihopQA (32B)",
        "dir": _EXPERIMENTS_DIR / "results" / "multihop_2wikimultihopqa_20260519_084025",
        "prefix": "multihop_2wikimultihopqa__",
        "profiles": ["high_recall", "trust_filtered", "path_trust", "path_trust_filtered"],
    },
    "wiki_72b": {
        "label": "2WikiMultihopQA (72B)",
        "dir": _EXPERIMENTS_DIR / "results" / "multihop_2wikimultihopqa_20260520_044806",
        "prefix": "multihop_2wikimultihopqa__",
        "profiles": ["high_recall", "trust_filtered", "path_trust", "path_trust_filtered"],
    },
    "hotpot_72b": {
        "label": "HotpotQA distractor (72B)",
        "dir": _EXPERIMENTS_DIR / "results" / "hotpotqa_20260520_172057",
        "prefix": "hotpotqa__",
        "profiles": ["high_recall", "trust_filtered", "path_trust", "path_trust_filtered"],
    },
}

# Pairwise comparisons to test: (run_key_a, profile_a, run_key_b, profile_b, label)
COMPARISONS = [
    # MuSiQue: model size effect (best profile)
    ("musique_32b", "path_trust_filtered", "musique_72b", "trust_filtered",
     "MuSiQue: 32B path_trust_filtered vs 72B trust_filtered"),
    ("musique_32b", "high_recall", "musique_72b", "high_recall",
     "MuSiQue: 32B high_recall vs 72B high_recall"),
    # MuSiQue: profile effect within 72B
    ("musique_72b", "high_recall", "musique_72b", "trust_filtered",
     "MuSiQue 72B: high_recall vs trust_filtered"),
    ("musique_72b", "high_recall", "musique_72b", "path_trust_filtered",
     "MuSiQue 72B: high_recall vs path_trust_filtered"),
    # 2Wiki: model size effect
    ("wiki_32b", "path_trust_filtered", "wiki_72b", "path_trust",
     "2Wiki: 32B path_trust_filtered vs 72B path_trust"),
    ("wiki_32b", "high_recall", "wiki_72b", "path_trust",
     "2Wiki: 32B high_recall vs 72B path_trust"),
    # 2Wiki: profile flip (key TPPR interaction claim)
    ("wiki_72b", "path_trust", "wiki_72b", "high_recall",
     "2Wiki 72B: path_trust vs high_recall"),
    ("wiki_72b", "path_trust", "wiki_72b", "path_trust_filtered",
     "2Wiki 72B: path_trust vs path_trust_filtered"),
    # HotpotQA: 72B profiles
    ("hotpot_72b", "high_recall", "hotpot_72b", "trust_filtered",
     "HotpotQA 72B: high_recall vs trust_filtered"),
    ("hotpot_72b", "path_trust_filtered", "hotpot_72b", "high_recall",
     "HotpotQA 72B: path_trust_filtered vs high_recall"),
]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class CI(NamedTuple):
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.1%}  [{self.lo:.1%}, {self.hi:.1%}]  (n={self.n})"


def bootstrap_ci(scores: list[float], n_bootstrap: int, alpha: float, seed: int = 42) -> CI:
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    return CI(
        mean=sum(scores) / n,
        lo=means[lo_idx],
        hi=means[hi_idx],
        n=n,
    )


def load_scores(run_cfg: dict, profile: str, metric: str = "answer_em") -> list[float]:
    path = run_cfg["dir"] / f"{run_cfg['prefix']}{profile}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [float(r[metric]) for r in rows if metric in r]


def diff_significant(scores_a: list[float], scores_b: list[float],
                     n_bootstrap: int, seed: int = 42) -> tuple[float, float, float]:
    """
    Paired bootstrap test for mean(a) - mean(b).
    Returns (observed_diff, lo_95ci, hi_95ci).
    Significant if CI excludes 0.
    """
    assert len(scores_a) == len(scores_b), "Paired test requires equal-length score lists"
    rng = random.Random(seed)
    n = len(scores_a)
    diffs = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        da = sum(scores_a[i] for i in idxs) / n
        db = sum(scores_b[i] for i in idxs) / n
        diffs.append(da - db)
    diffs.sort()
    lo_idx = int(n_bootstrap * 0.025)
    hi_idx = int(n_bootstrap * 0.975)
    observed = sum(scores_a) / n - sum(scores_b) / n
    return observed, diffs[lo_idx], diffs[hi_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--metric", default="answer_em", choices=["answer_em", "answer_f1"])
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    metric_label = "EM" if args.metric == "answer_em" else "F1"

    results: dict = {"metric": args.metric, "n_bootstrap": args.n_bootstrap, "runs": {}, "comparisons": []}

    print(f"\n{'='*70}")
    print(f"BOOTSTRAP 95% CI — {metric_label}  (n_bootstrap={args.n_bootstrap})")
    print(f"{'='*70}")

    all_scores: dict[tuple[str, str], list[float]] = {}

    for run_key, run_cfg in RUNS.items():
        print(f"\n--- {run_cfg['label']} ---")
        results["runs"][run_key] = {}
        for profile in run_cfg["profiles"]:
            try:
                scores = load_scores(run_cfg, profile, args.metric)
            except FileNotFoundError:
                print(f"  {profile:25s}  [file not found]")
                continue
            ci = bootstrap_ci(scores, args.n_bootstrap, args.alpha)
            all_scores[(run_key, profile)] = scores
            results["runs"][run_key][profile] = {
                "mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi, "n": ci.n,
            }
            sig = "  ← " if profile == "high_recall" else ""
            print(f"  {profile:25s}  {ci}{sig}")

    print(f"\n{'='*70}")
    print("PAIRWISE COMPARISONS (A vs B, positive = A is better)")
    print(f"{'='*70}")

    for run_a, prof_a, run_b, prof_b, label in COMPARISONS:
        scores_a = all_scores.get((run_a, prof_a))
        scores_b = all_scores.get((run_b, prof_b))
        if scores_a is None or scores_b is None:
            print(f"\n  [skip] {label} — scores not loaded")
            continue

        # For same-run comparisons, use paired bootstrap; cross-run use unpaired
        same_run = run_a == run_b
        if same_run:
            obs, lo, hi = diff_significant(scores_a, scores_b, args.n_bootstrap)
        else:
            # Unpaired: bootstrap difference of means
            rng = random.Random(42)
            n_a, n_b = len(scores_a), len(scores_b)
            diffs = []
            for _ in range(args.n_bootstrap):
                sa = sum(scores_a[rng.randrange(n_a)] for _ in range(n_a)) / n_a
                sb = sum(scores_b[rng.randrange(n_b)] for _ in range(n_b)) / n_b
                diffs.append(sa - sb)
            diffs.sort()
            obs = sum(scores_a) / n_a - sum(scores_b) / n_b
            lo, hi = diffs[int(args.n_bootstrap * 0.025)], diffs[int(args.n_bootstrap * 0.975)]

        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        direction = "paired" if same_run else "unpaired"
        print(f"\n  {label}")
        print(f"    Δ{metric_label} = {obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig} ({direction})")
        results["comparisons"].append({
            "label": label, "delta": obs, "ci_lo": lo, "ci_hi": hi,
            "significant": (lo > 0 or hi < 0), "direction": direction,
        })

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
