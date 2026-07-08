#!/usr/bin/env python3
"""Fit the calibrated trust combiner from minted labels and write a report.

Reads labels.jsonl (from scripts/mint_calibration_labels.py), fits logistic
regression + isotonic calibration over the signal vector, and writes:
  - calibrator.json       — the deployable artifact (Phase 3 consumes this)
  - calibration_report.md — metrics, reliability table, coefficients, and the
    PPI-corrected correctness rate with CI when human audit labels exist.

Usage:
    uv run python3 scripts/fit_trust_calibrator.py \
        [--labels experiments/data/calibration/labels.jsonl] \
        [--out-dir experiments/data/calibration]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LABELS = REPO_ROOT / "experiments/data/calibration/labels.jsonl"


def load_rows(labels_path: Path) -> list[dict]:
    return [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]


def fit_and_report(rows: list[dict], out_dir: Path, labels_path: Path) -> dict:
    from core.trust.calibration import FEATURES, fit_calibrator, save_artifact
    from core.trust.ppi import cohens_kappa, ppi_mean

    # Human label wins over the judge where both exist; unclear rows are excluded
    usable, labels = [], []
    n_unclear = 0
    for row in rows:
        label = row["human_label"] if row.get("human_label") is not None else row.get("label")
        if label is None:
            n_unclear += 1
            continue
        usable.append(row)
        labels.append(int(label))

    judge_models = sorted({r.get("judge_model", "?") for r in rows})
    result = fit_calibrator(
        usable,
        labels,
        metadata={
            "labels_file": str(labels_path),
            "judge_models": judge_models,
            "n_unclear_excluded": n_unclear,
        },
    )
    metrics = result.metrics

    # PPI: bias-corrected correctness rate anchored to the human audit
    audit = [r for r in rows if r.get("human_audited") and r.get("human_label") is not None
             and r.get("label") is not None]
    ppi = kappa = None
    if len(audit) >= 10:
        judge_all = [float(r["label"]) for r in rows if r.get("label") is not None]
        ppi = ppi_mean(
            judge_all,
            [float(r["label"]) for r in audit],
            [float(r["human_label"]) for r in audit],
        )
        kappa = cohens_kappa([r["label"] for r in audit], [r["human_label"] for r in audit])

    out_dir.mkdir(parents=True, exist_ok=True)
    save_artifact(result.artifact, out_dir / "calibrator.json")

    lines = [
        "# Trust calibration report",
        "",
        f"- labels: `{labels_path}` — {len(rows)} rows, {len(usable)} usable "
        f"({n_unclear} unclear excluded), base rate {metrics['base_rate']:.3f}",
        f"- judge model(s): {', '.join(judge_models)}",
        f"- fit: {metrics['n_folds']}-fold out-of-fold logistic + isotonic "
        f"(calibration metrics on a {metrics['n_eval_holdout']}-row stratified holdout)",
        "",
        "## Metrics (held out)",
        "",
        f"| metric | uncalibrated | calibrated |",
        f"|---|---|---|",
        f"| ECE | {metrics['ece_uncalibrated']:.4f} | {metrics['ece_calibrated']:.4f} |",
        f"| Brier | {metrics['brier_uncalibrated']:.4f} | {metrics['brier_calibrated']:.4f} |",
        f"| AUROC | {metrics['auroc']:.4f} | — |",
        "",
        "## Reliability (calibrated, holdout)",
        "",
        "| bin | n | mean predicted | empirical accuracy |",
        "|---|---|---|---|",
    ]
    for bin_row in metrics["reliability"]:
        pred = f"{bin_row['mean_predicted']:.3f}" if bin_row["mean_predicted"] is not None else "—"
        acc = f"{bin_row['empirical_accuracy']:.3f}" if bin_row["empirical_accuracy"] is not None else "—"
        lines.append(f"| {bin_row['bin']} | {bin_row['count']} | {pred} | {acc} |")

    lines += ["", "## Signal coefficients (logistic, standardized inputs not applied)", ""]
    for name, coef in zip(FEATURES + result.artifact["indicators"], result.artifact["coefficients"]):
        lines.append(f"- `{name}`: {coef:+.3f}")

    lines += ["", "## Human audit / PPI", ""]
    if ppi is not None:
        lines += [
            f"- audit size: {ppi.n_audit} of {ppi.n_total} judged rows",
            f"- judge–human agreement (Cohen's kappa): {kappa:.3f}",
            f"- judge raw correctness rate: {ppi.judge_mean_all:.3f}",
            f"- judge bias (rectifier): {ppi.rectifier:+.3f}",
            f"- **PPI-corrected correctness rate: {ppi.theta:.3f} "
            f"(95% CI [{ppi.ci_low:.3f}, {ppi.ci_high:.3f}])**",
        ]
    else:
        lines += [
            "- fewer than 10 audited rows — run "
            "`scripts/mint_calibration_labels.py --audit N` to anchor the judge "
            "to human ground truth before trusting these numbers.",
        ]

    (out_dir / "calibration_report.md").write_text("\n".join(lines) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"No labels file at {args.labels} — run scripts/mint_calibration_labels.py first")
        sys.exit(1)
    out_dir = args.out_dir or args.labels.parent

    rows = load_rows(args.labels)
    metrics = fit_and_report(rows, out_dir, args.labels)
    print(f"n={metrics['n']} auroc={metrics['auroc']:.3f} "
          f"ece={metrics['ece_calibrated']:.4f} brier={metrics['brier_calibrated']:.4f}")
    print(f"Wrote {out_dir / 'calibrator.json'} and {out_dir / 'calibration_report.md'}")


if __name__ == "__main__":
    main()
