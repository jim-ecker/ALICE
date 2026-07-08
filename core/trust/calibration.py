"""Calibrated trust combiner (Phase 2).

Fits a logistic regression over the per-triple signal vector, then isotonic
regression on out-of-fold predictions, so the output is a calibrated
P(triple correct): among triples scored p, a fraction ~p are correct on
held-out data. Fit on labels drawn from the SAME corpus/pipeline the score
will be deployed on — calibration does not survive domain shift.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CALIBRATOR_VERSION = "v4_calibrated"

# Signal features per triple. v2 features exist on every extracted row; the v3
# signals may be NULL (trust disabled, extra missing, not backfilled) and are
# mean-imputed with a per-group missingness indicator.
FEATURES = [
    "ptrue_score",
    "nli_entailment_score",
    "nli_contradiction_score",
    "self_consistency_freq",
    "evidence_alignment_score",
    "entity_anchor_score",
    "evidence_scope_score",
    "raw_certainty_score",
]
INDICATOR_GROUPS = {
    "ptrue_missing": ["ptrue_score"],
    "nli_missing": ["nli_entailment_score", "nli_contradiction_score"],
    "consistency_missing": ["self_consistency_freq"],
}
FEATURE_COLUMNS = FEATURES + list(INDICATOR_GROUPS)


def build_feature_matrix(
    rows: list[dict], means: dict[str, float] | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    """(n, len(FEATURE_COLUMNS)) matrix. Training call: means=None (computed and
    returned for the artifact). Inference call: pass the stored means."""
    if means is None:
        means = {}
        for feat in FEATURES:
            values = [r[feat] for r in rows if r.get(feat) is not None]
            means[feat] = float(np.mean(values)) if values else 0.5

    X = np.zeros((len(rows), len(FEATURE_COLUMNS)))
    for i, row in enumerate(rows):
        for j, feat in enumerate(FEATURES):
            value = row.get(feat)
            X[i, j] = means[feat] if value is None else float(value)
        for j, (indicator, group) in enumerate(INDICATOR_GROUPS.items(), start=len(FEATURES)):
            X[i, j] = 1.0 if all(row.get(f) is None for f in group) else 0.0
    return X, means


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        table.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f}{']' if hi == 1.0 else ')'}",
                "count": int(mask.sum()),
                "mean_predicted": float(probs[mask].mean()) if mask.any() else None,
                "empirical_accuracy": float(labels[mask].mean()) if mask.any() else None,
            }
        )
    return table


@dataclass
class FitResult:
    artifact: dict
    metrics: dict


def fit_calibrator(
    rows: list[dict],
    labels: list[int],
    *,
    n_folds: int = 5,
    seed: int = 42,
    metadata: dict | None = None,
) -> FitResult:
    """K-fold out-of-fold logistic scores -> isotonic calibration map.
    Metrics are computed on the OOF predictions (honest, not resubstitution)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(labels, dtype=float)
    if len(rows) != len(y):
        raise ValueError("rows and labels length mismatch")
    if len(np.unique(y)) < 2:
        raise ValueError("labels contain a single class — cannot fit a calibrator")
    X, means = build_feature_matrix(rows)

    n_folds = min(n_folds, int(min((y == 0).sum(), (y == 1).sum())))
    if n_folds < 2:
        raise ValueError("need at least 2 examples of each class")

    def _make_lr() -> LogisticRegression:
        return LogisticRegression(max_iter=1000, C=1.0)

    oof = np.zeros(len(y))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        model = _make_lr().fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    # Isotonic evaluated in-sample is ~0 ECE by construction, so metrics come
    # from a stratified holdout: fit the map on one half of the OOF scores,
    # measure calibration on the other. The deployed artifact refits on all.
    from sklearn.model_selection import train_test_split

    fit_idx, eval_idx = train_test_split(
        np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed
    )
    iso_holdout = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        oof[fit_idx], y[fit_idx]
    )
    eval_calibrated = iso_holdout.predict(oof[eval_idx])
    eval_raw, eval_y = oof[eval_idx], y[eval_idx]

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
    final = _make_lr().fit(X, y)

    from sklearn.metrics import brier_score_loss, roc_auc_score

    metrics = {
        "n": int(len(y)),
        "n_folds": n_folds,
        "n_eval_holdout": int(len(eval_idx)),
        "base_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, oof)),
        "ece_uncalibrated": expected_calibration_error(eval_raw, eval_y),
        "ece_calibrated": expected_calibration_error(eval_calibrated, eval_y),
        "brier_uncalibrated": float(brier_score_loss(eval_y, eval_raw)),
        "brier_calibrated": float(brier_score_loss(eval_y, eval_calibrated)),
        "reliability": reliability_table(eval_calibrated, eval_y),
    }

    artifact = {
        "version": CALIBRATOR_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "indicators": list(INDICATOR_GROUPS),
        "imputation_means": means,
        "coefficients": final.coef_[0].tolist(),
        "intercept": float(final.intercept_[0]),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "metrics": {k: v for k, v in metrics.items() if k != "reliability"},
        "metadata": metadata or {},
    }
    return FitResult(artifact=artifact, metrics=metrics)


def apply_calibrator(artifact: dict, row: dict) -> float:
    """Calibrated P(correct) for one triple's signal dict (keys as in FEATURES;
    missing/None values handled the same way as at fit time)."""
    X, _ = build_feature_matrix([row], means=artifact["imputation_means"])
    z = float(np.dot(X[0], np.asarray(artifact["coefficients"])) + artifact["intercept"])
    p = 1.0 / (1.0 + math.exp(-z))
    return float(np.interp(p, artifact["isotonic_x"], artifact["isotonic_y"]))


def save_artifact(artifact: dict, path: Path) -> None:
    path.write_text(json.dumps(artifact, indent=2))


def load_artifact(path: Path) -> dict:
    return json.loads(path.read_text())
