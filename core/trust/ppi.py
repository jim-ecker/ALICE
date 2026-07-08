"""Prediction-powered inference (Angelopoulos et al., 2023) for judge labels.

The LLM judge labels every sampled triple cheaply; a human audits a small
random subsample. PPI estimates the true correctness rate using all judge
labels while correcting the judge's bias with the audit, yielding valid
confidence intervals anchored to human ground truth — not to the judge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PPIEstimate:
    theta: float          # bias-corrected estimate of the true mean
    ci_low: float
    ci_high: float
    judge_mean_all: float
    rectifier: float      # mean(judge - human) on the audit set
    n_total: int
    n_audit: int


def ppi_mean(
    judge_all: list[float],
    judge_audit: list[float],
    human_audit: list[float],
    *,
    alpha: float = 0.05,
) -> PPIEstimate:
    """PPI estimate of mean(human label) over the full sample.

    judge_all: judge labels (0/1) on all N sampled items.
    judge_audit / human_audit: paired judge and human labels on the n audited
    items (the audit must be a uniform random subsample of the full sample).
    """
    if len(judge_audit) != len(human_audit):
        raise ValueError("judge_audit and human_audit must be paired")
    if not judge_all or not judge_audit:
        raise ValueError("empty inputs")

    n_total = len(judge_all)
    n_audit = len(judge_audit)

    judge_mean = sum(judge_all) / n_total
    diffs = [j - h for j, h in zip(judge_audit, human_audit)]
    rectifier = sum(diffs) / n_audit
    theta = judge_mean - rectifier

    var_judge = _sample_var(judge_all) / n_total
    var_diff = _sample_var(diffs) / n_audit
    se = math.sqrt(var_judge + var_diff)
    z = _z_quantile(1 - alpha / 2)

    return PPIEstimate(
        theta=theta,
        ci_low=max(0.0, theta - z * se),
        ci_high=min(1.0, theta + z * se),
        judge_mean_all=judge_mean,
        rectifier=rectifier,
        n_total=n_total,
        n_audit=n_audit,
    )


def cohens_kappa(a: list, b: list) -> float:
    """Chance-corrected agreement between two label sequences."""
    if len(a) != len(b) or not a:
        raise ValueError("sequences must be non-empty and paired")
    n = len(a)
    categories = set(a) | set(b)
    p_observed = sum(1 for x, y in zip(a, b) if x == y) / n
    p_expected = sum(
        (sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n)
        for c in categories
    )
    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1 - p_expected)


def _sample_var(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _z_quantile(p: float) -> float:
    """Standard normal quantile via Acklam's rational approximation (|err| < 1e-9)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    # Coefficients for the central region
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - p_low:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
