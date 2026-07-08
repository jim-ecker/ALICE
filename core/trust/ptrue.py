"""P(True) verification signal.

A second LLM pass asks whether the evidence sentence states or implies the
triple, answered Yes/No. Preferred estimator: normalized token-logprob mass
P(Yes)/(P(Yes)+P(No)) at the first Yes/No token position. When the backend
exposes no logprobs, we sample the answer m times at temperature > 0 and
report the Beta(1,1) posterior mean (v+1)/(m+2) over the parseable votes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.llm.base import LLMBackend
from core.trust.config import TrustConfig

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.I)

_SYSTEM = (
    "You are verifying a knowledge-graph fact against source text. "
    "Answer with exactly one word: Yes or No."
)


@dataclass(frozen=True)
class PTrueResult:
    score: float
    method: str   # "logprob" | "vote"
    samples: int  # 0 for the logprob path, parseable votes for the vote path


def verbalize_triple(subject: str, relation: str, object_: str) -> str:
    return f"{subject} {relation} {object_}."


def build_ptrue_messages(
    evidence_sentence: str, subject: str, relation: str, object_: str
) -> list[dict[str, str]]:
    claim = verbalize_triple(subject, relation, object_)
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Text:\n{evidence_sentence}\n\n"
                f"Does the text state or imply the following?\n{claim}"
            ),
        },
    ]


def parse_yes_no(text: str) -> bool | None:
    """First standalone yes/no in the completion; None if neither appears.
    Tolerates reasoning-model preambles before the answer."""
    match = _YES_NO_RE.search(text or "")
    if match is None:
        return None
    return match.group(1).lower() == "yes"


def vote_posterior_mean(votes_yes: int, m: int) -> float:
    """Beta(1,1) posterior mean over m Bernoulli votes."""
    return (votes_yes + 1) / (m + 2)


def compute_ptrue(
    backend: LLMBackend,
    *,
    evidence_sentence: str,
    subject: str,
    relation: str,
    object_: str,
    cfg: TrustConfig,
) -> PTrueResult | None:
    messages = build_ptrue_messages(evidence_sentence, subject, relation, object_)

    logprob_score = backend.score_yes_no(messages, max_tokens=4)
    if logprob_score is not None:
        return PTrueResult(score=logprob_score, method="logprob", samples=0)

    samples = backend.sample_n(
        messages,
        n=cfg.ptrue_fallback_samples,
        temperature=cfg.ptrue_temperature,
        max_tokens=cfg.ptrue_vote_max_tokens,
    )
    votes = [v for v in (parse_yes_no(s) for s in samples) if v is not None]
    if not votes:
        return None
    return PTrueResult(
        score=vote_posterior_mean(sum(votes), len(votes)),
        method="vote",
        samples=len(votes),
    )
