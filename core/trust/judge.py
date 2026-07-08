"""LLM judge for minting calibration labels.

Given a chunk and an extracted triple, the judge decides whether the chunk
states or directly implies the triple in the correct context. The judge should
be a different model than the extractor (self-grading inflates agreement);
its residual bias is corrected by the human audit via PPI (core/trust/ppi.py).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from core.trust.config import TrustConfig
from core.trust.ptrue import verbalize_triple

VERDICT_SUPPORTED = "supported"
VERDICT_NOT_SUPPORTED = "not_supported"
VERDICT_UNCLEAR = "unclear"

_VERDICT_RE = re.compile(r"\b(supported|notsupported|not_supported|not supported|unclear)\b", re.I)

_SYSTEM = (
    "You are auditing facts extracted from technical documents. Given a passage "
    "and a candidate fact, decide whether the passage states or directly implies "
    "the fact in its intended context. Pay attention to negation, hedging, wrong "
    "direction, and misattribution: a fact that reverses or contradicts what the "
    "passage says is NotSupported. Answer with exactly one word: "
    "Supported, NotSupported, or Unclear."
)


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str        # supported | not_supported | unclear
    judge_model: str
    raw_response: str


def build_judge_messages(
    chunk_text: str, subject: str, relation: str, object_: str, evidence_text: str | None = None
) -> list[dict[str, str]]:
    claim = verbalize_triple(subject, relation, object_)
    evidence_part = f"\n\nExtractor's cited evidence:\n{evidence_text}" if evidence_text else ""
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Passage:\n{chunk_text}{evidence_part}\n\n"
                f"Candidate fact:\n{claim}\n\n"
                "Is the fact supported by the passage, in the correct context? "
                "Answer with exactly one word: Supported, NotSupported, or Unclear."
            ),
        },
    ]


def parse_verdict(text: str) -> str | None:
    match = _VERDICT_RE.search(text or "")
    if match is None:
        return None
    word = match.group(1).lower().replace(" ", "_").replace("notsupported", "not_supported")
    return word


class Judge:
    def __init__(self, cfg: TrustConfig, *, model: str | None = None, backend: str | None = None,
                 base_url: str | None = None) -> None:
        self._backend_kind = backend or cfg.judge_backend
        self._model = model or cfg.judge_model
        self._base_url = base_url or cfg.judge_base_url
        self._api_key = os.environ.get(cfg.judge_api_key_env, "") if cfg.judge_api_key_env else ""
        if not self._model:
            raise ValueError(
                "No judge model configured — set [trust] judge_model in alice.toml or pass --judge-model"
            )
        self._client = None

    def _chat(self, messages: list[dict[str, str]]) -> str:
        if self._backend_kind == "anthropic":
            if self._client is None:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key or None)
            system = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_messages = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(
                model=self._model, max_tokens=512, system=system, messages=user_messages,
            )
            return "".join(block.text for block in resp.content if block.type == "text")

        if self._client is None:
            from core.llm.vllm import VLLMBackend
            self._client = VLLMBackend(self._model, base_url=self._base_url,
                                       api_key=self._api_key or "token")
        # Reasoning models need headroom before the one-word verdict reaches content
        return self._client.chat(messages, max_tokens=512)

    def judge_triple(
        self, chunk_text: str, subject: str, relation: str, object_: str,
        evidence_text: str | None = None,
    ) -> JudgeVerdict:
        messages = build_judge_messages(chunk_text, subject, relation, object_, evidence_text)
        response = self._chat(messages) or ""
        verdict = parse_verdict(response) or VERDICT_UNCLEAR
        return JudgeVerdict(verdict=verdict, judge_model=self._model, raw_response=response.strip())
