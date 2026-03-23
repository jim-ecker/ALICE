from __future__ import annotations

import re

from core.graph.retrieval import CitationChunk, RetrievedTriple
from core.llm.base import LLMBackend

_PROMPT = """\
You are a fact-verification assistant. Given a source text and a claim, output a \
single decimal number between 0.0 and 1.0 that represents how well the source text \
supports the claim.

Scoring guide:
  1.0 — the text explicitly and directly states this fact
  0.8 — the text strongly implies this fact
  0.5 — the text is loosely consistent with the fact but does not clearly support it
  0.2 — the text is unrelated or only tangentially related
  0.0 — the text contradicts the fact

Output ONLY the number. No explanation."""

_FLOAT_RE = re.compile(r"\b(1\.0+|0\.\d+)\b")


class GroundingScorer:
    """LLM-based entailment scorer: does the source chunk text actually support a triple?

    This is the most expensive signal — one LLM call per triple — so it is opt-in
    via `grounding_enabled = true` in [scoring] config.
    """

    def __init__(self, llm: LLMBackend, max_tokens: int = 16) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def score_one(self, triple: RetrievedTriple, source_chunk: CitationChunk) -> float:
        """Return a grounding score in [0, 1]."""
        claim = f"{triple.subject} {triple.relation} {triple.object_}"
        text = source_chunk.content[:1500]  # truncate to avoid context overflow

        messages = [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": f'Source text:\n"""\n{text}\n"""\n\nClaim: "{claim}"\n\nGrounding score:',
            },
        ]
        raw = self._llm.chat(messages, max_tokens=self._max_tokens).strip()
        m = _FLOAT_RE.search(raw)
        if m:
            return float(m.group(1))
        # Fallback: try parsing the whole response as a float
        try:
            return max(0.0, min(1.0, float(raw.split()[0])))
        except (ValueError, IndexError):
            return 0.5  # uncertain default

    def score_batch(
        self,
        triples: list[RetrievedTriple],
        chunk_map: dict[str, CitationChunk],
    ) -> list[float]:
        """Score each triple against its source chunk."""
        scores = []
        for triple in triples:
            chunk = chunk_map.get(triple.chunk_id)
            if chunk is None:
                scores.append(0.5)
            else:
                scores.append(self.score_one(triple, chunk))
        return scores
