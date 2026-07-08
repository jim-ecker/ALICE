from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str: ...

    def score_yes_no(self, messages: list[dict[str, str]], *, max_tokens: int = 4) -> float | None:
        """P(answer is 'Yes') from token logprobs. None = backend can't provide
        logprobs; the caller falls back to sampled voting."""
        return None

    def sample_n(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> list[str]:
        """n completions sampled at the given temperature. Backends without
        temperature control return n plain chat() calls, which may be
        near-identical on deterministic servers."""
        return [self.chat(messages, max_tokens=max_tokens) for _ in range(n)]
