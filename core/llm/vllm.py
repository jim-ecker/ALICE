import math
import time

from .base import LLMBackend

_YES_NO = {"yes", "no"}


class VLLMBackend(LLMBackend):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1", api_key: str = "token"):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model_name
        self._logprobs_supported: bool | None = None  # probed on first score_yes_no call

    def _create(self, **kwargs):
        from openai import RateLimitError
        for attempt in range(8):
            try:
                return self._client.chat.completions.create(model=self._model, **kwargs)
            except RateLimitError as e:
                if "insufficient_quota" in str(e) or attempt == 7:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
        response = self._create(messages=messages, max_tokens=max_tokens)
        return response.choices[0].message.content

    def score_yes_no(self, messages: list[dict[str, str]], *, max_tokens: int = 4) -> float | None:
        if self._logprobs_supported is False:
            return None
        try:
            response = self._create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                logprobs=True,
                top_logprobs=10,
            )
        except Exception:
            # Some OpenAI-compatible servers reject unknown params outright
            self._logprobs_supported = False
            return None

        logprobs = response.choices[0].logprobs
        content = getattr(logprobs, "content", None) if logprobs else None
        if not content:
            # Server accepted the request but returned no logprobs (Ollama does this)
            self._logprobs_supported = False
            return None
        self._logprobs_supported = True

        # Reasoning models may emit preamble tokens; score the first yes/no position
        for token_info in content:
            if token_info.token.strip().lower() not in _YES_NO:
                continue
            yes_mass = 0.0
            no_mass = 0.0
            alternatives = token_info.top_logprobs or [token_info]
            for alt in alternatives:
                word = alt.token.strip().lower()
                if word == "yes":
                    yes_mass += math.exp(alt.logprob)
                elif word == "no":
                    no_mass += math.exp(alt.logprob)
            total = yes_mass + no_mass
            if total > 0:
                return yes_mass / total
            return None
        return None  # no yes/no token in the response; caller falls back to voting

    def sample_n(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> list[str]:
        # n separate requests — the `n=` param is silently ignored by Ollama
        samples = []
        for _ in range(n):
            response = self._create(messages=messages, max_tokens=max_tokens, temperature=temperature)
            samples.append(response.choices[0].message.content)
        return samples
