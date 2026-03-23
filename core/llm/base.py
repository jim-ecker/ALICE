from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str: ...
