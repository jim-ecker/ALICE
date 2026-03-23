from .base import LLMBackend


class VLLMBackend(LLMBackend):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1", api_key: str = "token"):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model_name

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
