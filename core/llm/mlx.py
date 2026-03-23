import warnings

warnings.filterwarnings("ignore", message="mx.metal.device_info is deprecated")

from .base import LLMBackend


class MLXBackend(LLMBackend):
    def __init__(self, model_name: str):
        from mlx_lm import load
        self._model, self._tokenizer = load(model_name)

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
        from mlx_lm import generate
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(self._model, self._tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
