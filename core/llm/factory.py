from .config import LLMConfig
from .base import LLMBackend


def create_backend(cfg: LLMConfig) -> LLMBackend:
    if cfg.backend == "mlx":
        from .mlx import MLXBackend
        return MLXBackend(cfg.model)
    elif cfg.backend in {"vllm", "openai-compatible"}:
        from .vllm import VLLMBackend
        return VLLMBackend(cfg.model, base_url=cfg.base_url, api_key=cfg.api_key)
    else:
        raise ValueError(
            f"Unknown backend '{cfg.backend}'. Expected 'mlx', 'vllm', or 'openai-compatible'."
        )
