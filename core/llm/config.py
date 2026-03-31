from __future__ import annotations

import platform
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_MLX = "mlx-community/Qwen2.5-14B-Instruct-4bit"
DEFAULT_MODEL_OLLAMA = "qwen2.5:14b"
DEFAULT_BASE_URL_OLLAMA = "http://localhost:11434/v1"
DEFAULT_BASE_URL_VLLM = "http://localhost:8000/v1"


@dataclass
class LLMConfig:
    backend: str = "auto"   # "mlx" | "vllm" | "openai-compatible" | "auto"
    model: str = ""
    base_url: str = ""
    api_key: str = "token"
    workers: int = 4


def load_config(start_dir: Path | None = None) -> LLMConfig:
    """Walk up from start_dir looking for alice.toml and parse the [llm] section."""
    import tomllib

    search = (start_dir or Path.cwd()).resolve()
    for directory in [search, *search.parents]:
        candidate = directory / "alice.toml"
        if candidate.exists():
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            llm = data.get("llm", {})
            return LLMConfig(
                backend=llm.get("backend", "auto"),
                model=llm.get("model", ""),
                base_url=llm.get("base_url", ""),
                api_key=llm.get("api_key", "token"),
                workers=llm.get("workers", 4),
            )
    return LLMConfig()


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _first_available_model(base_url: str, api_key: str) -> str:
    import json
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return ""


def auto_detect() -> LLMConfig:
    """Detect the available LLM backend on this machine."""
    if platform.system() == "Darwin":
        return LLMConfig(backend="mlx", model=DEFAULT_MODEL_MLX)

    if _probe(DEFAULT_BASE_URL_OLLAMA + "/models"):
        return LLMConfig(backend="openai-compatible", base_url=DEFAULT_BASE_URL_OLLAMA)

    if _probe(DEFAULT_BASE_URL_VLLM + "/models"):
        return LLMConfig(backend="vllm", base_url=DEFAULT_BASE_URL_VLLM)

    raise RuntimeError(
        "No LLM backend detected. Please do one of the following:\n"
        "  • Ollama:  ollama pull qwen2.5:14b && ollama serve\n"
        "  • vLLM:    vllm serve <model>\n"
        "  • Config:  create alice.toml with an [llm] section (see README)\n"
    )


def resolve_config(
    *,
    cli_model: str | None,
    cli_backend: str | None,
    cli_base_url: str | None,
    cli_api_key: str | None,
    cli_workers: int | None,
    start_dir: Path | None = None,
) -> LLMConfig:
    """Resolve final LLMConfig using: CLI flags > alice.toml > auto-detect."""
    cfg = load_config(start_dir)

    if cfg.backend == "auto":
        detected = auto_detect()
        if not cfg.model:
            cfg.model = detected.model
        if not cfg.base_url:
            cfg.base_url = detected.base_url
        cfg.backend = detected.backend

    if cli_backend is not None:
        cfg.backend = cli_backend
    if cli_model is not None:
        cfg.model = cli_model
    if cli_base_url is not None:
        cfg.base_url = cli_base_url
    if cli_api_key is not None:
        cfg.api_key = cli_api_key
    if cli_workers is not None:
        cfg.workers = cli_workers

    if not cfg.model:
        if cfg.backend == "mlx":
            cfg.model = DEFAULT_MODEL_MLX
        else:
            cfg.model = _first_available_model(cfg.base_url, cfg.api_key)

    return cfg
