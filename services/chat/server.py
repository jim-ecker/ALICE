from __future__ import annotations


def start(
    db_path,
    *,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None,
    backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Wire up all dependencies and start the uvicorn server."""
    from pathlib import Path
    from services.chat.service import Chat
    from core.llm.config import resolve_config

    # Only resolve if the user passed CLI overrides; otherwise Chat picks up
    # [chat_llm] from alice.toml (falling back to [llm] if absent).
    llm_cfg = None
    if any(v is not None for v in (model, backend, base_url, api_key)):
        llm_cfg = resolve_config(
            cli_model=model,
            cli_backend=backend,
            cli_base_url=base_url,
            cli_api_key=api_key,
            cli_workers=None,
        )
    Chat(db_path=Path(db_path), llm_cfg=llm_cfg, host=host, port=port).serve()
