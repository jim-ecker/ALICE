from __future__ import annotations

from pathlib import Path


def default_chat_data_dir() -> Path:
    return Path("services/chat/data")


def default_chat_db_path() -> Path:
    return default_chat_data_dir() / "chat.db"


def resolve_chat_db_path(db_path: Path | str) -> Path:
    path = Path(db_path)
    if path.is_absolute():
        return path
    return default_chat_data_dir() / path
