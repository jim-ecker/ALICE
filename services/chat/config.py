from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.embeddings.client import EmbeddingsConfig
from core.scoring.composite import ScoringConfig


@dataclass
class ChatConfig:
    db_path: Path = Path("data/chat/chat.db")
    embeddings_path: Path = Path("chat.embeddings.npz")
    experts_dir: Path = Path("data/experts")
    host: str = "127.0.0.1"
    port: int = 8766
    history_turns: int = 10
    top_k_chunks: int = 8
    max_tokens: int = 2048
    max_context_chunks: int = 20
    entity_hop_depth: int = 1
    max_hop2_entities: int = 20
    ingest_folder: Path | None = None


def load_chat_config(
    start_dir: Path | None = None,
) -> tuple[ChatConfig, EmbeddingsConfig, ScoringConfig, "LLMConfig | None"]:
    """Walk up from start_dir looking for alice.toml; read [chat], [embeddings], [scoring], [chat_llm]."""
    import tomllib
    from core.llm.config import LLMConfig

    search = (start_dir or Path.cwd()).resolve()
    data: dict = {}
    for directory in [search, *search.parents]:
        candidate = directory / "alice.toml"
        if candidate.exists():
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            break

    chat_raw = data.get("chat", {})
    embed_raw = data.get("embeddings", {})
    score_raw = data.get("scoring", {})
    chat_llm_raw = data.get("chat_llm", None)

    chat_cfg = ChatConfig(
        db_path=Path(chat_raw.get("db_path", "data/chat/chat.db")),
        embeddings_path=Path(chat_raw.get("embeddings_path", "chat.embeddings.npz")),
        experts_dir=Path(chat_raw.get("experts_dir", "data/experts")),
        host=chat_raw.get("host", "127.0.0.1"),
        port=int(chat_raw.get("port", 8766)),
        history_turns=int(chat_raw.get("history_turns", 10)),
        top_k_chunks=int(chat_raw.get("top_k_chunks", 8)),
        max_tokens=int(chat_raw.get("max_tokens", 2048)),
        max_context_chunks=int(chat_raw.get("max_context_chunks", 20)),
        entity_hop_depth=int(chat_raw.get("entity_hop_depth", 1)),
        max_hop2_entities=int(chat_raw.get("max_hop2_entities", 20)),
        ingest_folder=Path(chat_raw["ingest_folder"]) if "ingest_folder" in chat_raw else None,
    )

    embed_cfg = EmbeddingsConfig(
        base_url=embed_raw.get("base_url", "http://localhost:11434/v1"),
        api_key=embed_raw.get("api_key", "token"),
        model=embed_raw.get("model", "nomic-embed-text"),
        batch_size=int(embed_raw.get("batch_size", 64)),
        max_chars=int(embed_raw.get("max_chars", 2200)),
    )

    filter_top_k = score_raw.get("relevance_filter_top_k", None)
    scoring_cfg = ScoringConfig(
        ingest_certainty_weight=float(score_raw.get("ingest_certainty_weight", 0.4)),
        relevance_weight=float(score_raw.get("relevance_weight", 0.4)),
        provenance_weight=float(score_raw.get("provenance_weight", 0.2)),
        grounding_weight=float(score_raw.get("grounding_weight", 0.0)),
        grounding_enabled=bool(score_raw.get("grounding_enabled", False)),
        relevance_filter_top_k=int(filter_top_k) if filter_top_k is not None else None,
        ingest_certainty_cap=float(score_raw.get("ingest_certainty_cap", 1.0)),
        ingest_certainty_exponent=float(score_raw.get("ingest_certainty_exponent", 1.0)),
    )

    chat_llm_cfg = None
    if chat_llm_raw is not None:
        chat_llm_cfg = LLMConfig(
            backend=chat_llm_raw.get("backend", "auto"),
            model=chat_llm_raw.get("model", ""),
            base_url=chat_llm_raw.get("base_url", ""),
            api_key=chat_llm_raw.get("api_key", "token"),
            workers=chat_llm_raw.get("workers", 1),
        )

    return chat_cfg, embed_cfg, scoring_cfg, chat_llm_cfg
