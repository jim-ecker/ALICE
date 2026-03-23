"""Module-level worker functions for ProcessPoolExecutor-based parallel extraction.

Functions must be at module level to be picklable by ProcessPoolExecutor.
Each worker process loads its own model instance via init_worker().
"""

_backend = None  # one instance per worker process


def init_worker(cfg) -> None:   # cfg: LLMConfig
    global _backend
    from core.llm.factory import create_backend
    _backend = create_backend(cfg)


def extract_chunk(chunk_id: str, document_id: str, content: str) -> tuple[str, list]:
    from core.llm.generate_triple import generate_triples
    triples = generate_triples(chunk_id, content, _backend)
    return (document_id, triples)
