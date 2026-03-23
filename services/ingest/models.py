from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceDocument:
    id: str
    path: Path
    doc_type: str
    source_url: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkProvenance:
    document_id: str
    source_url: str | None
    section_heading: str | None
    page_number: int | None


@dataclass
class Chunk:
    id: str
    content: str
    provenance: ChunkProvenance
