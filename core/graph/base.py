from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    content: str


@dataclass
class DocumentChunkCount:
    document_id: str
    source_url: str | None
    title: str
    total_chunks: int


class GraphStore(ABC):
    @abstractmethod
    def write_document(self, id: str, source_url: str | None, doc_type: str, title: str = "") -> None: ...

    @abstractmethod
    def write_chunk(
        self,
        id: str,
        document_id: str,
        content: str,
        section_heading: str | None,
        page_number: int | None,
    ) -> None: ...

    @abstractmethod
    def write_triple(
        self,
        subject: str,
        subject_type: str,
        relation: str,
        object_: str,
        object_type: str,
        certainty_score: float,
        chunk_id: str,
        raw_certainty_score: float | None = None,
        evidence_text: str | None = None,
        evidence_char_start: int | None = None,
        evidence_char_end: int | None = None,
        evidence_alignment_score: float | None = None,
        entity_anchor_score: float | None = None,
        evidence_scope_score: float | None = None,
        confidence_version: str | None = None,
    ) -> None: ...

    @abstractmethod
    def read_chunks(self) -> list[ChunkRecord]: ...

    @abstractmethod
    def read_document_chunk_counts(self) -> list[DocumentChunkCount]: ...

    @abstractmethod
    def clear_extraction(self) -> None: ...
