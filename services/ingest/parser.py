import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from .models import SourceDocument


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, doc: SourceDocument) -> DoclingDocument: ...


class DoclingParser(DocumentParser):
    def __init__(self):
        self._converter = DocumentConverter()

    def parse(self, doc: SourceDocument) -> DoclingDocument:
        result = self._converter.convert(str(doc.path))
        return result.document


def make_source_document(path: str | Path, source_url: str | None = None) -> SourceDocument:
    path = Path(path)
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
    return SourceDocument(
        id=doc_id,
        path=path,
        doc_type=path.suffix.lstrip(".").lower(),
        source_url=source_url,
    )
