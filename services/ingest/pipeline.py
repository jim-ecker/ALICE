from .chunker import DoclingChunker
from .models import Chunk, SourceDocument
from .parser import DocumentParser


class IngestionPipeline:
    def __init__(self, parser: DocumentParser, chunker: DoclingChunker):
        self._parser = parser
        self._chunker = chunker

    def run(self, doc: SourceDocument) -> list[Chunk]:
        parsed = self._parser.parse(doc)
        return self._chunker.chunk(parsed, doc)
