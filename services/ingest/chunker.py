import hashlib

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument

from .models import Chunk, ChunkProvenance, SourceDocument


class DoclingChunker:
    def __init__(self):
        self._chunker = HierarchicalChunker()

    def chunk(self, document: DoclingDocument, source: SourceDocument) -> list[Chunk]:
        chunks = []
        for base_chunk in self._chunker.chunk(document):
            meta = base_chunk.meta
            heading = meta.headings[0] if meta.headings else None
            page_number = (
                meta.doc_items[0].prov[0].page_no
                if meta.doc_items and meta.doc_items[0].prov
                else None
            )
            chunk_id = hashlib.sha256(f"{source.id}:{base_chunk.text}".encode()).hexdigest()
            chunks.append(Chunk(
                id=chunk_id,
                content=base_chunk.text,
                provenance=ChunkProvenance(
                    document_id=source.id,
                    source_url=source.source_url,
                    section_heading=heading,
                    page_number=page_number,
                ),
            ))
        return chunks
