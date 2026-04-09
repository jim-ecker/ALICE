import hashlib

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument

from .models import Chunk, ChunkProvenance, SourceDocument

_MIN_CHUNK_CHARS = 150


def _merge_short_chunks(chunks: list[Chunk], source_id: str, min_chars: int = _MIN_CHUNK_CHARS) -> list[Chunk]:
    """Merge adjacent chunks that are too short to yield meaningful triples.

    PDF parsers frequently split author bylines, affiliations, and footnotes into
    separate micro-chunks (e.g. "James E. Ecker ∗" / "NASA Langley Research Center").
    Each is extracted in isolation, so the LLM can't form triples connecting them.
    Merging these into one chunk gives the extractor the full context it needs.

    Merge rules:
    - Forward-merge: if a pending chunk is shorter than min_chars, absorb the next chunk.
    - Only merge across chunks on the same page with compatible section headings
      (same heading, or at least one side has no heading).
    - The merged chunk inherits the first chunk's page and heading; content is joined with newline.
    - A chunk that is still short at the end of the list is kept as-is (no content is dropped).
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    pending: Chunk | None = None

    for chunk in chunks:
        if pending is None:
            pending = chunk
            continue

        same_page = pending.provenance.page_number == chunk.provenance.page_number
        headings_compatible = (
            pending.provenance.section_heading is None
            or chunk.provenance.section_heading is None
            or pending.provenance.section_heading == chunk.provenance.section_heading
        )

        candidate_content = pending.content + "\n" + chunk.content
        if len(pending.content) < min_chars and len(chunk.content) < min_chars and same_page and headings_compatible:
            merged_content = candidate_content
            merged_id = hashlib.sha256(f"{source_id}:{merged_content}".encode()).hexdigest()
            pending = Chunk(
                id=merged_id,
                content=merged_content,
                provenance=ChunkProvenance(
                    document_id=pending.provenance.document_id,
                    source_url=pending.provenance.source_url,
                    section_heading=pending.provenance.section_heading or chunk.provenance.section_heading,
                    page_number=pending.provenance.page_number,
                ),
            )
        else:
            merged.append(pending)
            pending = chunk

    if pending is not None:
        merged.append(pending)

    return merged


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
        return _merge_short_chunks(chunks, source.id)
