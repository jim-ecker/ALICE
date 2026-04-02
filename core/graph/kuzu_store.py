from pathlib import Path

import kuzu

from .base import ChunkRecord, DocumentChunkCount, GraphStore

_SCHEMA = [
    """
    CREATE NODE TABLE IF NOT EXISTS IngestState(
        query_key STRING,
        next_offset INT64,
        PRIMARY KEY (query_key)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Document(
        id STRING,
        source_url STRING,
        doc_type STRING,
        title STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Chunk(
        id STRING,
        document_id STRING,
        content STRING,
        section_heading STRING,
        page_number INT64,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Entity(
        name STRING,
        type STRING,
        PRIMARY KEY (name)
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS CONTAINS(
        FROM Document TO Chunk
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS RELATES_TO(
        FROM Entity TO Entity,
        relation STRING,
        certainty_score DOUBLE,
        chunk_id STRING,
        raw_certainty_score DOUBLE,
        evidence_text STRING,
        evidence_char_start INT64,
        evidence_char_end INT64,
        evidence_alignment_score DOUBLE,
        entity_anchor_score DOUBLE,
        evidence_scope_score DOUBLE,
        confidence_version STRING
    )
    """,
]


_BUFFER_POOL_SIZE = 2 * 1024 ** 3  # 2 GB — fixed cap so Kuzu doesn't claim 80% of available RAM


class KuzuStore(GraphStore):
    def __init__(self, db_path: "str | Path | kuzu.Database"):
        if isinstance(db_path, kuzu.Database):
            self._db = db_path
            self._owns_db = False
        else:
            self._db = kuzu.Database(str(db_path), buffer_pool_size=_BUFFER_POOL_SIZE)
            self._owns_db = True
        self._conn = kuzu.Connection(self._db)
        try:
            self._init_schema()
        except RuntimeError as exc:
            if "Cannot read from file" in str(exc):
                raise RuntimeError(
                    "Database is corrupted (likely from a previous crash). "
                    "Run `alice chat reset` to delete it and start fresh."
                ) from exc
            raise

    def close(self) -> None:
        self._conn.close()
        if self._owns_db:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _init_schema(self) -> None:
        for statement in _SCHEMA:
            self._conn.execute(statement)
        # Migrate existing databases that predate the title column
        try:
            self._conn.execute("ALTER TABLE Document ADD title STRING DEFAULT ''")
        except Exception:
            pass
        # Migrate existing databases that predate the extracted_at column
        try:
            self._conn.execute("ALTER TABLE Chunk ADD extracted_at STRING")
        except Exception:
            pass
        rel_migrations = [
            "ALTER TABLE RELATES_TO ADD raw_certainty_score DOUBLE",
            "ALTER TABLE RELATES_TO ADD evidence_text STRING",
            "ALTER TABLE RELATES_TO ADD evidence_char_start INT64",
            "ALTER TABLE RELATES_TO ADD evidence_char_end INT64",
            "ALTER TABLE RELATES_TO ADD evidence_alignment_score DOUBLE",
            "ALTER TABLE RELATES_TO ADD entity_anchor_score DOUBLE",
            "ALTER TABLE RELATES_TO ADD evidence_scope_score DOUBLE",
            "ALTER TABLE RELATES_TO ADD confidence_version STRING",
        ]
        for statement in rel_migrations:
            try:
                self._conn.execute(statement)
            except Exception:
                pass
        # Backfill extracted_at for chunks that already have triples so they aren't re-processed
        self._conn.execute(
            """
            MATCH ()-[rel:RELATES_TO]->()
            WITH DISTINCT rel.chunk_id AS cid
            MATCH (c:Chunk {id: cid})
            WHERE c.extracted_at IS NULL
            SET c.extracted_at = '2000-01-01T00:00:00+00:00'
            """
        )

    def document_exists(self, id: str) -> bool:
        """Return True if a Document node with this ID is already in the graph."""
        r = self._conn.execute(
            "MATCH (d:Document {id: $id}) RETURN count(d)",
            parameters={"id": id},
        )
        return r.get_next()[0] > 0

    def document_exists_by_url(self, source_url: str) -> bool:
        """Return True if a Document with this source_url is already in the graph."""
        r = self._conn.execute(
            "MATCH (d:Document {source_url: $url}) RETURN count(d)",
            parameters={"url": source_url},
        )
        return r.get_next()[0] > 0

    def get_ingest_offset(self, query_key: str) -> int:
        """Return the stored next-offset for this query key (0 if never run before)."""
        r = self._conn.execute(
            "MATCH (s:IngestState {query_key: $k}) RETURN s.next_offset",
            parameters={"k": query_key},
        )
        return r.get_next()[0] if r.has_next() else 0

    def set_ingest_offset(self, query_key: str, next_offset: int) -> None:
        """Persist the next-offset for this query key."""
        self._conn.execute(
            "MERGE (s:IngestState {query_key: $k}) SET s.next_offset = $off",
            parameters={"k": query_key, "off": next_offset},
        )

    def write_document(self, id: str, source_url: str | None, doc_type: str, title: str = "") -> None:
        self._write_document(id, source_url, doc_type, title)

    def _write_document(self, id: str, source_url: str | None, doc_type: str, title: str = "") -> None:
        self._conn.execute(
            "MERGE (d:Document {id: $id}) SET d.source_url = $source_url, d.doc_type = $doc_type, d.title = $title",
            parameters={"id": id, "source_url": source_url, "doc_type": doc_type, "title": title},
        )

    def write_chunk(
        self,
        id: str,
        document_id: str,
        content: str,
        section_heading: str | None,
        page_number: int | None,
    ) -> None:
        self._write_chunk(id, document_id, content, section_heading, page_number)

    def _write_chunk(
        self,
        id: str,
        document_id: str,
        content: str,
        section_heading: str | None,
        page_number: int | None,
    ) -> None:
        self._conn.execute(
            """
            MERGE (c:Chunk {id: $id})
            SET c.document_id = $document_id,
                c.content = $content,
                c.section_heading = $section_heading,
                c.page_number = $page_number
            """,
            parameters={
                "id": id,
                "document_id": document_id,
                "content": content,
                "section_heading": section_heading,
                "page_number": page_number,
            },
        )
        self._conn.execute(
            """
            MATCH (d:Document {id: $document_id}), (c:Chunk {id: $chunk_id})
            MERGE (d)-[:CONTAINS]->(c)
            """,
            parameters={"document_id": document_id, "chunk_id": id},
        )

    def write_document_with_chunks(self, source, title: str, chunks: list) -> None:
        """Persist one document and all of its chunks in a single transaction."""
        self._conn.execute("BEGIN TRANSACTION")
        try:
            self._write_document(source.id, source.source_url, source.doc_type, title)
            for chunk in chunks:
                self._write_chunk(
                    id=chunk.id,
                    document_id=chunk.provenance.document_id,
                    content=chunk.content,
                    section_heading=chunk.provenance.section_heading,
                    page_number=chunk.provenance.page_number,
                )
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def read_chunks(self, unextracted_only: bool = False) -> list[ChunkRecord]:
        if unextracted_only:
            result = self._conn.execute(
                "MATCH (c:Chunk) WHERE c.extracted_at IS NULL RETURN c.id, c.document_id, c.content"
            )
        else:
            result = self._conn.execute("MATCH (c:Chunk) RETURN c.id, c.document_id, c.content")
        chunks = []
        while result.has_next():
            row = result.get_next()
            chunks.append(ChunkRecord(id=row[0], document_id=row[1], content=row[2]))
        return chunks

    def mark_chunks_extracted(self, chunk_ids: list[str]) -> None:
        """Stamp chunk(s) as extracted so they are skipped in future extract runs."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "MATCH (c:Chunk) WHERE c.id IN $ids SET c.extracted_at = $ts",
            parameters={"ids": chunk_ids, "ts": ts},
        )

    def skip_unextracted_chunks(self) -> int:
        """Stamp all currently-unextracted chunks as skipped. Returns count stamped."""
        from datetime import datetime, timezone
        chunks = self.read_chunks(unextracted_only=True)
        if not chunks:
            return 0
        ids = [c.id for c in chunks]
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "MATCH (c:Chunk) WHERE c.id IN $ids SET c.extracted_at = $ts",
            parameters={"ids": ids, "ts": ts},
        )
        return len(ids)

    def read_document_chunk_counts(self, chunk_ids: set[str] | None = None) -> list[DocumentChunkCount]:
        result = self._conn.execute(
            "MATCH (d:Document)-[:CONTAINS]->(c:Chunk) RETURN d.id, d.source_url, d.title, c.id, count(c)"
        )
        # Aggregate in Python so we can filter to a specific chunk_id set if provided
        from collections import defaultdict
        doc_meta: dict[str, tuple[str, str]] = {}
        doc_counts: dict[str, int] = defaultdict(int)
        while result.has_next():
            row = result.get_next()
            doc_id, source_url, title, chunk_id = row[0], row[1], row[2], row[3]
            doc_meta[doc_id] = (source_url, title)
            if chunk_ids is None or chunk_id in chunk_ids:
                doc_counts[doc_id] += 1
        docs = []
        for doc_id, count in doc_counts.items():
            source_url, title = doc_meta[doc_id]
            docs.append(DocumentChunkCount(document_id=doc_id, source_url=source_url, title=title, total_chunks=count))
        return docs

    def clear_document_extraction(self, document_id: str) -> int:
        """Delete all RELATES_TO triples for chunks in document_id. Returns count of cleared chunks."""
        r = self._conn.execute(
            "MATCH (d:Document {id: $id})-[:CONTAINS]->(c:Chunk) RETURN c.id",
            parameters={"id": document_id},
        )
        chunk_ids = []
        while r.has_next():
            chunk_ids.append(r.get_next()[0])
        if not chunk_ids:
            return 0
        self._conn.execute(
            "MATCH ()-[rel:RELATES_TO]->() WHERE rel.chunk_id IN $ids DELETE rel",
            parameters={"ids": chunk_ids},
        )
        self._conn.execute(
            "MATCH (c:Chunk) WHERE c.id IN $ids SET c.extracted_at = NULL",
            parameters={"ids": chunk_ids},
        )
        return len(chunk_ids)

    def clear_extraction(self) -> None:
        self._conn.execute("MATCH ()-[r:RELATES_TO]->() DELETE r")
        self._conn.execute("MATCH (e:Entity) DELETE e")
        self._conn.execute("MATCH (c:Chunk) SET c.extracted_at = NULL")

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
    ) -> None:
        self._conn.execute(
            "MERGE (e:Entity {name: $name}) SET e.type = $type",
            parameters={"name": subject, "type": subject_type},
        )
        self._conn.execute(
            "MERGE (e:Entity {name: $name}) SET e.type = $type",
            parameters={"name": object_, "type": object_type},
        )
        self._conn.execute(
            """
            MATCH (s:Entity {name: $subject}), (o:Entity {name: $object_})
            CREATE (s)-[:RELATES_TO {
                relation: $relation,
                certainty_score: $certainty_score,
                chunk_id: $chunk_id,
                raw_certainty_score: $raw_certainty_score,
                evidence_text: $evidence_text,
                evidence_char_start: $evidence_char_start,
                evidence_char_end: $evidence_char_end,
                evidence_alignment_score: $evidence_alignment_score,
                entity_anchor_score: $entity_anchor_score,
                evidence_scope_score: $evidence_scope_score,
                confidence_version: $confidence_version
            }]->(o)
            """,
            parameters={
                "subject": subject,
                "object_": object_,
                "relation": relation,
                "certainty_score": certainty_score,
                "chunk_id": chunk_id,
                "raw_certainty_score": raw_certainty_score,
                "evidence_text": evidence_text,
                "evidence_char_start": evidence_char_start,
                "evidence_char_end": evidence_char_end,
                "evidence_alignment_score": evidence_alignment_score,
                "entity_anchor_score": entity_anchor_score,
                "evidence_scope_score": evidence_scope_score,
                "confidence_version": confidence_version,
            },
        )
