from pathlib import Path

import kuzu

from .base import ChunkRecord, DocumentChunkCount, GraphStore

_SCHEMA = [
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
        chunk_id STRING
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

    def write_document(self, id: str, source_url: str | None, doc_type: str, title: str = "") -> None:
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
            CREATE (s)-[:RELATES_TO {relation: $relation, certainty_score: $certainty_score, chunk_id: $chunk_id}]->(o)
            """,
            parameters={
                "subject": subject,
                "object_": object_,
                "relation": relation,
                "certainty_score": certainty_score,
                "chunk_id": chunk_id,
            },
        )
