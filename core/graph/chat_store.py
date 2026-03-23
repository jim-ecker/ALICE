from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import kuzu

_SCHEMA = [
    """
    CREATE NODE TABLE IF NOT EXISTS Conversation(
        id STRING,
        title STRING,
        created_at STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Message(
        id STRING,
        conversation_id STRING,
        role STRING,
        content STRING,
        created_at STRING,
        citation_chunk_ids STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS HAS_MESSAGE(
        FROM Conversation TO Message
    )
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ConversationRecord:
    id: str
    title: str
    created_at: str


@dataclass
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    citation_chunk_ids: list[str]
    citations_json: str = ""


class ChatStore:
    def __init__(self, db_path: "str | Path | kuzu.Database") -> None:
        if isinstance(db_path, kuzu.Database):
            self._db = db_path
        else:
            self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        for statement in _SCHEMA:
            self._conn.execute(statement)
        try:
            self._conn.execute("ALTER TABLE Message ADD citations_json STRING DEFAULT ''")
        except Exception:
            pass

    def create_conversation(self, title: str) -> ConversationRecord:
        conv_id = _new_id()
        created_at = _now()
        self._conn.execute(
            "CREATE (:Conversation {id: $id, title: $title, created_at: $created_at})",
            parameters={"id": conv_id, "title": title, "created_at": created_at},
        )
        return ConversationRecord(id=conv_id, title=title, created_at=created_at)

    def list_conversations(self) -> list[ConversationRecord]:
        result = self._conn.execute(
            "MATCH (c:Conversation) RETURN c.id, c.title, c.created_at ORDER BY c.created_at DESC"
        )
        convs = []
        while result.has_next():
            row = result.get_next()
            convs.append(ConversationRecord(id=row[0], title=row[1], created_at=row[2]))
        return convs

    def get_conversation(self, id: str) -> ConversationRecord | None:
        result = self._conn.execute(
            "MATCH (c:Conversation {id: $id}) RETURN c.id, c.title, c.created_at",
            parameters={"id": id},
        )
        if result.has_next():
            row = result.get_next()
            return ConversationRecord(id=row[0], title=row[1], created_at=row[2])
        return None

    def update_conversation_title(self, id: str, title: str) -> None:
        self._conn.execute(
            "MATCH (c:Conversation {id: $id}) SET c.title = $title",
            parameters={"id": id, "title": title},
        )

    def delete_conversation(self, id: str) -> None:
        # Delete messages first
        self._conn.execute(
            "MATCH (c:Conversation {id: $id})-[:HAS_MESSAGE]->(m:Message) DETACH DELETE m",
            parameters={"id": id},
        )
        self._conn.execute(
            "MATCH (c:Conversation {id: $id}) DETACH DELETE c",
            parameters={"id": id},
        )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citation_chunk_ids: list[str],
        citations_json: str = "",
    ) -> MessageRecord:
        msg_id = _new_id()
        created_at = _now()
        self._conn.execute(
            """
            CREATE (:Message {
                id: $id,
                conversation_id: $conversation_id,
                role: $role,
                content: $content,
                created_at: $created_at,
                citation_chunk_ids: $citation_chunk_ids,
                citations_json: $citations_json
            })
            """,
            parameters={
                "id": msg_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "created_at": created_at,
                "citation_chunk_ids": json.dumps(citation_chunk_ids),
                "citations_json": citations_json,
            },
        )
        self._conn.execute(
            """
            MATCH (c:Conversation {id: $conv_id}), (m:Message {id: $msg_id})
            CREATE (c)-[:HAS_MESSAGE]->(m)
            """,
            parameters={"conv_id": conversation_id, "msg_id": msg_id},
        )
        return MessageRecord(
            id=msg_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=created_at,
            citation_chunk_ids=citation_chunk_ids,
            citations_json=citations_json,
        )

    def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        result = self._conn.execute(
            """
            MATCH (c:Conversation {id: $conv_id})-[:HAS_MESSAGE]->(m:Message)
            RETURN m.id, m.conversation_id, m.role, m.content, m.created_at, m.citation_chunk_ids, m.citations_json
            ORDER BY m.created_at ASC
            """,
            parameters={"conv_id": conversation_id},
        )
        messages = []
        while result.has_next():
            row = result.get_next()
            try:
                chunk_ids = json.loads(row[5]) if row[5] else []
            except (json.JSONDecodeError, TypeError):
                chunk_ids = []
            messages.append(
                MessageRecord(
                    id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    content=row[3],
                    created_at=row[4],
                    citation_chunk_ids=chunk_ids,
                    citations_json=row[6] or "",
                )
            )
        return messages
