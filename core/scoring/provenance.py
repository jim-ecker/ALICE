from __future__ import annotations

import kuzu

from core.graph.retrieval import RetrievedTriple


class ProvenanceScorer:
    """Counts distinct chunks that independently produced the same (s, relation, o) triple.

    A count > 1 means multiple source passages corroborate the fact, which is a
    strong signal of reliability independent of any single extraction's certainty.
    """

    def __init__(self, conn: kuzu.Connection) -> None:
        self._conn = conn

    def count_batch(self, triples: list[RetrievedTriple]) -> list[int]:
        """Return provenance count for each triple (same order as input)."""
        if not triples:
            return []

        # Deduplicate to minimise query count
        unique_sro: dict[tuple[str, str, str], int] = {}
        for t in triples:
            key = (t.subject, t.relation, t.object_)
            if key not in unique_sro:
                result = self._conn.execute(
                    """
                    MATCH (e1:Entity {name: $s})-[r:RELATES_TO {relation: $rel}]->(e2:Entity {name: $o})
                    RETURN count(*)
                    """,
                    parameters={"s": t.subject, "rel": t.relation, "o": t.object_},
                )
                count = result.get_next()[0] if result.has_next() else 1
                unique_sro[key] = int(count)

        return [unique_sro[(t.subject, t.relation, t.object_)] for t in triples]
