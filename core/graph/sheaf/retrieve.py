from __future__ import annotations

from .config import SheafConfig
from .extend import harmonic_extend, schur_abstention
from .laplacian import assemble_sheaf_laplacian
from .query_section import EntityEmbeddingProvider, build_query_section
from .readout import readout, union_top_chunks
from .restrictions import RestrictionProvider, make_restriction
from .types import Edge, RetrievalResult


def ego_subgraph(
    conn,
    anchors: list[str],
    radius: int,
    max_nodes: int,
) -> tuple[list[str], list[Edge]]:
    """Inline BFS over raw Kuzu Cypher. Expands out- and in-edges each hop."""
    visited_nodes = set(anchors)
    edges_seen: set[tuple[str, str, str, str]] = set()
    edges: list[Edge] = []
    nodes = list(anchors)
    frontier = list(anchors)

    for _ in range(radius):
        if not frontier or len(nodes) >= max_nodes:
            break
        next_frontier: list[str] = []

        res = conn.execute(
            "MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity) "
            "WHERE s.name IN $frontier "
            "RETURN s.name, r.relation, r.certainty_score, r.chunk_id, e.name",
            parameters={"frontier": frontier},
        )
        while res.has_next():
            row = res.get_next()
            u, rel, cert, cid, v = row[0], row[1], row[2], row[3], row[4]
            if v not in visited_nodes:
                if len(nodes) >= max_nodes:
                    continue
                visited_nodes.add(v)
                nodes.append(v)
                next_frontier.append(v)
            key = (u, v, rel, cid)
            if key not in edges_seen:
                edges_seen.add(key)
                edges.append(Edge(u=u, v=v, relation=rel, certainty=cert, chunk_id=cid))

        res = conn.execute(
            "MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity) "
            "WHERE e.name IN $frontier "
            "RETURN s.name, r.relation, r.certainty_score, r.chunk_id, e.name",
            parameters={"frontier": frontier},
        )
        while res.has_next():
            row = res.get_next()
            u, rel, cert, cid, v = row[0], row[1], row[2], row[3], row[4]
            if u not in visited_nodes:
                if len(nodes) >= max_nodes:
                    continue
                visited_nodes.add(u)
                nodes.append(u)
                next_frontier.append(u)
            key = (u, v, rel, cid)
            if key not in edges_seen:
                edges_seen.add(key)
                edges.append(Edge(u=u, v=v, relation=rel, certainty=cert, chunk_id=cid))

        frontier = next_frontier

    return nodes, edges


def retrieve(
    query: str,
    anchors: list[str],
    conn,
    provider: EntityEmbeddingProvider,
    cfg: SheafConfig,
    restriction: RestrictionProvider | None = None,
) -> RetrievalResult:
    """Full §3 pipeline."""
    d = cfg.stalk_dim

    nodes, edges = ego_subgraph(conn, anchors, cfg.radius, cfg.max_nodes)

    if restriction is None:
        restriction = make_restriction(cfg)

    L, idx = assemble_sheaf_laplacian(nodes, edges, d, restriction)

    node_set = set(nodes)
    boundary = [a for a in anchors if a in node_set]
    x_B = {name: build_query_section([name], provider, d)[name] for name in boundary}

    x, _dirichlet_energy, abstain_energy = harmonic_extend(
        L, idx, d, boundary, x_B, cfg
    )

    scored = readout(edges, x, idx, d, restriction, cfg)
    chunk_ids = union_top_chunks(scored, cfg.max_context_chunks)

    abstain = schur_abstention(abstain_energy, x_B, eps=cfg.ridge)

    return RetrievalResult(
        chunk_ids=chunk_ids,
        edges=scored,
        abstain=abstain,
        n_nodes=len(nodes),
        n_edges=len(edges),
    )
