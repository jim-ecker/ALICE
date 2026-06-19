from __future__ import annotations

import numpy as np

from .config import SheafConfig
from .restrictions import RestrictionProvider
from .types import Edge, ScoredEdge


def readout(
    edges: list[Edge],
    x: np.ndarray,
    idx: dict[str, int],
    d: int,
    restriction: RestrictionProvider,
    cfg: SheafConfig,
) -> list[ScoredEdge]:
    """Per-edge scores (spec §1.2). Returns list sorted descending by score."""
    beta = cfg.beta
    eps = cfg.ridge
    scored: list[ScoredEdge] = []
    for edge in edges:
        iu = idx.get(edge.u)
        iv = idx.get(edge.v)
        if iu is None or iv is None:
            continue
        x_u = x[iu : iu + d]
        x_v = x[iv : iv + d]

        a_e = 0.5 * (float(x_u @ x_u) + float(x_v @ x_v))

        a_head = restriction.map(edge.relation, "head", d)
        a_tail = restriction.map(edge.relation, "tail", d)
        coboundary = a_head @ x_u - a_tail @ x_v
        rho_e = float(coboundary @ coboundary)

        kappa_e = float(np.exp(-beta * rho_e / (a_e + eps)))
        score = float(edge.certainty) * a_e * kappa_e

        scored.append(
            ScoredEdge(
                edge=edge,
                activation=a_e,
                residual=rho_e,
                consistency=kappa_e,
                score=score,
            )
        )

    scored.sort(key=lambda se: se.score, reverse=True)
    return scored


def union_top_chunks(scored: list[ScoredEdge], max_chunks: int) -> list[str]:
    """Walk scored edges in order, collect distinct chunk_ids until max_chunks reached."""
    out: list[str] = []
    seen: set[str] = set()
    for se in scored:
        cid = se.edge.chunk_id
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
            if len(out) >= max_chunks:
                break
    return out
