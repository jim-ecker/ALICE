import numpy as np
import scipy.sparse

from .types import Edge
from .restrictions import RestrictionProvider


def build_block_index(nodes: list[str], d: int) -> dict[str, int]:
    """Maps node name -> first row/col index of its d×d block in L."""
    return {name: i * d for i, name in enumerate(nodes)}


def _accumulate_block(rows, cols, vals, ri, ci, block, d):
    r = np.repeat(np.arange(d), d) + ri
    c = np.tile(np.arange(d), d) + ci
    rows.append(r)
    cols.append(c)
    vals.append(block.reshape(-1))


def assemble_sheaf_laplacian(
    nodes: list[str],
    edges: list[Edge],
    d: int,
    restriction: RestrictionProvider,
) -> tuple[scipy.sparse.csr_matrix, dict[str, int]]:
    """Returns (L, block_index). L is symmetric PSD, size (n*d, n*d), CSR format."""
    idx = build_block_index(nodes, d)
    n = len(nodes)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for e in edges:
        c = e.certainty
        Fu = restriction.map(e.relation, "head", d)
        Fv = restriction.map(e.relation, "tail", d)
        bu = idx[e.u]
        bv = idx[e.v]
        _accumulate_block(rows, cols, vals, bu, bu, c * (Fu.T @ Fu), d)
        _accumulate_block(rows, cols, vals, bv, bv, c * (Fv.T @ Fv), d)
        _accumulate_block(rows, cols, vals, bu, bv, -c * (Fu.T @ Fv), d)
        _accumulate_block(rows, cols, vals, bv, bu, -c * (Fv.T @ Fu), d)

    size = n * d
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        L = scipy.sparse.coo_matrix((v, (r, c)), shape=(size, size)).tocsr()
    else:
        L = scipy.sparse.csr_matrix((size, size))
    return L, idx
