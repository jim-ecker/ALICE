import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .config import SheafConfig


def harmonic_extend(
    L: scipy.sparse.csr_matrix,
    idx: dict[str, int],
    d: int,
    boundary: list[str],
    x_B: dict[str, np.ndarray],
    cfg: SheafConfig,
) -> tuple[np.ndarray, float, float]:
    """
    Returns (x, dirichlet_energy, abstain_energy).
    x: full stacked solution vector, shape (n*d,)
    dirichlet_energy: x^T L x at optimum
    abstain_energy: dirichlet_energy (used as abstain proxy — cheap, no Schur solve)
    """
    size = L.shape[0]
    B = [i for bn in boundary for i in range(idx[bn], idx[bn] + d)]
    b_set = set(B)
    U = [i for i in range(size) if i not in b_set]

    L_csc = L.tocsc()
    L_UU = L_csc[U][:, U] + cfg.ridge * scipy.sparse.eye(len(U), format="csc")
    L_UB = L_csc[U][:, B]

    xb = np.concatenate([x_B[bn] for bn in boundary])
    rhs = -(L_UB @ xb)

    if cfg.solver == "cg":
        xU, _info = scipy.sparse.linalg.cg(L_UU.tocsr(), rhs, rtol=1e-4, atol=0.0, maxiter=200)
    else:
        xU = scipy.sparse.linalg.spsolve(L_UU.tocsc(), rhs)

    x = np.zeros(size)
    x[U] = xU
    x[B] = xb

    dirichlet_energy = float(x @ (L @ x))
    return x, dirichlet_energy, dirichlet_energy


def dirichlet_abstention(
    dirichlet_energy: float, x_B: dict[str, np.ndarray], eps: float = 1e-12
) -> float:
    """dirichlet_energy / (sum of ||x_b||^2 + eps), clamped to [0, 1]."""
    norm_sq = sum(float(v @ v) for v in x_B.values())
    return min(1.0, dirichlet_energy / (norm_sq + eps))
