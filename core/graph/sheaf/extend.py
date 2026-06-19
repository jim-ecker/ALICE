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
    abstain_energy: x_B^T S x_B where S is the Schur complement
        L_BB - L_BU L_UU^{-1} L_UB
    """
    size = L.shape[0]
    B = [i for bn in boundary for i in range(idx[bn], idx[bn] + d)]
    b_set = set(B)
    U = [i for i in range(size) if i not in b_set]

    L_csc = L.tocsc()
    L_UU = L_csc[U][:, U] + cfg.ridge * scipy.sparse.eye(len(U), format="csc")
    L_UB = L_csc[U][:, B]
    L_BB = L_csc[B][:, B].toarray()

    xb = np.concatenate([x_B[bn] for bn in boundary])

    rhs = -(L_UB @ xb)
    if cfg.solver == "cg":
        xU, _info = scipy.sparse.linalg.cg(L_UU.tocsr(), rhs, rtol=1e-10, atol=0.0)
    else:
        xU = scipy.sparse.linalg.spsolve(L_UU.tocsc(), rhs)

    x = np.zeros(size)
    x[U] = xU
    x[B] = xb

    # Schur complement: S = L_BB - L_UB.T @ L_UU^{-1} @ L_UB
    L_UB_dense = L_UB.toarray()
    Y = scipy.sparse.linalg.spsolve(L_UU.tocsc(), L_UB_dense)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    S = L_BB - L_UB_dense.T @ Y

    dirichlet_energy = float(x @ (L @ x))
    abstain_energy = float(xb @ S @ xb)
    return x, dirichlet_energy, abstain_energy


def schur_abstention(
    abstain_energy: float, x_B: dict[str, np.ndarray], eps: float = 1e-12
) -> float:
    """abstain_energy / (sum of ||x_b||^2 + eps), clamped to [0, 1]."""
    norm_sq = sum(float(v @ v) for v in x_B.values())
    return min(1.0, abstain_energy / (norm_sq + eps))
