from dataclasses import dataclass


@dataclass
class SheafConfig:
    stalk_dim: int = 32
    radius: int = 3
    max_nodes: int = 2000
    ridge: float = 1e-6
    beta: float = 4.0
    max_context_chunks: int = 40
    restriction: str = "identity"  # "identity" | "diagonal" | "typed" | "orthogonal"
    abstain_threshold: float = 0.5
    solver: str = "cg"  # "cg" | "spsolve"
