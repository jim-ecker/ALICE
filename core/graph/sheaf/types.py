from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    u: str  # head entity name
    v: str  # tail entity name
    relation: str
    certainty: float  # c_e in (0,1]
    chunk_id: str


@dataclass
class ScoredEdge:
    edge: Edge
    activation: float
    residual: float
    consistency: float
    score: float


@dataclass
class RetrievalResult:
    chunk_ids: list[str]
    edges: list[ScoredEdge]
    abstain: float
    n_nodes: int
    n_edges: int
