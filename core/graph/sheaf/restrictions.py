import numpy as np
from typing import Protocol

from .config import SheafConfig


class RestrictionProvider(Protocol):
    def map(self, relation: str, role: str, d: int) -> np.ndarray: ...


class IdentityRestriction:
    def __init__(self) -> None:
        self._cache: dict[int, np.ndarray] = {}

    def map(self, relation: str, role: str, d: int) -> np.ndarray:
        m = self._cache.get(d)
        if m is None:
            m = np.eye(d)
            self._cache[d] = m
        return m


class DiagonalRestriction:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def map(self, relation: str, role: str, d: int) -> np.ndarray:
        raise NotImplementedError


class TypedRestriction:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def map(self, relation: str, role: str, d: int) -> np.ndarray:
        raise NotImplementedError


class OrthogonalRestriction:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def map(self, relation: str, role: str, d: int) -> np.ndarray:
        raise NotImplementedError


def make_restriction(cfg: SheafConfig) -> RestrictionProvider:
    kind = cfg.restriction
    if kind == "identity":
        return IdentityRestriction()
    if kind == "diagonal":
        return DiagonalRestriction()
    if kind == "typed":
        return TypedRestriction()
    if kind == "orthogonal":
        return OrthogonalRestriction()
    raise ValueError(f"unknown restriction kind: {kind!r}")
