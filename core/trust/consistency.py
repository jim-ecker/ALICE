"""Self-consistency signal.

Re-run the chunk extraction k extra times at temperature > 0 and count how
often each canonicalized triple reappears. The frequency count/runs is an
extraction-stability estimate; the full posterior is Beta(count+1, runs-count+1),
derivable from the two stored integers.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.llm.ingest_confidence import normalize_text

CanonicalKey = tuple[str, str, str]


@dataclass(frozen=True)
class ConsistencyResult:
    freq: float
    count: int
    runs: int

    @property
    def alpha(self) -> float:
        return self.count + 1.0

    @property
    def beta(self) -> float:
        return self.runs - self.count + 1.0


def canonical_key(subject: str, relation: str, object_: str) -> CanonicalKey:
    return (normalize_text(subject), normalize_text(relation), normalize_text(object_))


def count_across_runs(
    base_keys: set[CanonicalKey],
    rerun_key_sets: list[set[CanonicalKey]],
) -> dict[CanonicalKey, ConsistencyResult]:
    """Frequency of each base-run triple across all runs (base run included)."""
    runs = 1 + len(rerun_key_sets)
    results: dict[CanonicalKey, ConsistencyResult] = {}
    for key in base_keys:
        count = 1 + sum(1 for keys in rerun_key_sets if key in keys)
        results[key] = ConsistencyResult(freq=count / runs, count=count, runs=runs)
    return results
