"""Unified retrieval result dataclass shared by all three algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RetrievalResult:
    patient_id: str
    algorithm: str
    score: float
    coverage_count: int
    coverage_ratio: float
    matched_conditions: list[str]
    condition_paths: dict[str, Any]
    nodes: list[str]
    edges: list[dict[str, Any]]
    graph_cost: float
    strict_match: bool
    visited_nodes: int = 0
    visited_edges: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
