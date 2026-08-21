"""Unified retrieval interface.

    results = retrieve(graph, conditions, algorithm="pcst")
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .base import RetrievalResult
from .bfs import retrieve_bfs
from .pcst import retrieve_pcst
from .steiner_keyword import retrieve_steiner


def retrieve(
    graph: nx.MultiDiGraph,
    conditions: list[dict[str, Any]],
    *,
    algorithm: str = "bfs",
    max_hops: int = 4,
    strict: bool = True,
    soft_threshold: float = 0.8,
    rho: float = 0.0,
    condition_prize: float = 10.0,
) -> list[RetrievalResult]:
    algo = algorithm.lower()
    if algo == "bfs":
        return retrieve_bfs(graph, conditions, max_hops=max_hops, strict=strict,
                            soft_threshold=soft_threshold, rho=rho)
    if algo == "pcst":
        return retrieve_pcst(graph, conditions, max_hops=max_hops, strict=strict,
                              soft_threshold=soft_threshold, rho=rho,
                              condition_prize=condition_prize)
    if algo == "steiner":
        return retrieve_steiner(graph, conditions, max_hops=max_hops, strict=strict,
                                soft_threshold=soft_threshold, rho=rho)
    raise ValueError(f"Unknown algorithm: {algorithm!r} (use bfs/pcst/steiner)")
