"""Steiner-tree based keyword search.

Each query condition is a keyword group / terminal anchor.  For every patient
candidate we find the shortest valid constrained path from EACH condition
anchor to the patient (within ``max_hops``), then union all paths into
``G_p``.  Shared edges are counted only once in the union cost.

Score (Steiner):
    Score = ALPHA * Coverage - BETA * C(G_p) - ETA * MaxDistance
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import networkx as nx

from ..constraints import build_query_conditioned_graph
from ..graph_model import patient_nodes, edge_cost
from ..scoring import (
    new_patient_accumulator,
    coverage_count,
    coverage_ratio,
    max_distance,
    score_steiner,
    passes_mode,
)
from .base import RetrievalResult


def _shortest_path_to_patient(
    G: nx.MultiDiGraph,
    anchor: str,
    patient_id: str,
    max_hops: int,
) -> tuple[list[str] | None, int, int]:
    """Shortest path (BFS) from anchor to patient within max_hops.

    Returns (path_or_None, visited_nodes, visited_edges).
    """
    visited_nodes: set[str] = {anchor}
    visited_edges = 0
    queue: deque[tuple[str, int, list[str]]] = deque([(anchor, 0, [anchor])])
    while queue:
        node, dist, path = queue.popleft()
        if dist >= max_hops:
            continue
        for _, nbr, k, edata in G.out_edges(node, keys=True, data=True):
            visited_edges += 1
            if nbr in path:
                continue
            new_path = path + [nbr]
            new_dist = dist + 1
            visited_nodes.add(nbr)
            if nbr == patient_id:
                return new_path, len(visited_nodes), visited_edges
            queue.append((nbr, new_dist, new_path))
    return None, len(visited_nodes), visited_edges


def retrieve_steiner(
    G: nx.MultiDiGraph,
    conditions: list[dict[str, Any]],
    *,
    max_hops: int = 4,
    strict: bool = True,
    soft_threshold: float = 0.8,
    rho: float = 0.0,
) -> list[RetrievalResult]:
    t0 = time.perf_counter()
    H = build_query_conditioned_graph(G, conditions)
    patients = patient_nodes(H)
    cond_ids = [c["id"] for c in conditions]
    m = len(conditions)

    total_vn = 0
    total_ve = 0

    # Precompute per-(condition, patient) shortest paths
    paths: dict[tuple[str, str], list[str] | None] = {}
    for cond in conditions:
        for pid in patients:
            p, vn, ve = _shortest_path_to_patient(H, cond["anchor"], pid, max_hops)
            paths[(cond["id"], pid)] = p
            total_vn += vn
            total_ve += ve

    results: list[RetrievalResult] = []
    for pid in patients:
        acc = new_patient_accumulator(cond_ids)
        for cond in conditions:
            cid = cond["id"]
            p = paths[(cid, pid)]
            if p is not None:
                cost = 0.0
                for i in range(len(p) - 1):
                    cost += edge_cost(H, p[i], p[i + 1], rho=rho)
                acc[cid] = {
                    "matched": True,
                    "distance": len(p) - 1,
                    "path": p,
                    "cost": cost,
                    "reason": "MATCH",
                }
            else:
                acc[cid] = {
                    "matched": False,
                    "distance": None,
                    "path": [],
                    "cost": 0.0,
                    "reason": "NOT REACHABLE within max_hops",
                }
        cov_n = coverage_count(acc)
        cov_r = coverage_ratio(acc, m)
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]

        # Build union subgraph G_p (shared edges counted once)
        sub_nodes: set[str] = set([pid])
        sub_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for cid in matched:
            p = acc[cid]["path"]
            for n in p:
                sub_nodes.add(n)
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                for _, vv, k, edata in H.out_edges(u, keys=True, data=True):
                    if vv == v and (u, v, k) not in seen_edges:
                        seen_edges.add((u, v, k))
                        row = {"source": u, "target": v, "relation_type": k}
                        row.update({kk: vv for kk, vv in edata.items()})
                        sub_edges.append(row)
                        break
        for e in sub_edges:
            e["_cost"] = edge_cost(H, e["source"], e["target"], rho=rho)
        union_cost = sum(e["_cost"] for e in sub_edges)

        score = score_steiner(acc, m, union_cost)
        results.append(RetrievalResult(
            patient_id=pid,
            algorithm="steiner",
            score=score,
            coverage_count=cov_n,
            coverage_ratio=cov_r,
            matched_conditions=matched,
            condition_paths={cid: acc[cid] for cid in cond_ids},
            nodes=sorted(sub_nodes),
            edges=sub_edges,
            graph_cost=union_cost,
            strict_match=cov_r >= 1.0 - 1e-9,
            visited_nodes=total_vn,
            visited_edges=total_ve,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        ))
    return results
