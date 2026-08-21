"""Bounded constrained BFS baseline.

For each condition anchor, perform a BFS up to ``max_hops`` on the
query-conditioned graph (numeric-failing edges already removed).  Collect
reachable patients with their distance and path.  If a shorter path to a
patient already exists for a given condition, the shorter one wins.

A patient is a candidate for a condition if reachable within ``max_hops``.
Coverage / scoring then determines the final cohort.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import networkx as nx

from ..constraints import build_query_conditioned_graph, is_numeric
from ..graph_model import patient_nodes, edge_cost, degree
from ..scoring import (
    new_patient_accumulator,
    coverage_count,
    coverage_ratio,
    avg_distance,
    score_generic,
    passes_mode,
)
from .base import RetrievalResult


def _bfs_from_anchor(
    G: nx.MultiDiGraph,
    anchor: str,
    max_hops: int,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """BFS from ``anchor`` up to ``max_hops``. Returns patient hits + visited counts."""
    # Each queue entry: (current_node, distance, path)
    visited_nodes: set[str] = set()
    visited_edges = 0
    best: dict[str, dict[str, Any]] = {}

    queue: deque[tuple[str, int, list[str]]] = deque()
    queue.append((anchor, 0, [anchor]))
    visited_nodes.add(anchor)

    while queue:
        node, dist, path = queue.popleft()
        if dist >= max_hops:
            continue
        for _, nbr, k, edata in G.out_edges(node, keys=True, data=True):
            visited_edges += 1
            if nbr in path:
                continue  # avoid cycles
            new_path = path + [nbr]
            new_dist = dist + 1
            visited_nodes.add(nbr)
            if G.nodes[nbr].get("type") == "patient":
                if nbr not in best or new_dist < best[nbr]["distance"]:
                    best[nbr] = {
                        "matched": True,
                        "distance": new_dist,
                        "path": new_path,
                        "cost": 0.0,
                        "reason": "MATCH",
                    }
            # continue expanding even past a patient (patient is not a
            # traversal hub per design principle #1, but we still allow
            # exploring other branches)
            queue.append((nbr, new_dist, new_path))
    return best, len(visited_nodes), visited_edges


def retrieve_bfs(
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

    total_visited_nodes = 0
    total_visited_edges = 0

    # Per condition: patient hits
    cond_hits: dict[str, dict[str, dict[str, Any]]] = {}
    for cond in conditions:
        hits, vn, ve = _bfs_from_anchor(H, cond["anchor"], max_hops)
        cond_hits[cond["id"]] = hits
        total_visited_nodes += vn
        total_visited_edges += ve

    results: list[RetrievalResult] = []
    for pid in patients:
        acc = new_patient_accumulator(cond_ids)
        for cid, hits in cond_hits.items():
            if pid in hits:
                acc[cid] = hits[pid]
        # compute edge costs along paths
        for cid, cond in zip(cond_ids, conditions):
            if acc[cid]["matched"]:
                path = acc[cid]["path"]
                cost = 0.0
                for i in range(len(path) - 1):
                    cost += edge_cost(H, path[i], path[i + 1], rho=rho)
                acc[cid]["cost"] = cost
        cov_n = coverage_count(acc)
        cov_r = coverage_ratio(acc, m)
        strict_match = cov_r >= 1.0 - 1e-9
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]
        # collect nodes/edges of union subgraph
        sub_nodes: set[str] = set([pid])
        sub_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for cid in matched:
            path = acc[cid]["path"]
            for n in path:
                sub_nodes.add(n)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # pick first edge key
                for _, vv, k, edata in H.out_edges(u, keys=True, data=True):
                    if vv == v and (u, v, k) not in seen_edges:
                        seen_edges.add((u, v, k))
                        row = {"source": u, "target": v, "relation_type": k}
                        row.update({kk: vv for kk, vv in edata.items()})
                        sub_edges.append(row)
                        break
        union_cost = sum(e.get("_cost", 0.0) for e in sub_edges)
        # attach per-edge cost for display
        for e in sub_edges:
            e["_cost"] = edge_cost(H, e["source"], e["target"], rho=rho)
        union_cost = sum(e["_cost"] for e in sub_edges)
        if passes_mode(acc, m, strict=strict, soft_threshold=soft_threshold) or cov_n > 0:
            score = score_generic(acc, m)
        else:
            score = score_generic(acc, m)
        results.append(RetrievalResult(
            patient_id=pid,
            algorithm="bfs",
            score=score,
            coverage_count=cov_n,
            coverage_ratio=cov_r,
            matched_conditions=matched,
            condition_paths={cid: acc[cid] for cid in cond_ids},
            nodes=sorted(sub_nodes),
            edges=sub_edges,
            graph_cost=union_cost,
            strict_match=strict_match,
            visited_nodes=total_visited_nodes,
            visited_edges=total_visited_edges,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        ))
    return results
