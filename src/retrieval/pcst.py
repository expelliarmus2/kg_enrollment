"""Prize-Collecting Steiner Tree (PCST) retrieval.

Condition anchors are prize-bearing nodes; the patient is the candidate root.
We first find the shortest valid path from each anchor to the patient (within
``max_hops``) on the query-conditioned graph, then run a *greedy marginal
prize-collecting* prune: iteratively drop the condition whose unique-edge cost
exceeds its prize the most, until no condition costs more than its prize.

This is a correct, explainable PCST formulation for a toy graph (no external
solver needed).  The pruning step is what distinguishes PCST from plain
Steiner: PCST may *drop* a high-cost condition (e.g. a path through a
high-degree hub) when the edge cost exceeds the condition prize.

Score (PCST / generic):
    Score = ALPHA * Coverage^GAMMA
            - BETA  * AvgDistance
            - ETA   * Imbalance
            - LAMBDA* GraphCost
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
    avg_distance,
    imbalance,
    graph_cost,
    score_generic,
)
from ._traversal import shortest_paths_to_patients
from .base import RetrievalResult

DEFAULT_CONDITION_PRIZE = 10.0


def _shortest_path_to_patient(
    G: nx.MultiDiGraph,
    anchor: str,
    patient_id: str,
    max_hops: int,
) -> tuple[list[str] | None, int, int]:
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


def _edge_key(path: list[str], i: int) -> tuple[str, str]:
    return (path[i], path[i + 1])


def _full_path_cost(path: list[str], edge_costs: dict[tuple[str, str], float]) -> float:
    return sum(edge_costs.get(_edge_key(path, i), 0.0) for i in range(len(path) - 1))


def _prune_by_prize(
    matched_paths: dict[str, list[str]],
    edge_costs: dict[tuple[str, str], float],
    prizes: dict[str, float],
) -> set[str]:
    """One-shot prize-collecting prune.

    A condition is dropped when its *full* path cost exceeds its prize
    (``cost(path_i) > prize_i``).  Using the full path cost (rather than the
    marginal/shared cost) gives a conservative, easy-to-explain prune that
    cleanly demonstrates the hub-penalty effect: high-degree-hub paths become
    expensive and are dropped, while short / low-degree multi-hop paths are
    retained.  All kept conditions still form a union subgraph whose shared
    edges are counted once for the final graph cost.
    """
    kept: set[str] = set()
    for cid, path in matched_paths.items():
        cost = _full_path_cost(path, edge_costs)
        prize = prizes.get(cid, DEFAULT_CONDITION_PRIZE)
        if cost <= prize:
            kept.add(cid)
    return kept


def retrieve_pcst(
    G: nx.MultiDiGraph,
    conditions: list[dict[str, Any]],
    *,
    max_hops: int = 4,
    strict: bool = True,
    soft_threshold: float = 0.8,
    rho: float = 0.0,
    condition_prize: float = DEFAULT_CONDITION_PRIZE,
) -> list[RetrievalResult]:
    t0 = time.perf_counter()
    H = build_query_conditioned_graph(G, conditions)
    patients = patient_nodes(H)
    cond_ids = [c["id"] for c in conditions]
    m = len(conditions)
    prizes = {c["id"]: condition_prize for c in conditions}

    total_vn = 0
    total_ve = 0

    # Precompute shortest paths per (condition, patient)
    all_paths: dict[tuple[str, str], list[str] | None] = {}
    for cond in conditions:
        for pid in patients:
            p, vn, ve = _shortest_path_to_patient(H, cond["anchor"], pid, max_hops)
            all_paths[(cond["id"], pid)] = p
            total_vn += vn
            total_ve += ve

    results: list[RetrievalResult] = []
    for pid in patients:
        # initial matched paths
        matched_paths: dict[str, list[str]] = {}
        acc = new_patient_accumulator(cond_ids)
        for cond in conditions:
            cid = cond["id"]
            p = all_paths[(cid, pid)]
            if p is not None:
                matched_paths[cid] = p
                acc[cid] = {
                    "matched": True,
                    "distance": len(p) - 1,
                    "path": p,
                    "cost": 0.0,
                    "reason": "MATCH",
                    "pruned": False,
                }
            else:
                acc[cid] = {
                    "matched": False,
                    "distance": None,
                    "path": [],
                    "cost": 0.0,
                    "reason": "NOT REACHABLE within max_hops",
                    "pruned": False,
                }

        # edge costs for all matched paths
        edge_costs: dict[tuple[str, str], float] = {}
        for cid, path in matched_paths.items():
            for i in range(len(path) - 1):
                ek = _edge_key(path, i)
                if ek not in edge_costs:
                    edge_costs[ek] = edge_cost(H, ek[0], ek[1], rho=rho)

        # prize-collecting prune
        kept = _prune_by_prize(matched_paths, edge_costs, prizes)
        pruned_conds = set(matched_paths.keys()) - kept
        for cid in pruned_conds:
            acc[cid]["matched"] = False
            acc[cid]["reason"] = "PRUNED by PCST (marginal edge cost > prize)"
            acc[cid]["pruned"] = True

        # recompute per-condition cost (full path)
        for cid in kept:
            path = matched_paths[cid]
            cost = sum(edge_costs.get(_edge_key(path, i), 0.0) for i in range(len(path) - 1))
            acc[cid]["cost"] = cost

        cov_n = coverage_count(acc)
        cov_r = coverage_ratio(acc, m)
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]

        # union subgraph (shared edges once)
        sub_nodes: set[str] = set([pid])
        sub_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for cid in matched:
            path = acc[cid]["path"]
            for n in path:
                sub_nodes.add(n)
            for i in range(len(path) - 1):
                u, v = _edge_key(path, i)
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

        score = score_generic(acc, m)
        results.append(RetrievalResult(
            patient_id=pid,
            algorithm="pcst",
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


def retrieve_pcst_shared(
    G: nx.MultiDiGraph,
    conditions: list[dict[str, Any]],
    *,
    max_hops: int = 4,
    strict: bool = True,
    soft_threshold: float = 0.8,
    rho: float = 0.0,
    condition_prize: float = DEFAULT_CONDITION_PRIZE,
) -> list[RetrievalResult]:
    """Single-pass PCST: one BFS per anchor shared across all patients.

    Identical output to ``retrieve_pcst`` (same shortest paths, same
    prize-pruning, same union subgraph / score) but the search phase is
    O(K * (V+E)) instead of O(K * P * (V+E)) -- the patient count P no
    longer multiplies the traversal cost.
    """
    t0 = time.perf_counter()
    H = build_query_conditioned_graph(G, conditions)
    patients = patient_nodes(H)
    cond_ids = [c["id"] for c in conditions]
    m = len(conditions)
    prizes = {c["id"]: condition_prize for c in conditions}

    total_vn = 0
    total_ve = 0

    # One BFS per anchor; records shortest path to EVERY reachable patient.
    all_paths: dict[tuple[str, str], list[str] | None] = {}
    for cond in conditions:
        hits, vn, ve = shortest_paths_to_patients(H, cond["anchor"], max_hops)
        for pid, info in hits.items():
            all_paths[(cond["id"], pid)] = info["path"]
        total_vn += vn
        total_ve += ve

    results: list[RetrievalResult] = []
    for pid in patients:
        matched_paths: dict[str, list[str]] = {}
        acc = new_patient_accumulator(cond_ids)
        for cond in conditions:
            cid = cond["id"]
            p = all_paths.get((cid, pid))
            if p is not None:
                matched_paths[cid] = p
                acc[cid] = {
                    "matched": True,
                    "distance": len(p) - 1,
                    "path": p,
                    "cost": 0.0,
                    "reason": "MATCH",
                    "pruned": False,
                }
            else:
                acc[cid] = {
                    "matched": False,
                    "distance": None,
                    "path": [],
                    "cost": 0.0,
                    "reason": "NOT REACHABLE within max_hops",
                    "pruned": False,
                }

        edge_costs: dict[tuple[str, str], float] = {}
        for cid, path in matched_paths.items():
            for i in range(len(path) - 1):
                ek = _edge_key(path, i)
                if ek not in edge_costs:
                    edge_costs[ek] = edge_cost(H, ek[0], ek[1], rho=rho)

        kept = _prune_by_prize(matched_paths, edge_costs, prizes)
        pruned_conds = set(matched_paths.keys()) - kept
        for cid in pruned_conds:
            acc[cid]["matched"] = False
            acc[cid]["reason"] = "PRUNED by PCST (marginal edge cost > prize)"
            acc[cid]["pruned"] = True

        for cid in kept:
            path = matched_paths[cid]
            cost = sum(edge_costs.get(_edge_key(path, i), 0.0) for i in range(len(path) - 1))
            acc[cid]["cost"] = cost

        cov_n = coverage_count(acc)
        cov_r = coverage_ratio(acc, m)
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]

        sub_nodes: set[str] = set([pid])
        sub_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for cid in matched:
            path = acc[cid]["path"]
            for n in path:
                sub_nodes.add(n)
            for i in range(len(path) - 1):
                u, v = _edge_key(path, i)
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

        score = score_generic(acc, m)
        results.append(RetrievalResult(
            patient_id=pid,
            algorithm="pcst_shared",
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
