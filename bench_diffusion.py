"""THROWAWAY benchmark: how much do diffusion-pruning strategies compress
retrieval time / visited set?

Strategies compared (search phase = BFS from each anchor):
  bfs           baseline (one BFS per anchor, already P-independent)   [exists]
  pcst          naive per-(anchor,patient) BFS = O(K*P*(V+E))          [exists]
  pcst_shared   one BFS per anchor shared across patients             [exists]
  bfs+A         bfs + reverse-reachability prefilter R (prune dead-end branches)
  B             shared search + candidate-only emission (skip 0-coverage patients;
                a true search saving needs an index -- reported honestly)

Measured on: (1) synthetic 8-patient multi-hop graph, (2) external 100-patient KG.
Reports: visited_nodes, visited_edges, num_results, wall_ms (avg of N repeats).

Read-only w.r.t. the external path; no writes there, no graph files copied.
"""
from __future__ import annotations

import os
import pickle
import time
from collections import deque
from typing import Any

import networkx as nx

from src.graph_model import build_graph, patient_nodes, edge_cost, degree
from src.constraints import build_query_conditioned_graph, is_numeric
from src.synthetic_data import build_default_conditions, load_ground_truth
from src.scoring import (
    new_patient_accumulator, coverage_count, coverage_ratio,
    avg_distance, imbalance, score_generic, passes_mode,
)
from src.retrieval.base import RetrievalResult
from src.retrieval.bfs import retrieve_bfs
from src.retrieval.pcst import retrieve_pcst, retrieve_pcst_shared
from src.retrieval.steiner_keyword import retrieve_steiner, retrieve_steiner_shared

EXT_ROOT = "/data/qc/time_kg/kg/build_run/graph_v2"


# ---------------------------------------------------------------------------
# Variant A: BFS + reverse-reachability prefilter R
# ---------------------------------------------------------------------------
def reachability_to_patients(H: nx.MultiDiGraph, max_hops: int) -> set[str]:
    """Nodes that can reach SOME patient within max_hops (reverse BFS from patients)."""
    patients = [n for n, d in H.nodes(data=True) if d.get("type") == "patient"]
    R: set[str] = set(patients)
    frontier = list(patients)
    for _ in range(max_hops):
        nxt: list[str] = []
        for v in frontier:
            for u, _ in H.in_edges(v):
                if u not in R:
                    R.add(u)
                    nxt.append(u)
        frontier = nxt
    return R


def _bfs_from_anchor_prefiltered(H, anchor, max_hops, R):
    visited_nodes: set[str] = {anchor}
    visited_edges = 0
    best: dict[str, dict[str, Any]] = {}
    q: deque = deque([(anchor, 0, [anchor])])
    while q:
        node, dist, path = q.popleft()
        if dist >= max_hops:
            continue
        for _, nbr, k, edata in H.out_edges(node, keys=True, data=True):
            visited_edges += 1
            if nbr in path:
                continue
            # PRUNE: skip neighbors that cannot reach any patient within budget
            # (patients themselves are always kept)
            if H.nodes[nbr].get("type") != "patient" and nbr not in R:
                continue
            visited_nodes.add(nbr)
            new_path = path + [nbr]
            new_dist = dist + 1
            if H.nodes[nbr].get("type") == "patient":
                if nbr not in best or new_dist < best[nbr]["distance"]:
                    best[nbr] = {"matched": True, "distance": new_dist,
                                 "path": new_path, "cost": 0.0, "reason": "MATCH"}
            q.append((nbr, new_dist, new_path))
    return best, len(visited_nodes), visited_edges


def retrieve_bfs_prefiltered(G, conditions, *, max_hops=4, strict=True,
                             soft_threshold=0.8, rho=0.0, condition_prize=10.0):
    t0 = time.perf_counter()
    H = build_query_conditioned_graph(G, conditions)
    R = reachability_to_patients(H, max_hops)          # one reverse pass
    patients = patient_nodes(H)
    cond_ids = [c["id"] for c in conditions]
    m = len(conditions)
    total_vn = total_ve = 0
    cond_hits: dict[str, dict[str, dict[str, Any]]] = {}
    for cond in conditions:
        hits, vn, ve = _bfs_from_anchor_prefiltered(H, cond["anchor"], max_hops, R)
        cond_hits[cond["id"]] = hits
        total_vn += vn
        total_ve += ve
    results: list[RetrievalResult] = []
    for pid in patients:
        acc = new_patient_accumulator(cond_ids)
        for cid, hits in cond_hits.items():
            if pid in hits:
                acc[cid] = hits[pid]
        for cid, cond in zip(cond_ids, conditions):
            if acc[cid]["matched"]:
                path = acc[cid]["path"]
                cost = sum(edge_cost(H, path[i], path[i + 1], rho=rho)
                           for i in range(len(path) - 1))
                acc[cid]["cost"] = cost
        sub_nodes: set[str] = {pid}
        sub_edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]
        for cid in matched:
            path = acc[cid]["path"]
            for n in path:
                sub_nodes.add(n)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                for _, vv, k, edata in H.out_edges(u, keys=True, data=True):
                    if vv == v and (u, v, k) not in seen:
                        seen.add((u, v, k))
                        row = {"source": u, "target": v, "relation_type": k}
                        row.update(edata)
                        sub_edges.append(row)
                        break
        for e in sub_edges:
            e["_cost"] = edge_cost(H, e["source"], e["target"], rho=rho)
        union_cost = sum(e["_cost"] for e in sub_edges)
        score = score_generic(acc, m)
        results.append(RetrievalResult(
            patient_id=pid, algorithm="bfs+A", score=score,
            coverage_count=coverage_count(acc), coverage_ratio=coverage_ratio(acc, m),
            matched_conditions=matched, condition_paths={cid: acc[cid] for cid in cond_ids},
            nodes=sorted(sub_nodes), edges=sub_edges, graph_cost=union_cost,
            strict_match=coverage_ratio(acc, m) >= 1.0 - 1e-9,
            visited_nodes=total_vn, visited_edges=total_ve,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        ))
    return results


# ---------------------------------------------------------------------------
# Variant B: shared search + candidate-only emission (skip 0-coverage patients)
# ---------------------------------------------------------------------------
from src.retrieval._traversal import shortest_paths_to_patients


def retrieve_candidate_only(G, conditions, *, max_hops=4, strict=True,
                            soft_threshold=0.8, rho=0.0, condition_prize=10.0):
    t0 = time.perf_counter()
    H = build_query_conditioned_graph(G, conditions)
    cond_ids = [c["id"] for c in conditions]
    m = len(conditions)
    total_vn = total_ve = 0
    all_paths: dict[tuple[str, str], list[str] | None] = {}
    for cond in conditions:
        hits, vn, ve = shortest_paths_to_patients(H, cond["anchor"], max_hops)
        for pid, info in hits.items():
            all_paths[(cond["id"], pid)] = info["path"]
        total_vn += vn
        total_ve += ve
    # candidate set = patients with >=1 condition matched (the ONLY ones we emit)
    candidates = sorted({pid for (cid, pid), p in all_paths.items() if p is not None})
    results: list[RetrievalResult] = []
    for pid in candidates:                       # <-- NOT all patients, only candidates
        acc = new_patient_accumulator(cond_ids)
        for cond in conditions:
            cid = cond["id"]
            p = all_paths.get((cid, pid))
            if p is not None:
                cost = sum(edge_cost(H, p[i], p[i + 1], rho=rho) for i in range(len(p) - 1))
                acc[cid] = {"matched": True, "distance": len(p) - 1, "path": p,
                            "cost": cost, "reason": "MATCH"}
            else:
                acc[cid] = {"matched": False, "distance": None, "path": [],
                            "cost": 0.0, "reason": "NOT REACHABLE"}
        sub_nodes: set[str] = {pid}
        sub_edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]
        for cid in matched:
            path = acc[cid]["path"]
            for n in path:
                sub_nodes.add(n)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                for _, vv, k, edata in H.out_edges(u, keys=True, data=True):
                    if vv == v and (u, v, k) not in seen:
                        seen.add((u, v, k))
                        row = {"source": u, "target": v, "relation_type": k}
                        row.update(edata)
                        sub_edges.append(row)
                        break
        for e in sub_edges:
            e["_cost"] = edge_cost(H, e["source"], e["target"], rho=rho)
        union_cost = sum(e["_cost"] for e in sub_edges)
        score = score_generic(acc, m)
        results.append(RetrievalResult(
            patient_id=pid, algorithm="B", score=score,
            coverage_count=coverage_count(acc), coverage_ratio=coverage_ratio(acc, m),
            matched_conditions=matched, condition_paths={cid: acc[cid] for cid in cond_ids},
            nodes=sorted(sub_nodes), edges=sub_edges, graph_cost=union_cost,
            strict_match=coverage_ratio(acc, m) >= 1.0 - 1e-9,
            visited_nodes=total_vn, visited_edges=total_ve,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        ))
    return results


# ---------------------------------------------------------------------------
# External graph loader (read-only)
# ---------------------------------------------------------------------------
def load_external_reversed(root=EXT_ROOT):
    with open(os.path.join(root, "CURRENT"), encoding="utf-8") as f:
        rel = f.read().strip()
    with open(os.path.join(root, rel, "graph.pkl"), "rb") as f:
        g = pickle.load(f)
    Gx = nx.MultiDiGraph()
    for n, d in g.nodes(data=True):
        if d.get("type") in ("patient", "entity"):
            Gx.add_node(n, type=d["type"], **{k: v for k, v in d.items() if k != "type"})
    for u, v, k, d in g.edges(keys=True, data=True):
        if g.nodes[u].get("type") == "patient" and g.nodes[v].get("type") == "entity":
            attrs = dict(d)
            if d.get("relation_type") == "number" and d.get("raw_relation") is not None:
                try:
                    attrs["value"] = float(d["raw_relation"])
                except (TypeError, ValueError):
                    pass
            Gx.add_edge(v, u, key=k, **attrs)
    return Gx


EXT_CONDS = [
    {"id": "C1", "anchor": "entity:经典型骨肉瘤", "kind": "categorical"},
    {"id": "C2", "anchor": "entity:乳酸脱氢酶(LDH)", "kind": "numeric",
     "operator": ">", "value": 300, "unit": "U/L"},
    {"id": "C3", "anchor": "entity:碱性磷酸酶(ALP)", "kind": "numeric",
     "operator": ">", "value": 300, "unit": "U/L"},
    {"id": "C4", "anchor": "entity:年龄", "kind": "numeric",
     "operator": ">=", "value": 10, "unit": "岁"},
    {"id": "C5", "anchor": "entity:血沉", "kind": "numeric",
     "operator": ">", "value": 30, "unit": "mm/h"},
]


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------
def measure(fn, G, conds, repeats=20, **kw):
    t0 = time.perf_counter()
    res = None
    for _ in range(repeats):
        res = fn(G, conds, max_hops=4, strict=True, rho=1.3, **kw)
    wall = (time.perf_counter() - t0) * 1000 / repeats
    vn = max((r.visited_nodes for r in res), default=0)
    ve = max((r.visited_edges for r in res), default=0)
    nres = len(res)
    return vn, ve, nres, wall


def run_block(title, G, conds):
    print(f"\n=== {title} ===  ({G.number_of_nodes()} nodes / {G.number_of_edges()} edges)")
    print(f"{'strategy':<14}{'vis_nodes':<11}{'vis_edges':<11}{'#results':<10}{'wall_ms':<10}")
    strategies = [
        ("bfs", retrieve_bfs),
        ("pcst", retrieve_pcst),
        ("pcst_shared", retrieve_pcst_shared),
        ("steiner_shared", retrieve_steiner_shared),
        ("bfs+A", retrieve_bfs_prefiltered),
        ("B", retrieve_candidate_only),
    ]
    base_wall = None
    for name, fn in strategies:
        vn, ve, nr, wall = measure(fn, G, conds)
        if name == "bfs":
            base_wall = wall
        speedup = f"{base_wall/wall:.2f}x" if base_wall else "-"
        print(f"{name:<14}{vn:<11}{ve:<11}{nr:<10}{wall:<8.3f}  {speedup}")


def main():
    Gs = build_graph()
    run_block("SYNTHETIC graph (8 patients, multi-hop, NO dead-ends)",
              Gs, build_default_conditions())

    print("\n-- loading external graph (read-only) --")
    Gx = load_external_reversed()
    run_block("EXTERNAL graph (100 patients, 1-hop, NO dead-ends)", Gx, EXT_CONDS)

    Gd, dconds = build_deadend_graph()
    run_block("DEAD-END graph (10 patients + 200 sink dead-end branches)", Gd, dconds)

    print("\nNote: A (reverse-reachability prefilter) only prunes branches that CANNOT")
    print("reach any patient -> 0 gain on graphs where everything reaches a patient")
    print("(synthetic/external), big gain on dead-end-heavy graphs. B saves")
    print("result-construction (skip 0-coverage patients); a true SEARCH saving")
    print("beyond shared needs an index (the graph ships FAISS+SQLite for that).")


def build_deadend_graph():
    """A graph with many sink dead-end branches (not reaching any patient),
    to show variant A's pruning benefit."""
    G = nx.MultiDiGraph()
    for i in range(10):
        G.add_node(f"P{i:02d}", type="patient")
    G.add_node("HubA", type="intermediate")
    G.add_node("Anchor1", type="disease")
    G.add_node("Anchor2", type="indicator", value=None, unit=None)
    # Anchor1 -> HubA -> patients (reaches patients, 2-hop)
    G.add_edge("Anchor1", "HubA", key="link", relation_type="link")
    for i in range(10):
        G.add_edge("HubA", f"P{i:02d}", key="hub_patient", relation_type="hub_patient")
    # Anchor1 -> 100 sink dead-end branches (NOT reaching any patient)
    for j in range(100):
        d = f"DEAD_A_{j}"
        G.add_node(d, type="intermediate")
        G.add_edge("Anchor1", d, key="dead", relation_type="dead")
        # chain of 2 sinks each
        d2 = f"DEAD_A_{j}_2"
        G.add_node(d2, type="intermediate")
        G.add_edge(d, d2, key="dead", relation_type="dead")
    # Anchor2 -> patients direct (1-hop) + 50 more sinks
    for i in range(10):
        G.add_edge("Anchor2", f"P{i:02d}", key="num", relation_type="num",
                   value=100 + i, unit="u")
    for j in range(50):
        d = f"DEAD_B_{j}"
        G.add_node(d, type="intermediate")
        G.add_edge("Anchor2", d, key="dead", relation_type="dead")
    conds = [
        {"id": "C1", "anchor": "Anchor1", "kind": "categorical"},
        {"id": "C2", "anchor": "Anchor2", "kind": "numeric",
         "operator": ">", "value": 5, "unit": "u"},
    ]
    return G, conds


if __name__ == "__main__":
    main()
