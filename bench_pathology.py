"""Verify: is the 36x speedup from SQL, or from baseline's O(n^2) subgraph re-scan?
Fixed baseline = BFS that stores edge_data along the path, so subgraph build is O(1)
per condition (no H.out_edges re-scan). If fixed ≈ SQL wall, the speedup was the
code pathology, not the SQL index.
"""
from __future__ import annotations
import time
from collections import deque
from typing import Any
import networkx as nx

from src.constraints import build_query_conditioned_graph
from src.scoring import new_patient_accumulator, coverage_count, coverage_ratio, score_generic
from src.retrieval.base import RetrievalResult
from src.retrieval.bfs import retrieve_bfs
from scale_bench import gen_reversed, CONDS
from hub_sql_retriever import identify_offload_hubs, build_sqlite, build_offloaded_graph, retrieve_sql_aware


def _bfs_fixed(H, anchor, max_hops):
    """BFS that records edge_data per hop -> no out_edges re-scan later."""
    visited = {anchor}; visited_edges = 0; best = {}
    queue: deque = deque([(anchor, 0, [anchor], [])])
    while queue:
        node, dist, path, edges = queue.popleft()
        if dist >= max_hops:
            continue
        for _, nbr, k, edata in H.out_edges(node, keys=True, data=True):
            visited_edges += 1
            if nbr in path:
                continue
            ed = {"source": node, "target": nbr, "relation_type": k, **edata}
            new_path = path + [nbr]; new_edges = edges + [ed]
            visited.add(nbr); nd = dist + 1
            if H.nodes[nbr].get("type") == "patient":
                if nbr not in best or nd < best[nbr]["distance"]:
                    best[nbr] = {"matched": True, "distance": nd, "path": new_path,
                                 "edges": new_edges, "cost": 0.0, "reason": "MATCH"}
            queue.append((nbr, nd, new_path, new_edges))
    return best, len(visited), visited_edges


def retrieve_bfs_fixed(G, conditions, *, max_hops=4, strict=True, rho=0.0):
    t0 = time.perf_counter()
    H = build_query_conditioned_graph(G, conditions)
    patients = [n for n, d in H.nodes(data=True) if d.get("type") == "patient"]
    cond_ids = [c["id"] for c in conditions]; m = len(conditions)
    total_vn = total_ve = 0
    cond_hits = {}
    for cond in conditions:
        hits, vn, ve = _bfs_fixed(H, cond["anchor"], max_hops)
        cond_hits[cond["id"]] = hits; total_vn += vn; total_ve += ve
    results = []
    for pid in patients:
        acc = new_patient_accumulator(cond_ids)
        for cid, hits in cond_hits.items():
            if pid in hits:
                acc[cid] = hits[pid]
        sub_nodes = {pid}; sub_edges = []; seen = set()
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]
        for cid in matched:
            for ed in acc[cid].get("edges", []):     # stored edges, NO out_edges scan
                ek = (ed["source"], ed["target"], ed["relation_type"])
                if ek in seen: continue
                seen.add(ek); sub_edges.append(ed)
            for n in acc[cid]["path"]:
                sub_nodes.add(n)
        results.append(RetrievalResult(
            patient_id=pid, algorithm="bfs_fixed", score=score_generic(acc, m),
            coverage_count=coverage_count(acc), coverage_ratio=coverage_ratio(acc, m),
            matched_conditions=matched, condition_paths={cid: acc[cid] for cid in cond_ids},
            nodes=sorted(sub_nodes), edges=sub_edges, graph_cost=0.0,
            strict_match=coverage_ratio(acc, m) >= 1.0 - 1e-9,
            visited_nodes=total_vn, visited_edges=total_ve,
            elapsed_ms=(time.perf_counter() - t0) * 1000))
    return results


def main():
    n = 10000
    G = gen_reversed(n)
    hubs = identify_offload_hubs(G, 20)
    conn = build_sqlite(G, hubs, f"/tmp/p_{n}.sqlite")
    Go = build_offloaded_graph(G, hubs)

    t = time.perf_counter(); base = retrieve_bfs(G, CONDS, max_hops=4, strict=True, rho=0.0); wb = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); fix = retrieve_bfs_fixed(G, CONDS, max_hops=4, strict=True, rho=0.0); wf = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); sql = retrieve_sql_aware(Go, CONDS, hubs, conn, max_hops=4, rho=0.0); ws = (time.perf_counter() - t) * 1000
    conn.close(); import os; os.remove(f"/tmp/p_{n}.sqlite")

    cb = max(r.visited_nodes for r in base); cf = max(r.visited_nodes for r in fix); cs = max(r.visited_nodes for r in sql)
    covb = [r.coverage_count for r in base]; covf = [r.coverage_count for r in fix]; covs = [r.coverage_count for r in sql]
    print(f"n={n}  edges={G.number_of_edges()}  hubs={len(hubs)}")
    print(f"{'strategy':<16}{'wall_ms':<10}{'vis_nodes':<10}{'cov==base?'}")
    print(f"{'baseline bfs':<16}{wb:<10.1f}{cb:<10}{True}")
    print(f"{'bfs_fixed':<16}{wf:<10.1f}{cf:<10}{covf==covb}")
    print(f"{'sql_offload':<16}{ws:<10.1f}{cs:<10}{covs==covb}")
    print(f"\nspeedup vs baseline:  fixed={wb/wf:.2f}x   sql={wb/ws:.2f}x")
    print(f"fixed vs sql wall ratio: {wf/ws:.2f}x  (close to 1 => speedup was the code fix, not SQL index)")


if __name__ == "__main__":
    main()
