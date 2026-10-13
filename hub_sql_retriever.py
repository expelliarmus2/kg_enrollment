"""SQL-offload adapter for retrieval on the local copy of the external graph.

Does NOT touch src/retrieval. Mirrors retrieve_bfs but splits storage:
  * high-fanout hub -> patient edges are MOVED to a SQLite table (indexed)
  * hub -> non-patient edges STAY in the graph
At a hub node the retriever does BOTH, in parallel:
  (1) a fixed SQL query for the hub's (offloaded) patients -- value-filtered when
      the hub is the anchor of a numeric condition
  (2) BFS continuation through the hub's kept non-patient out-edges
and merges the two hit sets.

Offload rule (per user): node patient-edge count > 20 AND edge points to patient
AND edge kind in {number, text (numeric values + categorical labels incl. diagnosis)}.

Run:
    python hub_sql_retriever.py
"""
from __future__ import annotations

import os
import pickle
import sqlite3
import time
from collections import deque
from typing import Any

import networkx as nx

from src.constraints import build_query_conditioned_graph
from src.graph_model import edge_cost
from src.scoring import (
    new_patient_accumulator, coverage_count, coverage_ratio, score_generic,
)
from src.retrieval.base import RetrievalResult
from src.retrieval.bfs import retrieve_bfs  # baseline for time comparison

LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "external_graph")
DB_PATH = os.path.join(LOCAL, "hub_patient_edges.sqlite")
HUB_THRESHOLD = 20  # patient-edge count > this => offload to SQL

CONDS = [
    {"id": "C1", "anchor": "entity:经典型骨肉瘤", "kind": "categorical"},
    {"id": "C2", "anchor": "entity:乳酸脱氢酶(LDH)", "kind": "numeric",
     "operator": ">", "value": 300, "unit": "U/L"},
    {"id": "C3", "anchor": "entity:碱性磷酸酶", "kind": "numeric",
     "operator": ">", "value": 300, "unit": "U/L"},
    {"id": "C4", "anchor": "entity:年龄", "kind": "numeric",
     "operator": ">=", "value": 10, "unit": "岁"},
    {"id": "C5", "anchor": "entity:血沉", "kind": "numeric",
     "operator": ">", "value": 30, "unit": "mm/h"},
]


# ---------------------------------------------------------------------------
# load + reverse
# ---------------------------------------------------------------------------
def load_local_graph() -> nx.MultiDiGraph:
    with open(os.path.join(LOCAL, "graph.pkl"), "rb") as f:
        return pickle.load(f)


def build_reversed_view(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """original patient -> entity  becomes  entity -> patient."""
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


# ---------------------------------------------------------------------------
# offload hubs -> SQLite
# ---------------------------------------------------------------------------
def identify_offload_hubs(Gx: nx.MultiDiGraph, threshold: int = HUB_THRESHOLD) -> set[str]:
    """Entities whose patient-edge count > threshold (edges point to patient)."""
    hubs: set[str] = set()
    for n, d in Gx.nodes(data=True):
        if d.get("type") != "entity":
            continue
        cnt = sum(1 for _, tgt, _, _ in Gx.out_edges(n, keys=True, data=True)
                  if Gx.nodes[tgt].get("type") == "patient")
        if cnt > threshold:
            hubs.add(n)
    return hubs


def build_sqlite(Gx: nx.MultiDiGraph, hubs: set[str], db_path: str = DB_PATH) -> sqlite3.Connection:
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE hub_patient_edges (
            hub TEXT, patient TEXT, kind TEXT,
            value REAL, unit TEXT, raw_relation TEXT, relation_type TEXT,
            chunk_id TEXT, valid_at TEXT, invalid_at TEXT)
    """)
    conn.execute("CREATE INDEX idx_hub ON hub_patient_edges(hub)")
    conn.execute("CREATE INDEX idx_hub_val ON hub_patient_edges(hub, value)")
    rows = []
    for hub in hubs:
        for _, tgt, k, ed in Gx.out_edges(hub, keys=True, data=True):
            if Gx.nodes[tgt].get("type") != "patient":
                continue  # only patient edges go to SQL; non-patient stay in graph
            rows.append((
                hub, tgt, ed.get("relation_type"),
                ed.get("value"), ed.get("unit"), ed.get("raw_relation"),
                ed.get("relation_type"), ed.get("chunk_id"),
                ed.get("valid_at"), ed.get("invalid_at"),
            ))
    conn.executemany("INSERT INTO hub_patient_edges VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return conn


def build_offloaded_graph(Gx: nx.MultiDiGraph, hubs: set[str]) -> nx.MultiDiGraph:
    """Copy of Gx with hubs' ->patient edges removed; hub nodes & non-patient edges kept."""
    Go = Gx.copy()
    to_remove = []
    for hub in hubs:
        for _, tgt, k, _ed in Go.out_edges(hub, keys=True, data=True):
            if Go.nodes[tgt].get("type") == "patient":
                to_remove.append((hub, tgt, k))
    for e in to_remove:
        Go.remove_edge(*e)
    return Go


# ---------------------------------------------------------------------------
# SQL-aware search (BFS + SQL at hubs, merged)
# ---------------------------------------------------------------------------
def sql_aware_search(H, anchor, max_hops, cond, hubs, conn):
    visited = {anchor}
    visited_edges = 0
    best: dict[str, dict[str, Any]] = {}
    is_num_anchor = cond.get("kind") == "numeric"
    op = cond.get("operator")
    val = cond.get("value")
    queue: deque = deque([(anchor, 0, [anchor])])
    while queue:
        node, dist, path = queue.popleft()
        # (1) SQL path: if node is an offload hub, query its (offloaded) patients
        if node in hubs:
            sql = "SELECT patient, value, unit, raw_relation, relation_type FROM hub_patient_edges WHERE hub=?"
            args: list = [node]
            # value-filter only when this hub IS the anchor of a numeric condition
            if node == anchor and is_num_anchor and op and val is not None:
                sql += f" AND value {op} ?"
                args.append(val)
            for p, value, unit, raw, rt in conn.execute(sql, args):
                nd = dist + 1
                edge_data = {"source": node, "target": p, "relation_type": rt,
                              "value": value, "unit": unit, "raw_relation": raw}
                if p not in best or nd < best[p]["distance"]:
                    best[p] = {"matched": True, "distance": nd, "path": path + [p],
                               "cost": 0.0, "edge_data": edge_data,
                               "reason": "MATCH(SQL)"}
        # (2) BFS path: expand kept (non-patient) out-edges, continue multi-hop
        if dist >= max_hops:
            continue
        for _, nbr, k, edata in H.out_edges(node, keys=True, data=True):
            visited_edges += 1
            if nbr in path:
                continue
            visited.add(nbr)
            nd = dist + 1
            edge_data = {"source": node, "target": nbr, "relation_type": k,
                         **{kk: vv for kk, vv in edata.items()}}
            if H.nodes[nbr].get("type") == "patient":
                if nbr not in best or nd < best[nbr]["distance"]:
                    best[nbr] = {"matched": True, "distance": nd, "path": path + [nbr],
                                 "cost": 0.0, "edge_data": edge_data,
                                 "reason": "MATCH(BFS)"}
            queue.append((nbr, nd, path + [nbr]))
    return best, len(visited), visited_edges


# ---------------------------------------------------------------------------
# SQL-aware retrieve (mirrors retrieve_bfs shape)
# ---------------------------------------------------------------------------
def retrieve_sql_aware(Go, conds, hubs, conn, *, max_hops=4, rho=0.0):
    t0 = time.perf_counter()
    H = build_query_conditioned_graph(Go, conds)   # filters non-offloaded numeric anchors (e.g. 血沉)
    patients = [n for n, d in H.nodes(data=True) if d.get("type") == "patient"]
    cond_ids = [c["id"] for c in conds]
    m = len(conds)
    total_vn = total_ve = 0
    cond_hits: dict[str, dict[str, dict[str, Any]]] = {}
    for cond in conds:
        hits, vn, ve = sql_aware_search(H, cond["anchor"], max_hops, cond, hubs, conn)
        cond_hits[cond["id"]] = hits
        total_vn += vn
        total_ve += ve
    results: list[RetrievalResult] = []
    for pid in patients:
        acc = new_patient_accumulator(cond_ids)
        for cid, hits in cond_hits.items():
            if pid in hits:
                acc[cid] = hits[pid]
        for cid in cond_ids:
            if acc[cid]["matched"]:
                path = acc[cid]["path"]
                cost = sum(edge_cost(H, path[i], path[i + 1], rho=rho)
                           for i in range(len(path) - 1))
                acc[cid]["cost"] = cost
        sub_nodes: set[str] = {pid}
        sub_edges: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        matched = [cid for cid in cond_ids if acc[cid]["matched"]]
        for cid in matched:
            ed = acc[cid].get("edge_data", {})
            ek = (ed.get("source"), ed.get("target"), ed.get("relation_type"))
            if ek in seen:
                continue
            seen.add(ek)
            sub_edges.append(ed)
            for n in acc[cid]["path"]:
                sub_nodes.add(n)
        union_cost = sum(edge_cost(H, e.get("source"), e.get("target"), rho=rho) for e in sub_edges)
        results.append(RetrievalResult(
            patient_id=pid, algorithm="sql_aware", score=score_generic(acc, m),
            coverage_count=coverage_count(acc), coverage_ratio=coverage_ratio(acc, m),
            matched_conditions=matched, condition_paths={cid: acc[cid] for cid in cond_ids},
            nodes=sorted(sub_nodes), edges=sub_edges, graph_cost=union_cost,
            strict_match=coverage_ratio(acc, m) >= 1.0 - 1e-9,
            visited_nodes=total_vn, visited_edges=total_ve,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        ))
    return results


# ---------------------------------------------------------------------------
def _short(n: str) -> str:
    return n.replace("patient:", "").replace("entity:", "")


def main():
    print("== B: local graph copy ==")
    g = load_local_graph()
    print(f"  {LOCAL}/graph.pkl  ->  {g.number_of_nodes()} nodes / {g.number_of_edges()} edges")

    print("\n== reverse view (entity -> patient) ==")
    Gx = build_reversed_view(g)
    print(f"  {Gx.number_of_nodes()} nodes / {Gx.number_of_edges()} edges")

    print(f"\n== identify offload hubs (patient-edges > {HUB_THRESHOLD}) ==")
    hubs = identify_offload_hubs(Gx)
    print(f"  {len(hubs)} hubs:")
    for h in sorted(hubs, key=lambda n: -Gx.out_degree(n)):
        cnt = sum(1 for _, t, *_ in Gx.out_edges(h, keys=True, data=True)
                  if Gx.nodes[t].get("type") == "patient")
        print(f"    {_short(h):<20} {cnt} patient-edges")

    print("\n== build SQLite (hub -> patient edges offloaded) ==")
    conn = build_sqlite(Gx, hubs)
    n = conn.execute("SELECT COUNT(*) FROM hub_patient_edges").fetchone()[0]
    print(f"  {DB_PATH}")
    print(f"  rows in hub_patient_edges: {n}  (offloaded from {len(hubs)} hubs)")

    print("\n== offloaded graph (hub -> patient edges removed; non-patient edges kept) ==")
    Go = build_offloaded_graph(Gx, hubs)
    print(f"  before: {Gx.number_of_nodes()} nodes / {Gx.number_of_edges()} edges")
    print(f"  after : {Go.number_of_nodes()} nodes / {Go.number_of_edges()} edges")

    print("\n== per-condition path (SQL vs BFS) ==")
    for c in CONDS:
        route = "SQL" if c["anchor"] in hubs else "BFS(graph)"
        print(f"  {c['id']} {_short(c['anchor']):<16} {c['kind']:<11} -> {route}")

    print("\n== SQL-aware retrieve (BFS + SQL at hubs, merged) ==")
    t0 = time.perf_counter()
    res = retrieve_sql_aware(Go, CONDS, hubs, conn, max_hops=4, rho=1.3)
    wall = (time.perf_counter() - t0) * 1000
    from collections import Counter
    cov = Counter(r.coverage_count for r in res)
    cohort = sorted(_short(r.patient_id) for r in res if r.coverage_count == len(CONDS))
    print(f"  coverage dist: {dict(sorted(cov.items(), reverse=True))}")
    print(f"  STRICT cohort (5/5): {cohort if cohort else '(empty)'}")
    print(f"  visited_nodes={max(r.visited_nodes for r in res)}  visited_edges={max(r.visited_edges for r in res)}")
    print(f"  wall: {wall:.3f} ms")

    print("\n== baseline retrieve_bfs on FULL reversed graph (for time comparison) ==")
    t0 = time.perf_counter()
    base = retrieve_bfs(Gx, CONDS, max_hops=4, strict=True, rho=1.3)
    base_wall = (time.perf_counter() - t0) * 1000
    bcov = Counter(r.coverage_count for r in base)
    print(f"  coverage dist: {dict(sorted(bcov.items(), reverse=True))}")
    print(f"  visited_nodes={max(r.visited_nodes for r in base)}  visited_edges={max(r.visited_edges for r in base)}")
    print(f"  wall: {base_wall:.3f} ms")
    print(f"\n  SQL-aware vs baseline: coverage identical? {dict(cov) == dict(bcov)}")
    print(f"  speedup: {base_wall/wall:.2f}x  (graph shrank {Gx.number_of_edges()}->{Go.number_of_edges()} edges)")

    conn.close()


if __name__ == "__main__":
    main()
