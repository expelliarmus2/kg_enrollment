"""Scale benchmark: does SQL-offload speedup grow with graph size?

Generates reversed (entity -> patient) graphs at increasing patient counts,
mirroring the external graph structure (hub labs/vitals fan out to ALL patients;
a small disease entity stays non-hub). At each scale compares:
  * baseline BFS search  vs  SQL-offload search (per-condition, isolated)
  * full retrieve wall   (includes per-patient result construction)
  * visited_nodes (search structural cost)
Then runs a 3-condition cohort retrieval (disease + 2 lab thresholds) and lists
the recalled patients + coverage dist + times.

Does NOT touch src/. Reuses offload helpers from hub_sql_retriever.
"""
from __future__ import annotations

import os
import random
import sqlite3
import time
from collections import Counter

import networkx as nx

from src.constraints import build_query_conditioned_graph
from src.retrieval.bfs import retrieve_bfs, _bfs_from_anchor
from hub_sql_retriever import (
    identify_offload_hubs, build_sqlite, build_offloaded_graph,
    retrieve_sql_aware, sql_aware_search,
)

# 8 hub entities -> fan out to ALL patients (labs/vitals/demographics)
HUB_ENTITIES = ["LDH", "ALP", "HGB", "SBP", "Temp", "WBC", "性别", "年龄"]
DISEASE = "骨肉瘤"  # small fan-out (15 patients) -> stays non-hub -> BFS path

CONDS = [
    {"id": "C1", "anchor": "entity:" + DISEASE, "kind": "categorical"},
    {"id": "C2", "anchor": "entity:LDH", "kind": "numeric",
     "operator": ">", "value": 250, "unit": "U/L"},
    {"id": "C3", "anchor": "entity:HGB", "kind": "numeric",
     "operator": "<", "value": 110, "unit": "g/L"},
]


def gen_reversed(n: int, seed: int = 42) -> nx.MultiDiGraph:
    rnd = random.Random(seed)
    G = nx.MultiDiGraph()
    for h in HUB_ENTITIES:
        G.add_node("entity:" + h, type="entity")
    G.add_node("entity:" + DISEASE, type="entity")
    patients = [f"P{i}" for i in range(n)]
    for p in patients:
        G.add_node(p, type="patient")
    for h in HUB_ENTITIES:
        is_num = h not in ("性别",)
        ent = "entity:" + h
        for i, p in enumerate(patients):
            v = round(rnd.gauss(200, 60), 2)
            if is_num:
                G.add_edge(ent, p, key=f"{h}-{i}", relation_type="number",
                           value=v, unit="U/L", raw_relation=str(v))
            else:
                G.add_edge(ent, p, key=f"{h}-{i}", relation_type="text",
                           value=None, unit=None, raw_relation="男")
    # disease: 15 patients fixed (non-hub BFS path)
    for i in rnd.sample(range(n), min(15, n)):
        G.add_edge("entity:" + DISEASE, f"P{i}", key=f"D-{i}",
                   relation_type="text", value=None, unit=None, raw_relation="确诊")
    return G


def run_scale(n: int) -> dict:
    G = gen_reversed(n)
    n_edges = G.number_of_edges()
    hubs = identify_offload_hubs(G, 20)
    db = f"/tmp/scale_hub_{n}.sqlite"
    conn = build_sqlite(G, hubs, db)
    sql_rows = conn.execute("SELECT COUNT(*) FROM hub_patient_edges").fetchone()[0]
    Go = build_offloaded_graph(G, hubs)
    off_edges = Go.number_of_edges()
    H_full = build_query_conditioned_graph(G, CONDS)
    H_off = build_query_conditioned_graph(Go, CONDS)

    # --- search-only (per-condition sum), isolated from result construction ---
    t = time.perf_counter()
    for c in CONDS:
        _bfs_from_anchor(H_full, c["anchor"], 4)
    search_base = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    for c in CONDS:
        sql_aware_search(H_off, c["anchor"], 4, c, hubs, conn)
    search_sql = (time.perf_counter() - t) * 1000

    # --- full retrieve (includes building n RetrievalResults) ---
    t = time.perf_counter()
    base = retrieve_bfs(G, CONDS, max_hops=4, strict=True, rho=0.0)
    wall_base = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    res = retrieve_sql_aware(Go, CONDS, hubs, conn, max_hops=4, rho=0.0)
    wall_sql = (time.perf_counter() - t) * 1000

    vn_base = max(r.visited_nodes for r in base)
    vn_sql = max(r.visited_nodes for r in res)
    cov = Counter(r.coverage_count for r in res)
    cohort = [r.patient_id for r in res if r.coverage_count == 3]
    conn.close()
    os.remove(db)
    return dict(n=n, edges=n_edges, n_hubs=len(hubs), sql_rows=sql_rows,
                off_edges=off_edges, search_base=search_base, search_sql=search_sql,
                wall_base=wall_base, wall_sql=wall_sql, vn_base=vn_base, vn_sql=vn_sql,
                cov=dict(sorted(cov.items(), reverse=True)), cohort=cohort)


def main():
    print("== Scaling: SQL-offload vs baseline BFS =="
          " (3 conds: 骨肉瘤 cat + LDH>250 + HGB<110)\n")
    print(f"{'n_pat':<8}{'edges':<9}{'hubs':<6}{'SQLrows':<9}"
          f"{'srch_base':<10}{'srch_sql':<10}{'srch_x':<7}"
          f"{'wall_base':<10}{'wall_sql':<9}{'wall_x':<7}"
          f"{'vis_base':<9}{'vis_sql':<8}")
    print("-" * 110)
    rows = []
    for n in [1000, 10000, 50000, 100000]:
        r = run_scale(n)
        rows.append(r)
        sx = r["search_base"] / r["search_sql"] if r["search_sql"] else 0
        wx = r["wall_base"] / r["wall_sql"] if r["wall_sql"] else 0
        print(f"{n:<8}{r['edges']:<9}{r['n_hubs']:<6}{r['sql_rows']:<9}"
              f"{r['search_base']:<10.2f}{r['search_sql']:<10.2f}{sx:<7.2f}"
              f"{r['wall_base']:<10.2f}{r['wall_sql']:<9.2f}{wx:<7.2f}"
              f"{r['vn_base']:<9}{r['vn_sql']:<8}")

    # cohort retrieval demo on the largest scale
    big = rows[-1]
    print(f"\n== Cohort retrieval (inclusion: 骨肉瘤 AND LDH>250 AND HGB<110) ==")
    print(f"  scale: {big['n']} patients / {big['edges']} edges")
    print(f"  coverage dist (sql-aware): {big['cov']}")
    print(f"  STRICT cohort (3/3) recalled {len(big['cohort'])} patients: "
          f"{big['cohort'][:20]}{' ...' if len(big['cohort']) > 20 else ''}")
    print(f"  times: search base={big['search_base']:.2f}ms sql={big['search_sql']:.2f}ms "
          f"| full base={big['wall_base']:.2f}ms sql={big['wall_sql']:.2f}ms")
    print(f"  visited: base={big['vn_base']} sql={big['vn_sql']}")

    # trend
    print("\n== Speedup trend (search phase) ==")
    for r in rows:
        sx = r["search_base"] / r["search_sql"] if r["search_sql"] else 0
        print(f"  n={r['n']:<8} search_base={r['search_base']:<8.2f}ms "
              f"search_sql={r['search_sql']:<8.2f}ms  speedup={sx:<6.1f}x")
    print("\nNote: baseline search visits ALL hub fan-out (O(n) per hub anchor);")
    print("SQL search = indexed range scan (O(log n + matches)). Speedup ~ n/(log n + matches) ->")
    print("grows with n. At 12M patients (can't fit in-memory NetworkX), hub fan-out ~12M,")
    print("SQL index query stays O(log n + matches) -> speedup becomes orders of magnitude.")


if __name__ == "__main__":
    main()
