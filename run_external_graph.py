"""Run our patient-centered retrieval on the EXTERNAL computable-graph.

Read-only: pickle.load the graph.pkl at /data/qc/time_kg/kg/build_run/graph_v2,
build an in-memory REVERSED MultiDiGraph (entity -> patient) so our anchor->patient
BFS works, then run retrieve() for bfs / pcst / steiner.

Constraints honored:
  * NO writes to the external path
  * NO graph files copied into this project (in-memory transform only)
  * skips evaluate() (its ground truth is the synthetic P001-P008, irrelevant here);
    reports coverage / cohort / subgraph size directly from RetrievalResult

Run:
    python run_external_graph.py
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Any

import networkx as nx

from src.retrieval import retrieve

EXT_ROOT = "/data/qc/time_kg/kg/build_run/graph_v2"

# Real entity anchors verified present in the external graph.
CONDITIONS: list[dict[str, Any]] = [
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


def _resolve_graph_pkl(root: str) -> str:
    """Resolve graph.pkl via the CURRENT generation pointer (read-only)."""
    with open(os.path.join(root, "CURRENT"), encoding="utf-8") as f:
        rel = f.read().strip()
    pkl = os.path.join(root, rel, "graph.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(pkl)
    return pkl


def load_external_graph(root: str = EXT_ROOT) -> nx.MultiDiGraph:
    """pickle.load the external graph.pkl (pure networkx, no time_kg import needed)."""
    pkl = _resolve_graph_pkl(root)
    with open(pkl, "rb") as f:
        return pickle.load(f)


def build_reversed_view(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """In-memory reversed view: original `patient -> entity` becomes `entity -> patient`.

    The external graph has patient in-degree 0 (patients are pure sources), so
    forward BFS from an entity anchor never reaches a patient. Reversing the
    patient<->entity edges makes our anchor->...->patient retrieval applicable.
    Parallel measurements (same entity, multiple values over time) are preserved
    as distinct edges via the original integer key.
    """
    Gx = nx.MultiDiGraph()
    for n, d in g.nodes(data=True):
        t = d.get("type")
        if t in ("patient", "entity"):
            Gx.add_node(n, type=t, **{k: v for k, v in d.items() if k != "type"})

    for u, v, k, d in g.edges(keys=True, data=True):
        if g.nodes[u].get("type") != "patient" or g.nodes[v].get("type") != "entity":
            continue
        rt = d.get("relation_type")
        raw = d.get("raw_relation")
        val: float | None = None
        if rt == "number" and raw is not None:
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = None
        attrs = dict(d)
        if not attrs.get("relation_type"):
            attrs["relation_type"] = rt or "rel"
        if val is not None:
            attrs["value"] = val
        Gx.add_edge(v, u, key=k, **attrs)
    return Gx


def _short_pid(n: str) -> str:
    return n.replace("patient:", "")


def verify_anchors(Gx: nx.MultiDiGraph, conds: list[dict[str, Any]]) -> None:
    print("  condition anchors:")
    for c in conds:
        a = c["anchor"]
        present = a in Gx
        deg = Gx.degree(a) if present else 0
        print(f"    {'OK ' if present else 'MISS'}  {c['id']} {a}  ({c['kind']}"
              + (f" {c['operator']}{c['value']} {c.get('unit','')}" if c["kind"] == "numeric" else "")
              + f")  out-deg={deg}")


def report(algo: str, results: list, m: int) -> None:
    from collections import Counter
    cov_dist = Counter(r.coverage_count for r in results)
    strict_cohort = sorted(_short_pid(r.patient_id) for r in results if r.coverage_count == m)
    # surface matched patients first: sort by coverage desc, then score desc.
    # (on this 1-hop graph with rho=1.3 the graph-cost term can make partial
    #  coverage score negative, so pure score-sort would hide matched patients)
    top = sorted(results, key=lambda r: (r.coverage_count, r.score), reverse=True)[:12]
    vn = max((r.visited_nodes for r in results), default=0)
    ve = max((r.visited_edges for r in results), default=0)
    el = max((r.elapsed_ms for r in results), default=0.0)
    print(f"\n--- {algo.upper()} ---")
    print(f"  patients: {len(results)}  coverage dist: {dict(sorted(cov_dist.items(), reverse=True))}")
    print(f"  STRICT cohort ({m}/{m}): {strict_cohort if strict_cohort else '(empty)'}")
    print(f"  search totals: visited_nodes={vn} visited_edges={ve} elapsed_ms={el:.3f}")
    print(f"  top-12 by (coverage, score):")
    print(f"    {'patient':<10}{'cov':<6}{'score':<9}{'strict':<8}{'sub_n':<7}{'sub_e':<7}{'gcost':<7}")
    for r in top:
        print(f"    {_short_pid(r.patient_id):<10}{r.coverage_count}/{m}{'':<3}"
              f"{r.score:<9.2f}{'Y' if r.strict_match else '-':<8}"
              f"{len(r.nodes):<7}{len(r.edges):<7}{r.graph_cost:<7.2f}")


def main() -> None:
    print("== Load external computable-graph (read-only) ==")
    t0 = time.perf_counter()
    g = load_external_graph()
    print(f"  source: {EXT_ROOT}")
    print(f"  loaded: {g.number_of_nodes()} nodes / {g.number_of_edges()} edges"
          f"  schema={g.graph.get('schema_version')}  ({(time.perf_counter()-t0)*1000:.1f} ms)")

    print("== Build reversed in-memory view (entity -> patient) ==")
    Gx = build_reversed_view(g)
    n_pat = sum(1 for _, d in Gx.nodes(data=True) if d.get("type") == "patient")
    n_ent = sum(1 for _, d in Gx.nodes(data=True) if d.get("type") == "entity")
    print(f"  reversed: {Gx.number_of_nodes()} nodes ({n_pat} patient / {n_ent} entity)"
          f" / {Gx.number_of_edges()} edges")

    conds = CONDITIONS
    m = len(conds)
    verify_anchors(Gx, conds)
    missing = [c["anchor"] for c in conds if c["anchor"] not in Gx]
    if missing:
        print(f"  WARNING missing anchors {missing} -- aborting (edit CONDITIONS).")
        return

    print("\n== Run retrieval (max_hops=4, rho=1.3, condition_prize=10) ==")
    for algo in ["bfs", "pcst", "steiner"]:
        t = time.perf_counter()
        results = retrieve(
            Gx, conds,
            algorithm=algo, max_hops=4, strict=True,
            rho=1.3, condition_prize=10.0,
        )
        # stamp elapsed with this run's wall time too (retrieval sets its own as well)
        report(algo, results, m)
        print(f"  (wall incl. report: {(time.perf_counter()-t)*1000:.1f} ms)")

    print("\nNote: no ground-truth file for this graph -> precision/recall not reported.")
    print("Cohort = patients whose coverage_count == 5/5 (all conditions reachable).")


if __name__ == "__main__":
    main()
