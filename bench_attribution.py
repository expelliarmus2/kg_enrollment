"""Attribute graph preparation and SQLite index costs on the scale fixture.

Examples (run from project root):
  python bench_attribution.py --sizes 1000 10000 --output attribution.json
  python bench_attribution.py --sizes 50000 100000 --repeats 10 --warmup 3
  python bench_attribution.py --sizes 10000 --include-original

Each size runs in a fresh process. SQLite uses a unique temporary directory.
Warm-cache benchmark; GC stays enabled, with collection outside each run.
No existing graph/database is overwritten. Only --output persists a report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gc
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import networkx as nx
import bench_pathology as fixed
import hub_sql_retriever as hybrid
from scale_bench import CONDS, gen_reversed
from src.constraints import build_query_conditioned_graph
from src.retrieval import bfs


HINTS = {
    "scan": "NOT INDEXED",
    "hub": "INDEXED BY idx_hub",
    "index": "INDEXED BY idx_hub_val",
}
PHASES = ("wall", "copy", "filter", "prepare", "search", "result_other")


class QueryConnection:
    """Force only the hub lookup's access path; keep fields and logic identical."""

    def __init__(self, conn, mode):
        self.conn = conn
        self.hint = HINTS[mode]

    def execute(self, sql, args=()):
        sql = sql.replace("FROM hub_patient_edges WHERE",
                          f"FROM hub_patient_edges {self.hint} WHERE")
        return self.conn.execute(sql, args)


def summary(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def timed_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


def signature(results):
    # Compare identities, matched condition IDs and distances, not just counts.
    return {
        r.patient_id: {
            cid: (c["matched"], c["distance"] if c["matched"] else None)
            for cid, c in r.condition_paths.items()
        }
        for r in results
    }


def profile_retrieval(module, retrieve, search_name, graph, **kwargs):
    """Instrument real functions; remainder includes scoring/subgraph construction."""
    times = {name: 0.0 for name in PHASES}
    real_search = getattr(module, search_name)
    real_copy = graph.copy

    def copy(*args, **kw):
        value, elapsed = timed_call(lambda: real_copy(*args, **kw))
        times["copy"] += elapsed
        return value

    def prepare(*args, **kw):
        value, elapsed = timed_call(lambda: build_query_conditioned_graph(*args, **kw))
        times["prepare"] += elapsed
        return value

    def search(*args, **kw):
        value, elapsed = timed_call(lambda: real_search(*args, **kw))
        times["search"] += elapsed
        return value

    # Patch overhead is small but nonzero. Mock installation is outside timing.
    with ExitStack() as stack:
        stack.enter_context(patch.object(graph, "copy", copy))
        stack.enter_context(patch.object(module, "build_query_conditioned_graph", prepare))
        stack.enter_context(patch.object(module, search_name, search))
        results, times["wall"] = timed_call(lambda: retrieve(graph, CONDS, **kwargs))
    times["filter"] = times["prepare"] - times["copy"]
    times["result_other"] = times["wall"] - times["prepare"] - times["search"]
    return results, times


def sql_queries(conn, mode):
    # Same projection as sql_aware_search, including edge evidence fields.
    result = {}
    for cond in CONDS:
        if cond["kind"] != "numeric":
            continue
        sql = (
            "SELECT patient, value, unit, raw_relation, relation_type "
            f"FROM hub_patient_edges {HINTS[mode]} "
            f"WHERE hub=? AND value {cond['operator']} ?"
        )
        result[cond["id"]] = conn.execute(sql, (cond["anchor"], cond["value"])).fetchall()
    return result


def run_size(args, n):
    print(f"n={n}: preparing fixture", file=sys.stderr, flush=True)
    graph, generate_ms = timed_call(lambda: gen_reversed(n, seed=args.seed))
    hubs, identify_ms = timed_call(lambda: hybrid.identify_offload_hubs(graph))
    reference = None
    raw = {}
    query_raw = {mode: [] for mode in HINTS}
    plans = {}
    with tempfile.TemporaryDirectory(prefix="kg_attribution_") as temp:
        db = str(Path(temp) / "hub.sqlite")
        conn, build_ms = timed_call(lambda: hybrid.build_sqlite(graph, sorted(hubs), db))
        try:
            off, offload_ms = timed_call(lambda: hybrid.build_offloaded_graph(graph, hubs))
            conn.execute("PRAGMA query_only=ON")
            rows = conn.execute("SELECT COUNT(*) FROM hub_patient_edges").fetchone()[0]
            for mode, hint in HINTS.items():
                plans[mode] = {}
                for cond in CONDS:
                    if cond["kind"] == "numeric":
                        sql = ("EXPLAIN QUERY PLAN SELECT patient, value, unit, raw_relation, relation_type "
                               f"FROM hub_patient_edges {hint} WHERE hub=? AND value {cond['operator']} ?")
                        details = [r[3] for r in conn.execute(sql, (cond["anchor"], cond["value"]))]
                        expected = {"scan": "SCAN", "hub": "idx_hub", "index": "idx_hub_val"}[mode]
                        if not any(expected in d for d in details):
                            raise AssertionError(f"Unexpected query plan: {details}")
                        plans[mode][cond["id"]] = details

            cases = {
                "bfs_fixed": lambda: profile_retrieval(
                    fixed, fixed.retrieve_bfs_fixed, "_bfs_fixed", graph, max_hops=4, rho=0.0),
            }
            if args.include_original:
                cases["bfs_original"] = lambda: profile_retrieval(
                    bfs, bfs.retrieve_bfs, "_bfs_from_anchor", graph, max_hops=4, rho=0.0)
            for mode in HINTS:
                adapter = QueryConnection(conn, mode)
                cases[f"sql_{mode}"] = lambda adapter=adapter: profile_retrieval(
                    hybrid, hybrid.retrieve_sql_aware, "sql_aware_search", off,
                    hubs=hubs, conn=adapter, max_hops=4, rho=0.0)
            raw = {name: [] for name in cases}
            rng = random.Random(args.seed)
            expected_rows = None
            for iteration in range(args.warmup + args.repeats):
                measured = iteration >= args.warmup
                print(f"n={n}: {'sample' if measured else 'warmup'} "
                      f"{iteration - args.warmup + 1 if measured else iteration + 1}",
                      file=sys.stderr, flush=True)
                order = list(cases)
                rng.shuffle(order)
                for name in order:
                    gc.collect()  # outside measured interval; GC remains enabled inside
                    results, times = cases[name]()
                    current = signature(results)
                    if reference is None:
                        reference = current
                    if current != reference:
                        raise AssertionError(f"Patient/condition/distance mismatch: {name}")
                    if measured:
                        raw[name].append(times)
                    del results, current

                # Independent SQL ablation. Fetch all rows before stopping the timer.
                modes = list(HINTS)
                rng.shuffle(modes)
                for mode in modes:
                    gc.collect()
                    result, elapsed = timed_call(lambda: sql_queries(conn, mode))
                    counters = {cid: Counter(values) for cid, values in result.items()}
                    if expected_rows is None:
                        expected_rows = counters
                    if counters != expected_rows:
                        raise AssertionError(f"SQL row multiset mismatch: {mode}")
                    if measured:
                        query_raw[mode].append(elapsed)
                    del result, counters

            stats = {name: {phase: summary([r[phase] for r in runs]) for phase in PHASES}
                     for name, runs in raw.items()}

            def delta(a, b, phase):
                # Pair samples by round. Differences can legitimately be negative.
                return summary([x[phase] - y[phase] for x, y in zip(raw[a], raw[b])])

            return {
                "patients": n, "nodes": graph.number_of_nodes(),
                "edges_full": graph.number_of_edges(), "edges_offloaded_graph": off.number_of_edges(),
                "hubs": len(hubs), "sql_table_rows": rows,
                "sql_returned_rows": {cid: sum(c.values()) for cid, c in expected_rows.items()},
                "setup_ms": {"generate": generate_ms, "identify_hubs": identify_ms,
                             "build_db_and_indexes": build_ms, "build_offloaded_graph": offload_ms},
                "stats": stats, "raw_samples_ms": raw,
                "sql_query_stats": {m: summary(v) for m, v in query_raw.items()},
                "sql_query_samples_ms": query_raw, "query_plans": plans,
                "savings_ms": {
                    "copy_full_minus_off": delta("bfs_fixed", "sql_index", "copy"),
                    "filter_full_minus_off": delta("bfs_fixed", "sql_index", "filter"),
                    "prepare_full_minus_off": delta("bfs_fixed", "sql_index", "prepare"),
                    "sql_scan_minus_index_search": delta("sql_scan", "sql_index", "search"),
                    "sql_hub_minus_composite_search": delta("sql_hub", "sql_index", "search"),
                },
                "correctness": "all patients, condition matches/distances and SQL row multisets agree",
            }
        finally:
            conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-original", action="store_true", help="Include slow original BFS with edge re-scans")
    parser.add_argument("--output", type=Path, help="Save metadata, plans and all timing samples as JSON")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or any(n <= 20 for n in args.sizes):
        parser.error("sizes must exceed 20, repeats >= 1, warmup >= 0")
    return args


def main():
    args = parse_args()
    if args.worker is not None:
        print(json.dumps(run_size(args, args.worker), ensure_ascii=True))
        return
    report = {
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "networkx": nx.__version__, "sqlite": sqlite3.sqlite_version},
        "settings": {"repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
                     "include_original": args.include_original, "gc_enabled": True,
                     "cache": "warm; OS cache not flushed", "p95": "nearest rank"},
        "notes": [
            "prepare = copy + filter; filter includes enumeration/deletion and small wrapper overhead",
            "wall = prepare + search + result_other per sample; medians need not add up",
            "Search compares prefiltered BFS with SQL that filters during lookup",
            "Index benefit is scan vs index within the SAME SQL pipeline, not BFS vs SQL",
            "Index and graph preparation ablations are different counterfactuals; do not add as total attribution",
            "Offloaded graph still contains every patient node; only edges are removed",
            "bfs_fixed omits cost computation; correctness checks exclude scores/subgraph costs",
            "One-hop synthetic fixture only; this does not establish general multihop equivalence",
            "Setup and sample teardown are excluded from online timings; no peak RAM measurement",
        ],
        "sizes": [],
    }
    for n in args.sizes:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(n),
                   "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--seed", str(args.seed)]
        if args.include_original:
            command.append("--include-original")
        child = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        result = json.loads(child.stdout)
        report["sizes"].append(result)
        print(f"\nn={n} edges={result['edges_full']} -> {result['edges_offloaded_graph']}", flush=True)
        print(f"{'strategy':<16} {'wall med':>10} {'wall P95':>10} {'copy':>10} {'filter':>10} {'search':>10} {'result+other':>13}")
        for name, stats in result["stats"].items():
            values = [stats['wall']['median_ms'], stats['wall']['p95_ms']]
            values += [stats[p]['median_ms'] for p in ("copy", "filter", "search", "result_other")]
            print(f"{name:<16} " + " ".join(f"{v:10.3f}" for v in values))
        print("SQL-only queries (same fields, all rows fetched), median / P95 ms:")
        for mode, stats in result["sql_query_stats"].items():
            print(f"  {mode:<8} {stats['median_ms']:.3f} / {stats['p95_ms']:.3f}")
        print("Paired savings, median ms (negative = slower):")
        for name, stats in result["savings_ms"].items():
            print(f"  {name}: {stats['median_ms']:.3f}")
        print(f"SQL rows returned: {result['sql_returned_rows']}; correctness: PASS", flush=True)
    print("\nTimings in ms. Warm-cache, setup excluded. Ablation savings must not be added together.")
    print("bfs_fixed omits scoring work; result_other and wall are not score-equivalent comparisons.")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
