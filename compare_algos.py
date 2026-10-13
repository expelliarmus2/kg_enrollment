"""Compare the three retrieval algorithms on the synthetic graph.

Runs BFS, PCST, Steiner (with and without hub penalty rho) and reports
per-patient coverage/score, precision/recall vs ground truth, and timing.
"""

from __future__ import annotations

import time

from src.graph_model import build_graph
from src.synthetic_data import build_default_conditions, load_ground_truth
from src.retrieval import retrieve
from src.evaluation import evaluate

ALGOS = ["bfs", "pcst", "steiner"]
PATIENTS = [f"P{i:03d}" for i in range(1, 9)]
COND_IDS = ["C1", "C2", "C3", "C4", "C5"]
GT = load_ground_truth()
GT_STRICT = set(GT["strict_valid_patients"])


def fmt_cov(r) -> str:
    return f"{r.coverage_count}/{len(COND_IDS)}"


def run_one(algo: str, max_hops: int, rho: float):
    G = build_graph()
    conds = build_default_conditions()
    t0 = time.perf_counter()
    results = retrieve(G, conds, algorithm=algo, max_hops=max_hops,
                       strict=True, rho=rho, condition_prize=10.0)
    wall = (time.perf_counter() - t0) * 1000
    eval_ = evaluate(results, strict=True)
    return results, eval_, wall


def run_shared(algo: str, max_hops: int, rho: float):
    """Average wall time over n repeats to stabilize small timings."""
    G = build_graph()
    conds = build_default_conditions()
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        retrieve(G, conds, algorithm=algo, max_hops=max_hops,
                 strict=True, rho=rho, condition_prize=10.0)
    wall = (time.perf_counter() - t0) * 1000 / n
    results = retrieve(G, conds, algorithm=algo, max_hops=max_hops,
                      strict=True, rho=rho, condition_prize=10.0)
    eval_ = evaluate(results, strict=True)
    return results, eval_, wall


def print_per_patient(algo: str, rho: float, results):
    tag = f"{algo} (rho={rho})"
    print(f"\n--- {tag} ---")
    print(f"{'Patient':<8}{'cov':<6}{'score':<9}{'gcost':<7}{'strict':<8}{'GT?':<5}matched")
    by_pid = {r.patient_id: r for r in results}
    for pid in PATIENTS:
        r = by_pid[pid]
        gt_flag = "Y" if pid in GT_STRICT else " "
        print(f"{pid:<8}{fmt_cov(r):<6}{r.score:<9.2f}{r.graph_cost:<7.2f}"
              f"{'Y' if r.strict_match else '-':<8}{gt_flag:<5}"
              f"{','.join(r.matched_conditions) if r.matched_conditions else '-'}")


def print_summary_table(rows):
    print("\n" + "=" * 82)
    print(f"{'algorithm':<16}{'rho':<6}{'prec':<7}{'rec':<7}{'f1':<7}"
          f"{'n_ret':<7}{'wall_ms':<9}{'vis_nodes':<10}{'vis_edges':<10}")
    print("-" * 82)
    for row in rows:
        ev = row["eval"]
        print(f"{row['algo']:<16}{row['rho']:<6}{ev['precision']:<7}{ev['recall']:<7}"
              f"{ev['f1']:<7}{ev['num_returned']:<7}{row['wall']:<9.3f}"
              f"{ev['total_visited_nodes']:<10}{ev['total_visited_edges']:<10}")
    print("=" * 82)


def main():
    print("Patient-Centered Graph Retrieval -- 3-algorithm comparison")
    print(f"Ground-truth strict cohort: {sorted(GT_STRICT)}")
    print(f"Default 5 conditions, MAX_HOPS=4, strict mode")

    summaries = []
    # Default: rho=0 (no hub penalty) for all three
    for algo in ALGOS:
        results, eval_, wall = run_one(algo, max_hops=4, rho=0.0)
        print_per_patient(algo, 0.0, results)
        summaries.append({"algo": algo, "rho": 0.0, "eval": eval_, "wall": wall})

    # Hub penalty on for PCST/Steiner (README demo: rho=1.3)
    for algo in ["pcst", "steiner"]:
        results, eval_, wall = run_one(algo, max_hops=4, rho=1.3)
        print_per_patient(algo, 1.3, results)
        summaries.append({"algo": algo, "rho": 1.3, "eval": eval_, "wall": wall})

    # ---- single-pass shared variants (new) ----
    print("\n" + "#" * 82)
    print("# Single-pass shared variants (one BFS per anchor, not per patient)")
    print("#" * 82)
    shared_rows = []
    for algo in ["pcst_shared", "steiner_shared"]:
        for rho in [0.0, 1.3]:
            results, eval_, wall = run_shared(algo, max_hops=4, rho=rho)
            shared_rows.append({"algo": algo, "rho": rho, "eval": eval_, "wall": wall})
    # compare old vs shared side by side
    print(f"\n{'algorithm':<16}{'rho':<6}{'prec':<7}{'rec':<7}{'f1':<7}"
          f"{'vis_nodes':<10}{'vis_edges':<10}{'wall_ms(50avg)':<13}")
    print("-" * 82)
    # include old variants (averaged too for fair timing) for the comparison block
    for algo in ["bfs", "pcst", "steiner", "pcst_shared", "steiner_shared"]:
        for rho in [0.0] + ([1.3] if algo not in ("bfs",) else []):
            _, ev, wall = run_shared(algo, max_hops=4, rho=rho)
            print(f"{algo:<16}{rho:<6}{ev['precision']:<7}{ev['recall']:<7}"
                  f"{ev['f1']:<7}{ev['total_visited_nodes']:<10}"
                  f"{ev['total_visited_edges']:<10}{wall:<13.3f}")
    print("=" * 82)

    print_summary_table(summaries)


if __name__ == "__main__":
    main()
