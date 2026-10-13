"""Tests for the single-pass shared-traversal variants.

The shared variants must produce results IDENTICAL to the per-patient
implementations (same coverage, paths, edges, scores) while visiting
far fewer edges (one BFS per anchor instead of one per (anchor, patient)).
"""

from src.graph_model import build_graph
from src.synthetic_data import load_conditions
from src.retrieval import retrieve

_PIDS = [f"P{i:03d}" for i in range(1, 9)]
_EQ_FIELDS = [
    "patient_id", "score", "coverage_count", "coverage_ratio",
    "matched_conditions", "condition_paths", "nodes", "edges",
    "graph_cost", "strict_match",
]


def _run(algo, **kw):
    G = build_graph()
    conds = load_conditions()
    return retrieve(G, conds, algorithm=algo, max_hops=4, strict=True, **kw)


def _by_pid(results):
    return {r.patient_id: r for r in results}


def _normalize(r):
    d = r.to_dict()
    return {k: d[k] for k in _EQ_FIELDS}


def _assert_identical(old, new, rho):
    old_map, new_map = _by_pid(old), _by_pid(new)
    assert set(old_map) == set(new_map) == set(_PIDS), "must return all 8 patients"
    for pid in _PIDS:
        o, n = _normalize(old_map[pid]), _normalize(new_map[pid])
        assert o == n, f"{pid} differs between old and shared (rho={rho}):\n{o}\n!=\n{n}"


def test_one_retrieval_returns_all_patients():
    """A single retrieve() call returns every patient center in one pass."""
    results = _run("pcst_shared")
    assert len(results) == 8
    assert {r.patient_id for r in results} == set(_PIDS)


def test_pcst_shared_matches_pcst_rho0():
    old = _run("pcst", rho=0.0)
    new = _run("pcst_shared", rho=0.0)
    _assert_identical(old, new, 0.0)


def test_pcst_shared_matches_pcst_hub_penalty():
    """Shared variant must reproduce hub-pruning of P008 exactly."""
    old = _run("pcst", rho=1.3, condition_prize=10.0)
    new = _run("pcst_shared", rho=1.3, condition_prize=10.0)
    _assert_identical(old, new, 1.3)
    # sanity: P008 pruned to 2/5 in both
    assert _by_pid(new)["P008"].coverage_count == 2


def test_steiner_shared_matches_steiner_rho0():
    old = _run("steiner", rho=0.0)
    new = _run("steiner_shared", rho=0.0)
    _assert_identical(old, new, 0.0)


def test_steiner_shared_matches_steiner_hub_penalty():
    old = _run("steiner", rho=1.3)
    new = _run("steiner_shared", rho=1.3)
    _assert_identical(old, new, 1.3)
    # P008 still 5/5 in Steiner (no pruning)
    assert _by_pid(new)["P008"].coverage_count == 5


def test_shared_visits_far_fewer_edges():
    """The whole point: one traversal per anchor, not one per patient."""
    old_pcst = _run("pcst")
    new_pcst = _run("pcst_shared")
    old_ste = _run("steiner")
    new_ste = _run("steiner_shared")
    assert _by_pid(new_pcst)["P001"].visited_edges < _by_pid(old_pcst)["P001"].visited_edges
    assert _by_pid(new_ste)["P001"].visited_edges < _by_pid(old_ste)["P001"].visited_edges


def test_shared_algorithm_label():
    assert _by_pid(_run("pcst_shared"))["P001"].algorithm == "pcst_shared"
    assert _by_pid(_run("steiner_shared"))["P001"].algorithm == "steiner_shared"
