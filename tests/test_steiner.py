"""Tests for the Steiner-tree based keyword search (shared edge once)."""

from src.graph_model import build_graph
from src.synthetic_data import load_conditions
from src.retrieval import retrieve


def _res(**kw):
    G = build_graph()
    conds = load_conditions()
    return retrieve(G, conds, algorithm="steiner", max_hops=4, strict=True, rho=0.0, **kw)


def _get(results, pid):
    return next(r for r in results if r.patient_id == pid)


def test_steiner_shared_edge_counted_once():
    """Test 7: P004's C3 and C4 paths share Encounter_P4->P004; union counts it once.

    C3: Diabetes -> MetabolicCondition -> Encounter_P4 -> P004  (3 edges)
    C4: HbA1c    -> LabPanel_P4      -> Encounter_P4 -> P004  (3 edges, shares last)
    Naive sum of edges over all 5 condition paths = 1+1+3+3+2 = 10.
    Shared edge -> union must have 9 edges (cost 9.0 at rho=0).
    """
    r = _get(_res(), "P004")
    assert r.coverage_count == 5
    # each edge in the union subgraph is unique
    keys = [(e["source"], e["target"]) for e in r.edges]
    assert len(keys) == len(set(keys)), "duplicate edges in union!"
    # naive count
    naive = 0
    for cid in ["C1", "C2", "C3", "C4", "C5"]:
        p = r.condition_paths[cid]["path"]
        naive += len(p) - 1
    assert naive == 10
    assert len(r.edges) == 9  # one shared edge removed
    assert r.graph_cost == 9.0  # rho=0 => 1.0 per edge


def test_steiner_finds_all_strict():
    """Steiner (rho=0) admits the same strict cohort as ground truth."""
    r = _res()
    strict = {x.patient_id for x in r if x.strict_match}
    assert strict == {"P001", "P004", "P007", "P008"}


def test_steiner_p003_sbp_excluded():
    """P003 SBP numeric fail -> not strict in Steiner too."""
    r = _get(_res(), "P003")
    assert r.condition_paths["C2"]["matched"] is False
    assert r.coverage_count == 4
