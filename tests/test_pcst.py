"""Tests for the PCST (prize-collecting) algorithm."""

from src.graph_model import build_graph
from src.synthetic_data import load_conditions
from src.retrieval import retrieve


def _res(**kw):
    G = build_graph()
    conds = load_conditions()
    return retrieve(G, conds, algorithm="pcst", max_hops=4, strict=True, rho=0.0, **kw)


def _get(results, pid):
    return next(r for r in results if r.patient_id == pid)


def test_pcst_keeps_all_when_no_hub_penalty():
    """With rho=0 (no hub penalty) PCST keeps all reachable conditions."""
    r = _get(_res(), "P008")
    assert r.coverage_count == 5
    for cid in ["C1", "C2", "C3", "C4", "C5"]:
        assert r.condition_paths[cid]["matched"] is True


def test_pcst_hub_penalty_prunes_p008_hub_paths():
    """Test 8: with hub penalty PCST prunes P008's hub conditions (C1/C3/C5)."""
    G = build_graph()
    conds = load_conditions()
    results = retrieve(G, conds, algorithm="pcst", max_hops=4, strict=True,
                        rho=1.3, condition_prize=10.0)
    p008 = _get(results, "P008")
    pruned = {c for c, v in p008.condition_paths.items() if v.get("pruned")}
    assert pruned == {"C1", "C3", "C5"}
    assert p008.coverage_count == 2  # only C2 (SBP) and C4 (HbA1c) kept
    # P004 / P007 must NOT be pruned (their multi-hop paths are low-degree)
    p004 = _get(results, "P004")
    assert p004.coverage_count == 5
    p007 = _get(results, "P007")
    assert p007.coverage_count == 5


def test_pcst_vs_steiner_structural_difference():
    """Test 8: PCST (hub on) drops P008 conditions while Steiner keeps them."""
    G = build_graph()
    conds = load_conditions()
    pcst = retrieve(G, conds, algorithm="pcst", max_hops=4, strict=True, rho=1.3)
    steiner = retrieve(G, conds, algorithm="steiner", max_hops=4, strict=True, rho=1.3)
    p008_pcst = _get(pcst, "P008").coverage_count
    p008_steiner = _get(steiner, "P008").coverage_count
    assert p008_pcst == 2          # PCST pruned hub paths
    assert p008_steiner == 5       # Steiner keeps everything (only scores cost)
