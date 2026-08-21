"""Tests for the bounded constrained BFS baseline."""

from src.graph_model import build_graph
from src.synthetic_data import load_conditions, load_ground_truth
from src.retrieval import retrieve


def _res(algorithm="bfs", max_hops=4, rho=0.0, strict=True, soft=0.8):
    G = build_graph()
    conds = load_conditions()
    return retrieve(G, conds, algorithm=algorithm, max_hops=max_hops,
                     strict=strict, soft_threshold=soft, rho=rho)


def _get(results, pid):
    return next(r for r in results if r.patient_id == pid)


def test_p001_matches_all_conditions():
    """Test 1: P001 satisfies all 5 conditions."""
    r = _get(_res(), "P001")
    assert r.coverage_count == 5
    assert r.strict_match is True
    # all conditions 1-hop
    for cid in ["C1", "C2", "C3", "C4", "C5"]:
        assert r.condition_paths[cid]["matched"] is True
        assert r.condition_paths[cid]["distance"] == 1


def test_p004_multihop_maxhops1_fails():
    """Test 4: P004 needs multi-hop for Diabetes; MAX_HOPS=1 -> not matched."""
    r = _get(_res(max_hops=1), "P004")
    assert r.condition_paths["C3"]["matched"] is False  # Diabetes unreachable in 1 hop
    assert r.coverage_count < 5
    assert r.strict_match is False


def test_p004_multihop_maxhops3_passes():
    """Test 4: with MAX_HOPS>=3 P004 satisfies Diabetes (3-hop)."""
    r = _get(_res(max_hops=3), "P004")
    assert r.condition_paths["C3"]["matched"] is True
    assert r.condition_paths["C3"]["distance"] == 3  # Diabetes -> MetabolicCondition -> Encounter_P4 -> P004
    assert r.condition_paths["C4"]["matched"] is True
    assert r.condition_paths["C4"]["distance"] == 3
    assert r.coverage_count == 5
    assert r.strict_match is True


def test_all_patients_are_candidates():
    """Test 5: every patient appears in results (none hardcode-excluded)."""
    results = _res()
    pids = {r.patient_id for r in results}
    assert pids == {"P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008"}


def test_ground_truth_reproduced_independently():
    """Test 5: retrieval (rho=0, hops=4) independently reproduces ground truth."""
    results = _res()
    gt = load_ground_truth()
    cond_ids = ["C1", "C2", "C3", "C4", "C5"]
    for pid in ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008"]:
        r = _get(results, pid)
        for cid in cond_ids:
            assert r.condition_paths[cid]["matched"] == gt[pid][cid], (
                f"{pid} {cid}: algo={r.condition_paths[cid]['matched']} gt={gt[pid][cid]}"
            )
    assert set(gt["strict_valid_patients"]) == {
        r.patient_id for r in results if r.strict_match
    }


def test_prefer_shortest_path_over_multihop():
    """Test 6: when both 1-hop and 3-hop exist, the 1-hop distance wins."""
    # P001 has a direct Hypertension->P001 (1-hop); verify distance is 1.
    r = _get(_res(max_hops=4), "P001")
    assert r.condition_paths["C1"]["distance"] == 1
    # Build a tiny graph where both a direct and a 3-hop path exist and
    # confirm the algorithm reports the shortest.
    import networkx as nx
    g = nx.MultiDiGraph()
    for n in ["A", "X", "Y", "P"]:
        g.add_node(n, type="patient" if n == "P" else "disease")
    g.add_edge("A", "P", key="direct", relation_type="r")           # 1-hop
    g.add_edge("A", "X", key="r1", relation_type="r")
    g.add_edge("X", "Y", key="r2", relation_type="r")
    g.add_edge("Y", "P", key="r3", relation_type="r")                # 3-hop
    conds = [{"id": "C1", "anchor": "A", "kind": "categorical"}]
    res = retrieve(g, conds, algorithm="bfs", max_hops=4, strict=True)
    p = next(r for r in res if r.patient_id == "P")
    assert p.condition_paths["C1"]["distance"] == 1  # shortest wins
    assert p.condition_paths["C1"]["path"] == ["A", "P"]


def test_strict_mode_excludes_partial():
    """Test 9: strict mode does NOT admit 4/5 patients."""
    results = _res(strict=True)
    for pid in ["P002", "P003"]:  # both 4/5
        r = _get(results, pid)
        assert r.coverage_count == 4
        assert r.strict_match is False


def test_soft_mode_includes_partial():
    """Test 10: soft mode (threshold 0.8) admits 4/5 patients."""
    results = _res(strict=False, soft=0.8)
    for pid in ["P002", "P003"]:
        r = _get(results, pid)
        assert r.coverage_ratio >= 0.8
    # 2/5 patients still excluded
    r5 = _get(results, "P005")
    assert r5.coverage_ratio < 0.8
