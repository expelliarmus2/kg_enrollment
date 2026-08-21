"""Tests for ground-truth consistency and candidate handling."""

from src.graph_model import build_graph
from src.synthetic_data import build_ground_truth, load_ground_truth, PATIENTS
from src.constraints import build_query_conditioned_graph
from src.synthetic_data import load_conditions


def test_ground_truth_completeness():
    """Every patient has a full ground-truth record."""
    gt = build_ground_truth()
    for pid in PATIENTS:
        assert pid in gt
        assert len(gt[pid]) == 5
        assert all(isinstance(v, bool) for v in gt[pid].values())
    assert "strict_valid_patients" in gt
    assert set(gt["strict_valid_patients"]) <= set(PATIENTS)


def test_strict_valid_are_full_matches():
    """strict_valid_patients must be exactly the 5/5 patients."""
    gt = load_ground_truth()
    cids = ["C1", "C2", "C3", "C4", "C5"]
    full = [pid for pid in PATIENTS if all(gt[pid][c] for c in cids)]
    assert set(full) == set(gt["strict_valid_patients"])
    assert set(full) == {"P001", "P004", "P007", "P008"}


def test_partial_patients_documented():
    """Documented partial matches have correct coverage."""
    gt = load_ground_truth()
    cids = ["C1", "C2", "C3", "C4", "C5"]

    def cov(pid):
        return sum(gt[pid][c] for c in cids)

    assert cov("P002") == 4  # missing Amlodipine
    assert cov("P003") == 4  # SBP=132 fail
    assert cov("P005") == 2  # only Hypertension + SBP
    assert cov("P006") == 3  # HbA1c=6.1 fail
    assert gt["P002"]["C5"] is False
    assert gt["P003"]["C2"] is False
    assert gt["P006"]["C4"] is False


def test_no_unintended_hub_shortcuts():
    """The hub must not create unintended short paths to non-P008 patients.

    On the query-conditioned graph, CommonEncounter must only reach P008
    among patients (so the hub exclusively serves the P008 demo).
    """
    G = build_graph()
    conds = load_conditions()
    H = build_query_conditioned_graph(G, conds)
    # who does CommonEncounter reach?
    reachable_patients = set()
    stack = ["CommonEncounter"]
    seen = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if H.nodes[n].get("type") == "patient" and n != "CommonEncounter":
            reachable_patients.add(n)
        for _, nbr in H.out_edges(n):
            stack.append(nbr)
    assert reachable_patients == {"P008"}
