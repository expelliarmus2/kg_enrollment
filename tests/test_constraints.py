"""Tests for numeric constraint hard filtering."""

import pytest

from src.graph_model import build_graph
from src.synthetic_data import load_conditions
from src.constraints import edge_satisfies_condition, build_query_conditioned_graph


def _cond(cid):
    return next(c for c in load_conditions() if c["id"] == cid)


def test_p001_sbp_passes():
    """P001 SBP=158 satisfies SBP > 140."""
    G = build_graph()
    for _, v, k, d in G.out_edges("Systolic Blood Pressure", keys=True, data=True):
        if v == "P001":
            assert edge_satisfies_condition(d, _cond("C2")) is True
            assert d["value"] == 158


def test_p003_sbp_fails():
    """P003 SBP=132 must FAIL SBP > 140 (hard filter)."""
    G = build_graph()
    for _, v, k, d in G.out_edges("Systolic Blood Pressure", keys=True, data=True):
        if v == "P003":
            assert d["value"] == 132
            assert edge_satisfies_condition(d, _cond("C2")) is False


def test_p006_hba1c_fails():
    """P006 HbA1c=6.1 must FAIL HbA1c > 7.0 (hard filter)."""
    G = build_graph()
    for _, v, k, d in G.out_edges("HbA1c", keys=True, data=True):
        if v == "P006":
            assert d["value"] == 6.1
            assert edge_satisfies_condition(d, _cond("C4")) is False


def test_p001_hba1c_passes():
    """P001 HbA1c=8.2 satisfies HbA1c > 7.0."""
    G = build_graph()
    for _, v, k, d in G.out_edges("HbA1c", keys=True, data=True):
        if v == "P001":
            assert edge_satisfies_condition(d, _cond("C4")) is True


def test_categorical_always_passes():
    """Categorical condition edges always satisfy (no value filter)."""
    G = build_graph()
    for _, v, k, d in G.out_edges("Hypertension", keys=True, data=True):
        assert edge_satisfies_condition(d, _cond("C1")) is True


def test_query_conditioned_graph_removes_failing_edges():
    """The filtered graph must drop P003's SBP edge and P006's HbA1c edge."""
    G = build_graph()
    conds = load_conditions()
    H = build_query_conditioned_graph(G, conds)
    # P003 SBP edge removed
    has_p003_sbp = any(
        v == "P003" for _, v, k, d in H.out_edges("Systolic Blood Pressure", keys=True, data=True)
    )
    assert has_p003_sbp is False
    # P001 SBP edge kept
    has_p001_sbp = any(
        v == "P001" for _, v, k, d in H.out_edges("Systolic Blood Pressure", keys=True, data=True)
    )
    assert has_p001_sbp is True
    # P006 HbA1c edge removed
    has_p006_hba1c = any(
        v == "P006" for _, v, k, d in H.out_edges("HbA1c", keys=True, data=True)
    )
    assert has_p006_hba1c is False
