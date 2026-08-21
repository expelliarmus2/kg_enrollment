"""Condition constraints and numeric hard filtering.

A condition is either:
  * ``categorical``  - any edge leaving the anchor is traversable
  * ``numeric``      - only edges carrying a ``value`` that satisfies the
    operator/threshold are traversable (hard filter, NOT a soft penalty).

Edges failing a numeric constraint are removed from the query-conditioned
graph BEFORE any search (BFS / PCST / Steiner) runs.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def _cmp(value: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    return False


def edge_satisfies_condition(edge: dict[str, Any], condition: dict[str, Any]) -> bool:
    """Return True if ``edge`` is allowed under ``condition``.

    For categorical conditions the answer is always True.
    For numeric conditions the edge must carry a ``value`` attribute and that
    value must satisfy the operator/threshold.
    """
    if condition.get("kind", "categorical") == "numeric":
        if "value" not in edge:
            return False
        return _cmp(float(edge["value"]), condition["operator"], float(condition["value"]))
    return True


def condition_anchor(condition: dict[str, Any]) -> str:
    return condition["anchor"]


def is_numeric(condition: dict[str, Any]) -> bool:
    return condition.get("kind") == "numeric"


def build_query_conditioned_graph(
    G: nx.MultiDiGraph,
    conditions: list[dict[str, Any]],
) -> nx.MultiDiGraph:
    """Return a copy of ``G`` with numeric-failing anchor edges removed.

    Only edges leaving a numeric condition's anchor are subject to filtering;
    the filter removes the specific failing measurement edges.  All other
    edges (categorical anchors, intermediate->intermediate, etc.) are kept.
    """
    H = G.copy()
    numeric_anchors: dict[str, dict[str, Any]] = {}
    for cond in conditions:
        if is_numeric(cond):
            numeric_anchors[cond["anchor"]] = cond

    # Collect edges to remove (by (u,v,key)) - cannot mutate while iterating
    to_remove: list[tuple[str, str, str]] = []
    for u, v, k, d in H.edges(keys=True, data=True):
        if u in numeric_anchors:
            cond = numeric_anchors[u]
            if not edge_satisfies_condition(d, cond):
                to_remove.append((u, v, k))
    for (u, v, k) in to_remove:
        H.remove_edge(u, v, k)
    return H


def describe_filter(
    G: nx.MultiDiGraph,
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Human-readable list of which edges were filtered out and why."""
    numeric_anchors = {c["anchor"]: c for c in conditions if is_numeric(c)}
    report: list[dict[str, Any]] = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u in numeric_anchors:
            cond = numeric_anchors[u]
            ok = edge_satisfies_condition(d, cond)
            report.append({
                "condition_id": cond.get("id"),
                "anchor": u,
                "target": v,
                "value": d.get("value"),
                "operator": cond["operator"],
                "threshold": cond["value"],
                "satisfied": ok,
                "reason": (
                    "MATCH" if ok else
                    f"{d.get('value')} {cond['operator']} {cond['value']} -> FAIL"
                ),
            })
    return report
