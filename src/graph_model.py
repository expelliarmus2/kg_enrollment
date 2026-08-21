"""Graph model: build a NetworkX MultiDiGraph from synthetic data.

Edges are directed condition-anchor -> ... -> patient.  Numeric measurement
edges carry a ``value`` attribute used by hard constraint filtering.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from . import synthetic_data as sd

MAX_HOPS = sd.MAX_HOPS


def build_graph() -> nx.MultiDiGraph:
    """Build the directed multigraph from synthetic data (writes data files)."""
    data = sd.load_graph_data()
    G = nx.MultiDiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], **node)
    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            key=edge.get("relation_type", "rel"),
            **edge,
        )
    return G


def node_type(G: nx.MultiDiGraph, node_id: str) -> str:
    return G.nodes[node_id].get("type", "intermediate")


def patient_nodes(G: nx.MultiDiGraph) -> list[str]:
    return [n for n, d in G.nodes(data=True) if d.get("type") == "patient"]


def all_edges(G: nx.MultiDiGraph) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for u, v, k, d in G.edges(keys=True, data=True):
        row = {"source": u, "target": v, "relation_type": k}
        row.update({kk: vv for kk, vv in d.items() if kk not in ("source", "target", "relation_type")})
        out.append(row)
    return out


def degree(G: nx.MultiDiGraph, node_id: str) -> int:
    return G.degree(node_id)


def edge_cost(
    G: nx.MultiDiGraph,
    u: str,
    v: str,
    *,
    base: float = 1.0,
    rho: float = 0.0,
) -> float:
    """Edge traversal cost with optional hub penalty on the *target* node.

    ``c(e) = base + rho * log(1 + deg(v))``
    """
    import math

    deg_v = degree(G, v)
    return base + rho * math.log(1.0 + deg_v)
