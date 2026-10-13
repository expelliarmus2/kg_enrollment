"""Shared single-pass traversal helper.

One BFS per condition anchor records the shortest valid path to EVERY
reachable patient in a single pass, instead of re-running a BFS per
(anchor, patient) pair.  This collapses the search phase of PCST/Steiner
from O(K * P * (V+E)) to O(K * (V+E)) -- the patient count P drops out.

Semantics are identical to the per-patient ``_shortest_path_to_patient``
used by the original implementations: BFS order, first (shortest) hit
wins ties, and cycle avoidance is done via path membership.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx


def shortest_paths_to_patients(
    G: nx.MultiDiGraph,
    anchor: str,
    max_hops: int,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """BFS from ``anchor`` up to ``max_hops``; return shortest path to each patient.

    Returns ``(hits, visited_nodes, visited_edges)`` where
    ``hits[patient_id] = {"distance": int, "path": [anchor, ..., patient]}``.
    A patient appears in ``hits`` iff reachable within ``max_hops``.
    """
    visited_nodes: set[str] = {anchor}
    visited_edges = 0
    best: dict[str, dict[str, Any]] = {}
    queue: deque[tuple[str, int, list[str]]] = deque([(anchor, 0, [anchor])])
    while queue:
        node, dist, path = queue.popleft()
        if dist >= max_hops:
            continue
        for _, nbr, _k, _edata in G.out_edges(node, keys=True, data=True):
            visited_edges += 1
            if nbr in path:
                continue  # avoid cycles
            new_path = path + [nbr]
            new_dist = dist + 1
            visited_nodes.add(nbr)
            if G.nodes[nbr].get("type") == "patient":
                if nbr not in best or new_dist < best[nbr]["distance"]:
                    best[nbr] = {"distance": new_dist, "path": new_path}
            queue.append((nbr, new_dist, new_path))
    return best, len(visited_nodes), visited_edges
