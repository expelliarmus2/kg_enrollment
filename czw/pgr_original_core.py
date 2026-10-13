"""Original PGR bounded BFS with high-fanout patient edges in SQLite.

The retrieval graph keeps every structured relation between non-text nodes so
that paths such as ``anchor -> intermediate -> patient`` remain searchable.
Only ``patient -> entity`` fact edges are reversed to ``entity -> patient``;
only high-fanout edges of that reversed kind are offloaded to SQLite.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
from typing import Any

import networkx as nx


SQL_OPERATORS = {">", ">=", "<", "<=", "=", "=="}


@dataclass(frozen=True)
class SetupStats:
    patient_edges: int
    hubs: int
    sql_rows: int
    graph_edges: int
    setup_ms: float


def _numeric_value(edge: dict[str, Any]) -> float | None:
    raw = edge.get("value", edge.get("raw_relation"))
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _is_numeric(condition: dict[str, Any]) -> bool:
    return condition.get("kind") == "numeric" or (
        condition.get("operator") is not None and condition.get("value") is not None
    )


class PGRSQLRetriever:
    """Original bounded BFS with only high-fanout patient edges in SQLite."""

    def __init__(
        self,
        original: nx.MultiDiGraph,
        db_path: str | Path,
        *,
        hub_threshold: int = 20,
    ) -> None:
        started = time.perf_counter()
        self.graph, patient_edges = self._build_retrieval_graph(original)
        self.hubs = self._identify_hubs(hub_threshold)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(db_path))
        sql_rows = self._offload_hub_patient_edges()
        self.stats = SetupStats(
            patient_edges=patient_edges,
            hubs=len(self.hubs),
            sql_rows=sql_rows,
            graph_edges=self.graph.number_of_edges(),
            setup_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _build_retrieval_graph(
        original: nx.MultiDiGraph,
    ) -> tuple[nx.MultiDiGraph, int]:
        """Build the PGR-oriented graph without discarding multi-hop edges.

        External KG patient facts point ``patient -> entity`` while PGR searches
        from a condition anchor toward patients.  Those fact edges are reversed.
        Every other edge keeps its original direction and attributes, preserving
        the intermediate paths used by the original bounded BFS.
        """
        graph = nx.MultiDiGraph()
        for node, data in original.nodes(data=True):
            if data.get("type") != "original_text":
                graph.add_node(node, **data)
        patient_edges = 0
        for source, target, key, edge in original.edges(keys=True, data=True):
            if source not in graph or target not in graph:
                continue
            attributes = dict(edge)
            value = _numeric_value(edge)
            if value is not None:
                attributes["value"] = value
            if (
                original.nodes[source].get("type") == "patient"
                and original.nodes[target].get("type") == "entity"
            ):
                graph.add_edge(target, source, key=key, **attributes)
                patient_edges += 1
            else:
                graph.add_edge(source, target, key=key, **attributes)
        return graph, patient_edges

    def _identify_hubs(self, threshold: int) -> set[str]:
        hubs: set[str] = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "patient":
                continue
            patient_edges = sum(
                1
                for _source, target, _key in self.graph.out_edges(node, keys=True)
                if self.graph.nodes[target].get("type") == "patient"
            )
            if patient_edges > threshold:
                hubs.add(str(node))
        return hubs

    def _offload_hub_patient_edges(self) -> int:
        self.connection.executescript(
            """
            DROP TABLE IF EXISTS hub_patient_edges;
            CREATE TABLE hub_patient_edges (
                hub TEXT NOT NULL, patient TEXT NOT NULL, value REAL, unit TEXT,
                raw_relation TEXT, relation_type TEXT, chunk_id TEXT,
                valid_at TEXT, invalid_at TEXT
            );
            """
        )
        rows: list[tuple[Any, ...]] = []
        remove: list[tuple[str, str, Any]] = []
        for hub in self.hubs:
            for _source, patient, key, edge in self.graph.out_edges(hub, keys=True, data=True):
                if self.graph.nodes[patient].get("type") != "patient":
                    continue
                rows.append((
                    hub, patient, edge.get("value"), edge.get("unit"),
                    edge.get("raw_relation"), edge.get("relation_type"),
                    edge.get("chunk_id"), edge.get("valid_at"), edge.get("invalid_at"),
                ))
                remove.append((hub, patient, key))
        self.connection.executemany(
            "INSERT INTO hub_patient_edges VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        self.connection.executescript(
            """
            CREATE INDEX idx_hub_patient ON hub_patient_edges(hub, patient);
            CREATE INDEX idx_hub_unit_value_patient
                ON hub_patient_edges(hub, unit, value, patient);
            ANALYZE;
            """
        )
        self.connection.commit()
        for source, target, key in remove:
            self.graph.remove_edge(source, target, key)
        return len(rows)

    @staticmethod
    def _compare(value: float, operator: str, threshold: float) -> bool:
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<":
            return value < threshold
        if operator == "<=":
            return value <= threshold
        if operator in {"=", "=="}:
            return value == threshold
        raise ValueError(f"unsupported numeric operator: {operator}")

    def _conditioned_view(self, condition: dict[str, Any]) -> nx.MultiDiGraph:
        anchor = str(condition["anchor"])

        def edge_allowed(source: str, target: str, key: Any) -> bool:
            if source != anchor or not _is_numeric(condition):
                return True
            edge = self.graph.get_edge_data(source, target, key)
            if edge is None:
                return False
            value = _numeric_value(edge)
            if value is None or not self._compare(
                value, str(condition["operator"]), float(condition["value"])
            ):
                return False
            unit = condition.get("unit")
            return unit is None or edge.get("unit") == unit

        return nx.subgraph_view(self.graph, filter_edge=edge_allowed)

    def _query_hub(self, hub: str, condition: dict[str, Any], anchor: str):
        sql = """
            SELECT patient, value, unit, raw_relation, relation_type,
                   chunk_id, valid_at, invalid_at
            FROM hub_patient_edges WHERE hub = ?
        """
        parameters: list[Any] = [hub]
        if hub == anchor and _is_numeric(condition):
            operator = str(condition["operator"])
            if operator not in SQL_OPERATORS:
                raise ValueError(f"unsupported numeric operator: {operator}")
            if operator == "==":
                operator = "="
            if condition.get("unit") is not None:
                sql += " AND unit = ?"
                parameters.append(condition["unit"])
            sql += f" AND value {operator} ?"
            parameters.append(float(condition["value"]))
        return self.connection.execute(sql, parameters)

    def search(
        self,
        condition: dict[str, Any],
        *,
        max_hops: int = 4,
    ) -> tuple[dict[str, dict[str, Any]], int, int]:
        """Run the original queue/path/visited BFS for one condition."""
        graph_view = self._conditioned_view(condition)
        anchor = str(condition["anchor"])
        visited_nodes: set[str] = {anchor}
        visited_edges = 0
        hits: dict[str, dict[str, Any]] = {}
        queue: deque[
            tuple[str, int, list[str], list[tuple[str, str, Any, dict[str, Any]]]]
        ] = deque([(anchor, 0, [anchor], [])])

        while queue:
            node, distance, path, path_edges = queue.popleft()
            # SQL rows represent ordinary outgoing graph edges.  They must obey
            # the same depth boundary as graph-resident edges.
            if distance >= max_hops:
                continue
            if node in self.hubs:
                for row in self._query_hub(node, condition, anchor):
                    patient, value, unit, raw, relation_type, chunk, valid, invalid = row
                    if patient in path:
                        continue
                    edge = {
                        "source": node, "target": patient,
                        "relation_type": relation_type, "value": value,
                        "unit": unit, "raw_relation": raw, "chunk_id": chunk,
                        "valid_at": valid, "invalid_at": invalid, "storage": "sql",
                    }
                    new_distance = distance + 1
                    if patient not in hits or new_distance < hits[patient]["distance"]:
                        hits[patient] = {
                            "distance": new_distance, "path": path + [patient],
                            "edge_data": edge, "reason": "MATCH(SQL_HUB)",
                            "_edges": path_edges + [(node, patient, "sql", edge)],
                        }
                    visited_nodes.add(patient)
                    queue.append((
                        patient,
                        new_distance,
                        path + [patient],
                        path_edges + [(node, patient, "sql", edge)],
                    ))

            for _source, neighbor, key, edge_data in graph_view.out_edges(
                node, keys=True, data=True
            ):
                visited_edges += 1
                if neighbor in path:
                    continue
                new_distance = distance + 1
                new_path = path + [neighbor]
                edge = {
                    "source": node, "target": neighbor,
                    "relation_type": edge_data.get("relation_type", key),
                    **edge_data, "storage": "graph",
                }
                new_edges = path_edges + [(node, neighbor, key, edge)]
                visited_nodes.add(neighbor)
                if graph_view.nodes[neighbor].get("type") == "patient":
                    if neighbor not in hits or new_distance < hits[neighbor]["distance"]:
                        hits[neighbor] = {
                            "distance": new_distance, "path": new_path,
                            "edge_data": edge, "reason": "MATCH(BFS_VIEW)",
                            "_edges": new_edges,
                        }
                queue.append((neighbor, new_distance, new_path, new_edges))
        return hits, len(visited_nodes), visited_edges

    def close(self) -> None:
        self.connection.close()
