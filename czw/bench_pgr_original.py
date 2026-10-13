#!/usr/bin/env python3
"""Evaluate the ten structured queries with the original PGR BFS+SQL search."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import pickle
import re
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from pgr_original_core import PGRSQLRetriever


CHUNK_SUFFIX = re.compile(r"^(.*)-chunk-\d+$")


class UnsupportedQuery(ValueError):
    pass


def patient_key(value: Any) -> str:
    """Normalize graph node IDs such as ``patient:97122651`` for scoring."""
    text = str(value)
    return text.removeprefix("patient:")


def case_key(chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    match = CHUNK_SUFFIX.match(chunk_id)
    return match.group(1) if match else chunk_id


def parsed_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def xlsx_rows(path: Path) -> list[list[str]]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(".//x:t", namespace))
                for item in root.findall("x:si", namespace)
            ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in sheet.findall(".//x:sheetData/x:row", namespace):
        values: dict[int, str] = {}
        for cell in row.findall("x:c", namespace):
            letters = re.match(r"[A-Z]+", cell.get("r", "A1"))
            if letters is None:
                continue
            column = 0
            for letter in letters.group(0):
                column = column * 26 + ord(letter) - 64
            value_node = cell.find("x:v", namespace)
            if cell.get("t") == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//x:t", namespace))
            elif value_node is None:
                value = ""
            elif cell.get("t") == "s":
                value = shared[int(value_node.text or "0")]
            else:
                value = value_node.text or ""
            values[column - 1] = value
        rows.append([values.get(index, "") for index in range(max(values, default=-1) + 1)])
    return rows


def load_answers(path: Path) -> dict[str, set[str]]:
    rows = xlsx_rows(path)
    header = {name: index for index, name in enumerate(rows[0])}
    answers: dict[str, set[str]] = defaultdict(set)
    for row in rows[1:]:
        if row[header["baseline_match"]] in {"1", "1.0"}:
            answers[row[header["question_id"]]].add(patient_key(row[header["patient_key"]]))
    return dict(answers)


def term_matches(name: str, term: str) -> bool:
    if term.isascii() and term.isalpha():
        return re.search(rf"(?<![A-Z0-9]){re.escape(term.upper())}(?![A-Z0-9])", name.upper()) is not None
    return term in name


class StructuredPGR:
    def __init__(self, retriever: PGRSQLRetriever, max_hops: int):
        self.retriever = retriever
        self.max_hops = max_hops
        self.entities = [
            (str(node), str(data.get("name") or node))
            for node, data in retriever.graph.nodes(data=True)
            if data.get("type") == "entity"
        ]
        self.anchor_cache: dict[str, list[str]] = {}

    def anchors(self, term: str) -> list[str]:
        if term not in self.anchor_cache:
            exact = [node for node, name in self.entities if name == term or node == term]
            matched = exact or [node for node, name in self.entities if term_matches(name, term)]
            self.anchor_cache[term] = matched
        return self.anchor_cache[term]

    def leaf_groups(self, leaf: dict[str, Any]) -> list[dict[str, Any]]:
        kind = leaf["type"]
        required = set(leaf.get("required", ()))
        if kind == "field_compare":
            raise UnsupportedQuery("graph has no structured reference_upper field")
        if kind == "text":
            groups: list[dict[str, Any]] = []
            for term in leaf.get("all", ()):
                groups.append(self._term_group([term], required, numeric=False))
            if leaf.get("any"):
                groups.append(self._term_group(list(leaf["any"]), required, numeric=False))
            return groups
        if kind == "number":
            term = str(leaf["entity"])
            if leaf.get("operator") == "between":
                low, high = leaf["values"]
                return [
                    self._term_group([term], required, numeric=True, operator=">=", value=low, unit=leaf.get("unit")),
                    self._term_group([term], required, numeric=True, operator="<=", value=high, unit=leaf.get("unit")),
                ]
            return [self._term_group([term], required, numeric=True, unit=leaf.get("unit"))]
        raise UnsupportedQuery(f"unsupported condition type: {kind}")

    def _term_group(
        self,
        terms: list[str],
        required: set[str],
        *,
        numeric: bool,
        operator: str | None = None,
        value: float | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        alternatives: list[dict[str, Any]] = []
        for term in terms:
            for anchor in self.anchors(term):
                condition: dict[str, Any] = {
                    "anchor": anchor,
                    "kind": "numeric" if numeric and operator is not None else "categorical",
                }
                if operator is not None:
                    condition.update(operator=operator, value=value)
                if unit is not None:
                    condition["unit"] = unit
                alternatives.append(condition)
        return {
            "alternatives": alternatives,
            "require_number": numeric or "value" in required,
            "require_date": bool({"event_date", "sample_date"} & required),
        }

    @staticmethod
    def _edge_valid(hit: dict[str, Any], group: dict[str, Any]) -> bool:
        edge = hit["edge_data"]
        if group["require_number"]:
            try:
                float(edge.get("value", edge.get("raw_relation")))
            except (TypeError, ValueError):
                return False
        if group["require_date"] and not edge.get("valid_at"):
            return False
        return True

    def run_group(self, group: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for condition in group["alternatives"]:
            hits, _visited_nodes, _visited_edges = self.retriever.search(
                condition, max_hops=self.max_hops
            )
            for patient, hit in hits.items():
                if self._edge_valid(hit, group):
                    merged[str(patient)].append(hit)
        return dict(merged)

    def all_groups(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for item in items:
            groups.extend(self.leaf_groups(item))
        return groups

    @staticmethod
    def patient_intersection(results: list[dict[str, list[dict[str, Any]]]]) -> set[str]:
        if not results:
            return set()
        common = set(results[0])
        for result in results[1:]:
            common &= set(result)
        return common

    @staticmethod
    def same_case(patient: str, results: list[dict[str, list[dict[str, Any]]]]) -> bool:
        common: set[str] | None = None
        for result in results:
            cases = {
                key for hit in result[patient]
                if (key := case_key(hit["edge_data"].get("chunk_id"))) is not None
            }
            common = cases if common is None else common & cases
        return bool(common)

    def execute(self, query: dict[str, Any]) -> set[str]:
        node = query["query"]
        operator = node["operator"]
        if operator in {"patient_all", "same_case_all"}:
            groups = self.all_groups(node["items"])
            results = [self.run_group(group) for group in groups]
            candidates = self.patient_intersection(results)
            if operator == "same_case_all":
                candidates = {patient for patient in candidates if self.same_case(patient, results)}
            return candidates
        if operator == "same_case_temporal":
            from_results = [self.run_group(group) for group in self.leaf_groups(node["from"])]
            to_results = [self.run_group(group) for group in self.leaf_groups(node["to"])]
            candidates = self.patient_intersection(from_results + to_results)
            low, high = node["date_gap_days"]
            matched: set[str] = set()
            for patient in candidates:
                left = [hit for result in from_results for hit in result[patient]]
                right = [hit for result in to_results for hit in result[patient]]
                for before in left:
                    for after in right:
                        before_edge, after_edge = before["edge_data"], after["edge_data"]
                        if case_key(before_edge.get("chunk_id")) != case_key(after_edge.get("chunk_id")):
                            continue
                        before_time = parsed_time(before_edge.get("valid_at"))
                        after_time = parsed_time(after_edge.get("valid_at"))
                        if before_time and after_time and low <= (after_time - before_time).total_seconds() / 86400 <= high:
                            matched.add(patient)
            return matched
        if operator == "patient_after_first":
            first_results = [self.run_group(group) for group in self.leaf_groups(node["first"])]
            after_results = [self.run_group(group) for group in self.leaf_groups(node["after"])]
            candidates = self.patient_intersection(first_results + after_results)
            matched: set[str] = set()
            for patient in candidates:
                first_dates = [
                    stamp for result in first_results for hit in result[patient]
                    if (stamp := parsed_time(hit["edge_data"].get("valid_at"))) is not None
                ]
                after_dates = [
                    stamp for result in after_results for hit in result[patient]
                    if (stamp := parsed_time(hit["edge_data"].get("valid_at"))) is not None
                ]
                if first_dates and any(stamp > min(first_dates) for stamp in after_dates):
                    matched.add(patient)
            return matched
        if operator == "same_case_count_distinct":
            raise UnsupportedQuery("reference_upper/count_distinct is unavailable in the original PGR fields")
        raise UnsupportedQuery(f"unsupported query operator: {operator}")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("pgr.sqlite"))
    parser.add_argument("--hub-threshold", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    answers = load_answers(args.answers)
    started = perf_counter()
    with args.graph.open("rb") as handle:
        graph = pickle.load(handle)
    graph_load_ms = (perf_counter() - started) * 1000
    retriever = PGRSQLRetriever(graph, args.db, hub_threshold=args.hub_threshold)
    engine = StructuredPGR(retriever, args.max_hops)
    print(
        f"graph_load={graph_load_ms:.2f}ms setup={retriever.stats.setup_ms:.2f}ms "
        f"hubs={retriever.stats.hubs} sql_rows={retriever.stats.sql_rows} "
        f"graph_edges={retriever.stats.graph_edges} max_hops={args.max_hops}"
    )
    line = "=" * 132
    print(line)
    print(
        f"{'问题':<6}{'基线正例':>10}{'检索数':>10}{'TP':>8}{'FP':>8}{'FN':>8}"
        f"{'准确率':>12}{'召回率':>12}{'F1':>12}{'中位时延ms':>14}{'P95时延ms':>14}"
    )
    print("-" * 132)
    total_tp = total_fp = total_fn = 0
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    unsupported: list[tuple[str, str]] = []
    for query in spec["queries"]:
        qid = query["id"]
        expected = answers.get(qid, set())
        try:
            for _ in range(args.warmup):
                engine.execute(query)
            samples: list[float] = []
            predicted: set[str] = set()
            for _ in range(args.runs):
                run_started = perf_counter()
                raw_predicted = engine.execute(query)
                samples.append((perf_counter() - run_started) * 1000)
                predicted = {patient_key(patient) for patient in raw_predicted}
        except UnsupportedQuery as error:
            print(
                f"{qid:<6}{len(expected):>10}{'-':>10}{'-':>8}{'-':>8}{'-':>8}"
                f"{'N/A':>12}{'N/A':>12}{'N/A':>12}{'-':>14}{'-':>14}"
            )
            unsupported.append((qid, str(error)))
            continue
        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        f1 = f1_score(precision, recall)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1_values.append(f1)
        median, tail = statistics.median(samples), percentile95(samples)
        print(
            f"{qid:<6}{len(expected):>10}{len(predicted):>10}{tp:>8}{fp:>8}{fn:>8}"
            f"{metric(precision):>12}{metric(recall):>12}{metric(f1):>12}{median:>14.2f}{tail:>14.2f}"
        )
    micro_precision = ratio(total_tp, total_tp + total_fp)
    micro_recall = ratio(total_tp, total_tp + total_fn)
    micro_f1 = f1_score(micro_precision, micro_recall)
    macro_precision = statistics.mean(precisions) if precisions else None
    macro_recall = statistics.mean(recalls) if recalls else None
    macro_f1 = statistics.mean(f1_values) if f1_values else None
    print("-" * 132)
    print(
        f"总体 micro: TP={total_tp} FP={total_fp} FN={total_fn}  "
        f"准确率 {metric(micro_precision)}  召回率 {metric(micro_recall)}  F1 {metric(micro_f1)}"
    )
    print(
        f"总体 macro(各题平均):  准确率 {metric(macro_precision)}  "
        f"召回率 {metric(macro_recall)}  F1 {metric(macro_f1)}"
    )
    for qid, reason in unsupported:
        print(f"{qid} 未统计: {reason}")
    print(line)
    retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
