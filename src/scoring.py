"""Patient accumulator and scoring.

Each condition may match multiple patients.  We accumulate per-patient,
per-condition match results then compute coverage and a unified score.

Score (PCST / generic):
    Score = ALPHA * Coverage^GAMMA
            - BETA  * AvgDistance
            - ETA   * Imbalance
            - LAMBDA* GraphCost

Score (Steiner):
    Score = ALPHA * Coverage
            - BETA  * C(G_p)
            - ETA   * MaxDistance
"""

from __future__ import annotations

from typing import Any

ALPHA = 10.0
BETA = 1.0
ETA = 0.5
LAMBDA = 0.3
GAMMA = 2.0


def empty_condition_match() -> dict[str, Any]:
    return {
        "matched": False,
        "distance": None,
        "path": [],
        "cost": 0.0,
        "reason": None,
    }


def new_patient_accumulator(condition_ids: list[str]) -> dict[str, dict[str, Any]]:
    return {cid: empty_condition_match() for cid in condition_ids}


def coverage_count(acc: dict[str, dict[str, Any]]) -> int:
    return sum(1 for c in acc.values() if c["matched"])


def coverage_ratio(acc: dict[str, dict[str, Any]], m: int) -> float:
    return coverage_count(acc) / m if m else 0.0


def avg_distance(acc: dict[str, dict[str, Any]]) -> float:
    dists = [c["distance"] for c in acc.values() if c["matched"] and c["distance"] is not None]
    return sum(dists) / len(dists) if dists else 0.0


def max_distance(acc: dict[str, dict[str, Any]]) -> float:
    dists = [c["distance"] for c in acc.values() if c["matched"] and c["distance"] is not None]
    return max(dists) if dists else 0.0


def imbalance(acc: dict[str, dict[str, Any]]) -> float:
    dists = [c["distance"] for c in acc.values() if c["matched"] and c["distance"] is not None]
    if not dists:
        return 0.0
    avg = sum(dists) / len(dists)
    return sum(max(0.0, d - avg) for d in dists)


def graph_cost(acc: dict[str, dict[str, Any]]) -> float:
    return sum(c["cost"] for c in acc.values())


def score_generic(acc: dict[str, dict[str, Any]], m: int) -> float:
    """PCST-style score: reward coverage, penalise distance/imbalance/cost."""
    cov = coverage_ratio(acc, m)
    return (
        ALPHA * (cov ** GAMMA)
        - BETA * avg_distance(acc)
        - ETA * imbalance(acc)
        - LAMBDA * graph_cost(acc)
    )


def score_steiner(acc: dict[str, dict[str, Any]], m: int, union_cost: float) -> float:
    """Steiner keyword score: linear coverage, union graph cost, max dist."""
    cov = coverage_ratio(acc, m)
    return ALPHA * cov - BETA * union_cost - ETA * max_distance(acc)


def passes_mode(acc: dict[str, dict[str, Any]], m: int, *, strict: bool, soft_threshold: float = 0.8) -> bool:
    cov = coverage_ratio(acc, m)
    if strict:
        return cov >= 1.0 - 1e-9
    return cov >= soft_threshold - 1e-9
