"""Evaluation: precision/recall vs ground truth + per-patient ranking."""

from __future__ import annotations

from typing import Any

from .retrieval.base import RetrievalResult
from .synthetic_data import load_ground_truth


def evaluate(results: list[RetrievalResult], *, strict: bool = True, soft_threshold: float = 0.8) -> dict[str, Any]:
    gt = load_ground_truth()
    gt_strict = set(gt["strict_valid_patients"])

    # candidates that pass the mode
    m = max((r.coverage_count for r in results), default=0) or 1
    # determine m from results length is unreliable; use ground truth cond count
    cond_ids = ["C1", "C2", "C3", "C4", "C5"]
    m = len(cond_ids)

    if strict:
        returned = {r.patient_id for r in results if r.coverage_ratio >= 1.0 - 1e-9}
    else:
        returned = {r.patient_id for r in results if r.coverage_ratio >= soft_threshold - 1e-9}

    tp = len(returned & gt_strict)
    fp = len(returned - gt_strict)
    fn = len(gt_strict - returned)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    ranking = [{"patient_id": r.patient_id, "score": round(r.score, 4),
                "coverage": f"{r.coverage_count}/{m}", "strict_match": r.strict_match}
               for r in ranked]

    return {
        "algorithm": results[0].algorithm if results else "",
        "strict": strict,
        "soft_threshold": soft_threshold,
        "ground_truth_strict": sorted(gt_strict),
        "returned_patients": sorted(returned),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_returned": len(returned),
        "total_visited_nodes": max((r.visited_nodes for r in results), default=0),
        "total_visited_edges": max((r.visited_edges for r in results), default=0),
        "elapsed_ms": round(max((r.elapsed_ms for r in results), default=0.0), 3),
        "ranking": ranking,
    }
