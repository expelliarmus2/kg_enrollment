"""FastAPI backend for the Patient-Centered Graph Retrieval demo.

Run with:
    uvicorn src.api:app --reload --port 8000
or:
    python -m src.api
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .graph_model import build_graph, all_edges, degree
from .synthetic_data import load_conditions, load_ground_truth, build_ground_truth
from .constraints import describe_filter, build_query_conditioned_graph
from .retrieval import retrieve
from .evaluation import evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Patient-Centered Graph Retrieval")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Build graph once at startup
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


class RetrieveRequest(BaseModel):
    algorithm: str = "bfs"
    max_hops: int = 4
    strict: bool = True
    soft_threshold: float = 0.8
    rho: float = 0.0
    condition_prize: float = 10.0
    conditions: list[dict[str, Any]] | None = None


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/graph")
def api_graph():
    G = get_graph()
    nodes = [{"id": n, **d} for n, d in G.nodes(data=True)]
    for nd in nodes:
        nd["degree"] = degree(G, nd["id"])
    edges = all_edges(G)
    return {"nodes": nodes, "edges": edges, "max_hops_default": 4}


@app.get("/api/conditions")
def api_conditions():
    return {"conditions": load_conditions()}


@app.get("/api/ground_truth")
def api_ground_truth():
    return load_ground_truth()


@app.get("/api/filter_report")
def api_filter_report():
    G = get_graph()
    conds = load_conditions()
    return {"report": describe_filter(G, conds)}


@app.post("/api/retrieve")
def api_retrieve(req: RetrieveRequest):
    G = get_graph()
    conds = req.conditions if req.conditions else load_conditions()
    results = retrieve(
        G, conds,
        algorithm=req.algorithm,
        max_hops=req.max_hops,
        strict=req.strict,
        soft_threshold=req.soft_threshold,
        rho=req.rho,
        condition_prize=req.condition_prize,
    )
    ev = evaluate(results, strict=req.strict, soft_threshold=req.soft_threshold)
    # sort by score desc, keep all (even 0-coverage) so UI can show table
    payload = [r.to_dict() for r in sorted(results, key=lambda r: r.score, reverse=True)]
    return {
        "algorithm": req.algorithm,
        "results": payload,
        "evaluation": ev,
        "filter_report": describe_filter(G, conds),
        "params": {
            "max_hops": req.max_hops,
            "strict": req.strict,
            "soft_threshold": req.soft_threshold,
            "rho": req.rho,
            "condition_prize": req.condition_prize,
        },
    }


@app.post("/api/compare")
def api_compare(req: RetrieveRequest):
    G = get_graph()
    conds = req.conditions if req.conditions else load_conditions()
    out: dict[str, Any] = {"per_algorithm": {}, "comparison_table": []}
    for algo in ["bfs", "pcst", "steiner"]:
        results = retrieve(
            G, conds,
            algorithm=algo,
            max_hops=req.max_hops,
            strict=req.strict,
            soft_threshold=req.soft_threshold,
            rho=req.rho,
            condition_prize=req.condition_prize,
        )
        ev = evaluate(results, strict=req.strict, soft_threshold=req.soft_threshold)
        out["per_algorithm"][algo] = {
            "evaluation": ev,
            "results": [r.to_dict() for r in sorted(results, key=lambda r: r.score, reverse=True)],
        }
    # build comparison table per patient
    gt = load_ground_truth()
    gt_strict = set(gt["strict_valid_patients"])
    patients = sorted({r["patient_id"] for algo in out["per_algorithm"].values() for r in algo["results"]})
    for pid in patients:
        row = {"patient": pid, "ground_truth": pid in gt_strict}
        for algo in ["bfs", "pcst", "steiner"]:
            res = next((r for r in out["per_algorithm"][algo]["results"] if r["patient_id"] == pid), None)
            if res:
                rank_list = [r["patient_id"] for r in out["per_algorithm"][algo]["results"]]
                rank = rank_list.index(pid) + 1
                row[algo] = {
                    "coverage": f"{res['coverage_count']}/5",
                    "score": round(res["score"], 3),
                    "strict": res["strict_match"],
                    "rank": rank,
                    "visited_nodes": res["visited_nodes"],
                    "visited_edges": res["visited_edges"],
                    "elapsed_ms": round(res["elapsed_ms"], 3),
                }
            else:
                row[algo] = None
        out["comparison_table"].append(row)
    out["params"] = {
        "max_hops": req.max_hops,
        "strict": req.strict,
        "soft_threshold": req.soft_threshold,
        "rho": req.rho,
    }
    return out


@app.get("/api/animate_search")
def api_animate_search(algorithm: str = "bfs", max_hops: int = 4, rho: float = 0.0):
    """Return per-condition BFS expansion fronts for search animation."""
    from collections import deque
    G = get_graph()
    conds = load_conditions()
    H = build_query_conditioned_graph(G, conds)
    frames: list[dict[str, Any]] = []
    for cond in conds:
        anchor = cond["anchor"]
        visited = {anchor}
        frontier = [anchor]
        steps = [{"condition": cond["id"], "anchor": anchor, "hop": 0, "frontier": [anchor], "patients": []}]
        queue = deque([(anchor, 0, [anchor])])
        while queue:
            level_size = len(queue)
            next_frontier = []
            next_patients = []
            for _ in range(level_size):
                node, dist, path = queue.popleft()
                if dist >= max_hops:
                    continue
                for _, nbr, k, edata in H.out_edges(node, keys=True, data=True):
                    if nbr in path or nbr in visited:
                        continue
                    visited.add(nbr)
                    next_frontier.append(nbr)
                    if H.nodes[nbr].get("type") == "patient":
                        next_patients.append(nbr)
                    queue.append((nbr, dist + 1, path + [nbr]))
            if next_frontier or next_patients:
                steps.append({
                    "condition": cond["id"],
                    "anchor": anchor,
                    "hop": len(steps),
                    "frontier": next_frontier,
                    "patients": next_patients,
                })
        frames.append({"condition": cond["id"], "anchor": anchor, "steps": steps})
    return {"frames": frames, "max_hops": max_hops, "algorithm": algorithm}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
