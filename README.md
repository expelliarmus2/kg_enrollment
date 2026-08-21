# Patient-Centered Graph Retrieval

A small, fully-synthetic, interactive demo of **multi-condition patient cohort
retrieval** on a medical knowledge graph. Given several *condition anchor*
nodes (diseases, indicators, drugs), the system finds all `patient_id` roots
that are reachable from every anchor within `MAX_HOPS` hops, subject to
numeric *hard* constraints, and returns one **patient-centered subgraph**
per matching patient.

Three retrieval algorithms are implemented and compared on the **same** graph:

1. **BFS baseline** – bounded constrained BFS per anchor, then coverage intersection.
2. **PCST** – Prize-Collecting Steiner Tree (greedy prize/cost pruning).
3. **Steiner keyword search** – shortest valid path per anchor, unioned into `G_p`.

---

## Problem

Input: multiple medical condition anchors `C1 … Cm` (e.g. *Hypertension*,
*SBP>140*, *Diabetes*, *HbA1c>7*, *Amlodipine*).  
Goal: find **multiple** `patient_id` roots `p` such that

```
a_i  leadsto  p   with  |path_i| <= MAX_HOPS
```

and every numeric / temporal constraint on the *edge* is satisfied (hard filter,
not a soft penalty). Numeric values live on **edges**, indicator names live on
**nodes**:

```
Systolic Blood Pressure --[value=158]--> P001
```

Output: a ranked list of patient-centered subgraphs (one per candidate).

### Key design principles (all enforced)

| # | Principle |
|---|-----------|
| 1 | Patient is the **target root**, never a traversal hub. |
| 4 | 1-hop wins over multi-hop for the same (condition, patient). |
| 5 | Indicator name = node; value = edge attribute. |
| 7 | Failing numeric edges are **hard-removed before** any search. |
| 9 | Multi-hop (≤4) is the fallback when 1-hop can't reach. |
| 11 | PCST and Steiner run on **identical** input. |

---

## Synthetic data

`RANDOM_SEED = 42`, `MAX_HOPS = 4`, **8 patients** (P001–P008). The graph is
hand-designed (not random) so ground truth is explicit:

| Patient | C1 HTN | C2 SBP>140 | C3 Diabetes | C4 HbA1c>7 | C5 Amlodipine | Cov | Strict |
|---------|:------:|:----------:|:-----------:|:---------:|:-------------:|:---:|:------:|
| P001 | ✓ | 158 ✓ | ✓ | 8.2 ✓ | ✓ | 5/5 | ✓ |
| P002 | ✓ | 151 ✓ | ✓ | 7.8 ✓ | ✗ | 4/5 | |
| P003 | ✓ | 132 ✗ | ✓ | 8.5 ✓ | ✓ | 4/5 | |
| P004 | ✓ | 148 ✓ | 3-hop ✓ | 3-hop ✓ | 2-hop ✓ | 5/5 | ✓ |
| P005 | ✓ | 155 ✓ | ✗ | ✗ | ✗ | 2/5 | |
| P006 | ✓ | 145 ✓ | ✓ | 6.1 ✗ | ✗ | 3/5 | |
| P007 | ✓ | 160 ✓ | 2-hop ✓ | 3-hop ✓ | ✓ | 5/5 | ✓ |
| P008 | hub ✓ | 152 ✓ | hub ✓ | 7.5 ✓ | hub ✓ | 5/5 | ✓ |

`strict_valid_patients = [P001, P004, P007, P008]`

**P004** needs multi-hop for Diabetes (3-hop via `MetabolicCondition →
Encounter_P4`) and HbA1c (3-hop via `LabPanel_P4 → Encounter_P4`); its C3 and
C4 paths **share** the edge `Encounter_P4 → P004` (tests the "shared edge once"
rule).  
**P008** reaches C1/C3/C5 only through the high-degree hub
`CommonClinicalConcept` (degree ≈ 20) to demonstrate hub penalty.

---

## Algorithms

### BFS baseline
For each anchor, BFS up to `MAX_HOPS` on the query-conditioned graph; collect
reachable patients with distance/path; coverage intersection gives the cohort.

### PCST
Condition anchors carry `prize` (default 10). For each patient, find the
shortest valid path per anchor, then **greedily prune** any condition whose
full path cost exceeds its prize. Edge cost:
```
c(e) = 1 + ρ·log(1 + deg(target))
```
With hub penalty `ρ>0`, high-degree-hub paths become expensive and are dropped
(P008 → 2/5). Score:
```
Score = α·Coverage^γ − β·AvgDistance − η·Imbalance − λ·GraphCost
```

### Steiner keyword search
Each condition = keyword group / terminal. For each patient, find the shortest
valid path per anchor, union into `G_p` (shared edges counted **once**).
```
Score = α·Coverage − β·C(G_p) − η·MaxDistance
```

All three return a unified `RetrievalResult` and are called via:
```python
from src.retrieval import retrieve
results = retrieve(graph, conditions, algorithm="pcst", max_hops=4, rho=1.3)
```

---

## Project layout

```
patient_graph_retrieval/
├── data/                  # auto-generated JSON
│   ├── synthetic_graph.json
│   ├── conditions.json
│   └── ground_truth.json
├── src/
│   ├── synthetic_data.py  # graph + ground truth construction
│   ├── graph_model.py     # NetworkX multigraph builder + edge_cost
│   ├── constraints.py     # numeric hard filtering
│   ├── scoring.py         # accumulator + score formulas
│   ├── evaluation.py      # precision/recall vs ground truth
│   ├── api.py             # FastAPI app
│   └── retrieval/
│       ├── base.py        # RetrievalResult dataclass
│       ├── bfs.py
│       ├── pcst.py
│       └── steiner_keyword.py
├── static/                # interactive HTML UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── tests/                 # 24 pytest tests
```

---

## Run

```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
# or
python -m src.api
```

Open <http://localhost:8000>.

### Tests
```bash
python -m pytest tests/ -v      # 24 passed
python -m src.synthetic_data    # print graph + ground-truth summary
```

---

## Interactive UI

| Region | What it does |
|--------|--------------|
| **A** | Full synthetic graph (vis-network), styled by node type, hover = id/type/degree |
| **B** | Query panel: toggle conditions, edit `SBP`/`HbA1c` thresholds, `MAX_HOPS` slider, strict/soft, hub-penalty checkbox |
| **C** | Algorithm selector + Run / Compare / Animate |
| **D** | Candidate patient table (coverage, score, avg hops, graph cost) — click a row |
| **E** | Patient-centered retrieved subgraph (anchors + patient highlighted) |
| **F** | Filtered-edge report (which numeric edges passed/failed and why) |
| **G** | Algorithm comparison table + precision/recall/runtime metrics |

### Demos to try

- **Multi-hop value**: set `MAX_HOPS=1` → only P001 is strict (5/5). Set
  `MAX_HOPS=3` → P001/P004/P007/P008 all strict (P004/P007/P008 need 2–3 hops).
- **Numeric hard filter**: P003 (SBP=132) and P006 (HbA1c=6.1) never match —
  see region F.
- **Hub penalty**: enable it → PCST prunes P008's hub paths (→2/5) while
  Steiner keeps P008 at 5/5 but ranks it last (high cost).
- **Animate Search**: watch each condition expand hop-by-hop and light up
  reachable patients.
- **Compare**: side-by-side BFS/PCST/Steiner precision/recall/runtime.

---

## Scoring constants

```python
ALPHA = 10.0   BETA = 1.0   ETA = 0.5   LAMBDA = 0.3   GAMMA = 2.0
CONDITION_PRIZE = 10.0   HUB_RHO = 1.3 (when enabled)
```

All configurable via the API `RetrieveRequest` body.
