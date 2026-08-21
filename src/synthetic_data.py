"""Synthetic graph generation for Patient-Centered Graph Retrieval.

All data is synthetic and reproducible (RANDOM_SEED=42). The graph is
constructed *by design* (not randomly) so that the ground-truth patient
cohort is explicit and testable.

Ground-truth design (default query: C1 Hypertension, C2 SBP>140, C3 Diabetes,
C4 HbA1c>7.0, C5 Amlodipine):

    P001  5/5  all 1-hop direct                    STRICT
    P002  4/5  no Amlodipine
    P003  4/5  SBP=132 (numeric fail)
    P004  5/5  Diabetes 3-hop, HbA1c 3-hop         STRICT  (multi-hop)
    P005  2/5  only Hypertension + SBP
    P006  3/5  HbA1c=6.1 (numeric fail)
    P007  5/5  Diabetes 2-hop, HbA1c 3-hop         STRICT
    P008  5/5  C1/C3/C5 via high-degree hub        STRICT (hub penalty demo)
"""

from __future__ import annotations

import json
import os
from typing import Any

RANDOM_SEED = 42
MAX_HOPS = 4

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ---------------------------------------------------------------------------
# Node type constants
# ---------------------------------------------------------------------------
PATIENT = "patient"
DISEASE = "disease"
INDICATOR = "indicator"
DRUG = "drug"
INTERMEDIATE = "intermediate"

PATIENTS = [f"P{i:03d}" for i in range(1, 9)]

DISEASES = [
    "Hypertension",
    "Diabetes",
    "Chronic Kidney Disease",
    "Heart Failure",
    "Hyperlipidemia",
]

INDICATORS = [
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "HbA1c",
    "BMI",
    "LDL",
    "Creatinine",
]

DRUGS = [
    "Amlodipine",
    "Metformin",
    "Lisinopril",
    "Atorvastatin",
    "Insulin",
]

# Intermediate nodes (some shared categories, some per-patient chains)
INTERMEDIATES = [
    "MetabolicCondition",
    "DiagnosisCategory",
    "CardiovascularCondition",
    "TreatmentEpisode",
    "MedicationOrder_P4",
    "LabPanel_P4",
    "Encounter_P4",
    "LabPanel_P7",
    "Encounter_P7",
    "CommonClinicalConcept",   # high-degree hub
    "CommonEncounter",         # second hop of hub chain
]


def _node(node_id: str, node_type: str, **extra: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, **extra}


def build_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for p in PATIENTS:
        nodes.append(_node(p, PATIENT))
    for d in DISEASES:
        nodes.append(_node(d, DISEASE))
    for ind in INDICATORS:
        nodes.append(_node(ind, INDICATOR))
    for dr in DRUGS:
        nodes.append(_node(dr, DRUG))
    for it in INTERMEDIATES:
        nodes.append(_node(it, INTERMEDIATE))
    return nodes


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------
def _edge(src: str, tgt: str, relation_type: str, **attrs: Any) -> dict[str, Any]:
    e = {"source": src, "target": tgt, "relation_type": relation_type}
    e.update(attrs)
    return e


def build_edges() -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []

    # === C1 Hypertension (categorical diagnosis) =========================
    # P001-P007 direct 1-hop; P008 via hub (3-hop)
    for p in ["P001", "P002", "P003", "P004", "P005", "P006", "P007"]:
        edges.append(_edge("Hypertension", p, "diagnosis"))
    # P008 hub path: Hypertension -> CommonClinicalConcept -> CommonEncounter -> P008
    edges.append(_edge("Hypertension", "CommonClinicalConcept", "concept_link"))
    edges.append(_edge("CommonClinicalConcept", "CommonEncounter", "concept_to_encounter"))
    edges.append(_edge("CommonEncounter", "P008", "encounter_patient"))

    # === C2 Systolic Blood Pressure > 140 (numeric, 1-hop direct) =========
    edges.append(_edge("Systolic Blood Pressure", "P001", "measurement", value=158, unit="mmHg", time="2026-01-12"))
    edges.append(_edge("Systolic Blood Pressure", "P002", "measurement", value=151, unit="mmHg", time="2026-01-20"))
    edges.append(_edge("Systolic Blood Pressure", "P003", "measurement", value=132, unit="mmHg", time="2026-01-05"))  # FAIL
    edges.append(_edge("Systolic Blood Pressure", "P004", "measurement", value=148, unit="mmHg", time="2026-01-18"))
    edges.append(_edge("Systolic Blood Pressure", "P005", "measurement", value=155, unit="mmHg", time="2026-01-22"))
    edges.append(_edge("Systolic Blood Pressure", "P006", "measurement", value=145, unit="mmHg", time="2026-01-09"))
    edges.append(_edge("Systolic Blood Pressure", "P007", "measurement", value=160, unit="mmHg", time="2026-01-14"))
    edges.append(_edge("Systolic Blood Pressure", "P008", "measurement", value=152, unit="mmHg", time="2026-01-16"))

    # === C3 Diabetes =====================================================
    # P001,P002,P003,P006 direct; P004 3-hop; P007 2-hop; P008 via hub 3-hop
    for p in ["P001", "P002", "P003", "P006"]:
        edges.append(_edge("Diabetes", p, "diagnosis"))
    # P004: Diabetes -> MetabolicCondition -> Encounter_P4 -> P004  (3-hop)
    edges.append(_edge("Diabetes", "MetabolicCondition", "diagnosis_category"))
    edges.append(_edge("MetabolicCondition", "Encounter_P4", "has_encounter"))
    edges.append(_edge("Encounter_P4", "P004", "encounter_patient"))
    # P007: Diabetes -> DiagnosisCategory -> P007  (2-hop)
    edges.append(_edge("Diabetes", "DiagnosisCategory", "diagnosis_category"))
    edges.append(_edge("DiagnosisCategory", "P007", "category_patient"))
    # P008: Diabetes -> CommonClinicalConcept -> CommonEncounter -> P008 (hub, shared with C1/C5)
    edges.append(_edge("Diabetes", "CommonClinicalConcept", "concept_link"))

    # === C4 HbA1c > 7.0 (numeric) ========================================
    # P001,P002,P003,P008 direct; P004 3-hop; P006 FAIL 1-hop; P007 3-hop
    edges.append(_edge("HbA1c", "P001", "lab_measurement", value=8.2, unit="%", time="2026-01-15"))
    edges.append(_edge("HbA1c", "P002", "lab_measurement", value=7.8, unit="%", time="2026-01-21"))
    edges.append(_edge("HbA1c", "P003", "lab_measurement", value=8.5, unit="%", time="2026-01-06"))
    edges.append(_edge("HbA1c", "P006", "lab_measurement", value=6.1, unit="%", time="2026-01-10"))  # FAIL
    edges.append(_edge("HbA1c", "P008", "lab_measurement", value=7.5, unit="%", time="2026-01-17"))
    # P004: HbA1c -> LabPanel_P4 -> Encounter_P4 -> P004  (3-hop, shares Encounter_P4->P004 with C3)
    edges.append(_edge("HbA1c", "LabPanel_P4", "lab_measurement", value=7.9, unit="%", time="2026-01-19"))
    edges.append(_edge("LabPanel_P4", "Encounter_P4", "lab_to_encounter"))
    # (Encounter_P4 -> P004 already added in C3 section - shared edge)
    # P007: HbA1c -> LabPanel_P7 -> Encounter_P7 -> P007  (3-hop)
    edges.append(_edge("HbA1c", "LabPanel_P7", "lab_measurement", value=8.0, unit="%", time="2026-01-13"))
    edges.append(_edge("LabPanel_P7", "Encounter_P7", "lab_to_encounter"))
    edges.append(_edge("Encounter_P7", "P007", "encounter_patient"))

    # === C5 Amlodipine (categorical drug) =================================
    # P001,P003,P007 direct; P004 2-hop; P008 via hub 3-hop
    for p in ["P001", "P003", "P007"]:
        edges.append(_edge("Amlodipine", p, "medication"))
    # P004: Amlodipine -> MedicationOrder_P4 -> P004  (2-hop)
    edges.append(_edge("Amlodipine", "MedicationOrder_P4", "medication_order"))
    edges.append(_edge("MedicationOrder_P4", "P004", "order_patient"))
    # P008: Amlodipine -> CommonClinicalConcept -> CommonEncounter -> P008 (hub)
    edges.append(_edge("Amlodipine", "CommonClinicalConcept", "concept_link"))

    # === Hub degree inflation for CommonClinicalConcept ===================
    # Many nodes link into the hub to inflate its degree (does not create
    # unintended patient matches because the hub only reaches P008).
    for node in [
        "Chronic Kidney Disease",
        "Heart Failure",
        "Hyperlipidemia",
        "Diastolic Blood Pressure",
        "BMI",
        "LDL",
        "Creatinine",
        "Metformin",
        "Lisinopril",
        "Atorvastatin",
        "Insulin",
        "CardiovascularCondition",
        "TreatmentEpisode",
        "DiagnosisCategory",
    ]:
        edges.append(_edge(node, "CommonClinicalConcept", "concept_link"))
    # A few out-edges from the hub to further raise degree (no patient targets)
    edges.append(_edge("CommonClinicalConcept", "TreatmentEpisode", "concept_link"))
    edges.append(_edge("CommonClinicalConcept", "CardiovascularCondition", "concept_link"))
    edges.append(_edge("CommonEncounter", "TreatmentEpisode", "encounter_episode"))

    # === Enrichment noise (non-query conditions on patients) =============
    # These add realistic structure but do NOT affect the 5 query conditions.
    edges.append(_edge("Chronic Kidney Disease", "P001", "diagnosis"))
    edges.append(_edge("Heart Failure", "P003", "diagnosis"))
    edges.append(_edge("Hyperlipidemia", "P002", "diagnosis"))
    edges.append(_edge("Chronic Kidney Disease", "P006", "diagnosis"))
    edges.append(_edge("Heart Failure", "P008", "diagnosis"))
    edges.append(_edge("Hyperlipidemia", "P004", "diagnosis"))
    edges.append(_edge("Metformin", "P001", "medication"))
    edges.append(_edge("Metformin", "P002", "medication"))
    edges.append(_edge("Metformin", "P006", "medication"))
    edges.append(_edge("Lisinopril", "P003", "medication"))
    edges.append(_edge("Lisinopril", "P007", "medication"))
    edges.append(_edge("Atorvastatin", "P004", "medication"))
    edges.append(_edge("Atorvastatin", "P005", "medication"))
    edges.append(_edge("Insulin", "P006", "medication"))
    edges.append(_edge("Diastolic Blood Pressure", "P001", "measurement", value=96, unit="mmHg", time="2026-01-12"))
    edges.append(_edge("Diastolic Blood Pressure", "P005", "measurement", value=94, unit="mmHg", time="2026-01-22"))
    edges.append(_edge("BMI", "P005", "measurement", value=31, unit="kg/m2", time="2026-01-22"))
    edges.append(_edge("BMI", "P002", "measurement", value=29, unit="kg/m2", time="2026-01-20"))
    edges.append(_edge("LDL", "P002", "measurement", value=160, unit="mg/dL", time="2026-01-21"))
    edges.append(_edge("LDL", "P004", "measurement", value=145, unit="mg/dL", time="2026-01-18"))
    edges.append(_edge("Creatinine", "P006", "measurement", value=1.8, unit="mg/dL", time="2026-01-09"))

    return edges


# ---------------------------------------------------------------------------
# Default query conditions
# ---------------------------------------------------------------------------
def build_default_conditions() -> list[dict[str, Any]]:
    return [
        {"id": "C1", "anchor": "Hypertension", "kind": "categorical"},
        {"id": "C2", "anchor": "Systolic Blood Pressure", "kind": "numeric",
         "operator": ">", "value": 140, "unit": "mmHg"},
        {"id": "C3", "anchor": "Diabetes", "kind": "categorical"},
        {"id": "C4", "anchor": "HbA1c", "kind": "numeric",
         "operator": ">", "value": 7.0, "unit": "%"},
        {"id": "C5", "anchor": "Amlodipine", "kind": "categorical"},
    ]


# ---------------------------------------------------------------------------
# Ground truth (explicit, pre-declared)
# ---------------------------------------------------------------------------
def build_ground_truth() -> dict[str, Any]:
    gt: dict[str, dict[str, bool]] = {}
    table = {
        "P001": [True, True, True, True, True],   # 5/5
        "P002": [True, True, True, True, False],   # 4/5  no Amlodipine
        "P003": [True, False, True, True, True],   # 4/5  SBP=132 fail
        "P004": [True, True, True, True, True],    # 5/5  multi-hop
        "P005": [True, True, False, False, False],  # 2/5
        "P006": [True, True, True, False, False],   # 3/5  HbA1c=6.1 fail
        "P007": [True, True, True, True, True],    # 5/5  multi-hop
        "P008": [True, True, True, True, True],    # 5/5  via hub
    }
    cond_ids = ["C1", "C2", "C3", "C4", "C5"]
    for pid, flags in table.items():
        gt[pid] = {cid: f for cid, f in zip(cond_ids, flags)}
    gt["strict_valid_patients"] = ["P001", "P004", "P007", "P008"]
    return gt


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _ensure_data_files() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    graph = {"nodes": build_nodes(), "edges": build_edges()}
    with open(os.path.join(DATA_DIR, "synthetic_graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DATA_DIR, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(build_default_conditions(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(DATA_DIR, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(build_ground_truth(), f, indent=2, ensure_ascii=False)


def load_graph_data() -> dict[str, Any]:
    _ensure_data_files()
    with open(os.path.join(DATA_DIR, "synthetic_graph.json"), encoding="utf-8") as f:
        return json.load(f)


def load_conditions() -> list[dict[str, Any]]:
    _ensure_data_files()
    with open(os.path.join(DATA_DIR, "conditions.json"), encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth() -> dict[str, Any]:
    _ensure_data_files()
    with open(os.path.join(DATA_DIR, "ground_truth.json"), encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    _ensure_data_files()
    nodes = build_nodes()
    edges = build_edges()
    gt = build_ground_truth()
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print("\nGround truth (default 5 conditions):")
    cids = ["C1", "C2", "C3", "C4", "C5"]
    print(f"{'Patient':<8}" + "".join(f"{c:<5}" for c in cids) + "cov")
    for pid in PATIENTS:
        row = gt[pid]
        flags = "  ".join("Y" if row[c] else "-" for c in cids)
        cov = sum(row[c] for c in cids)
        print(f"{pid:<8}{flags}   {cov}/5")
    print(f"\nstrict_valid_patients: {gt['strict_valid_patients']}")
    # report hub degree
    from collections import Counter
    deg = Counter()
    for e in edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    print(f"\nCommonClinicalConcept degree: {deg['CommonClinicalConcept']}")
    print(f"CommonEncounter degree: {deg['CommonEncounter']}")
