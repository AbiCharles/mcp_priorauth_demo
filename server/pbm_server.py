from __future__ import annotations
import json, os, pathlib, re
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env from repo root or current cwd
load_dotenv()

# Resolve data directory (env override or default /data)
DEFAULT_BASE = pathlib.Path(__file__).resolve().parents[1] / "data"
BASE = pathlib.Path(os.getenv("PBM_DATA_DIR") or DEFAULT_BASE)

FORMULARY_PATH = BASE / "formulary.json"
GUIDELINES_DIR = BASE / "guidelines"
PRECEDENTS_PATH = BASE / "precedents.jsonl"

mcp = FastMCP("PharmacyBenefitsMCP")

def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())

# ---------------- RESOURCES ----------------

@mcp.resource("pbm://formulary/{plan}")
def read_formulary(plan: str) -> str:
    data = _load_json(FORMULARY_PATH)
    plans = data.get("plans", {})
    if plan == "ALL":
        return json.dumps(plans, indent=2)
    return json.dumps(plans.get(plan, {}), indent=2)

@mcp.resource("pbm://guideline/{topic}")
def read_guideline(topic: str) -> str:
    p = GUIDELINES_DIR / f"{topic}.md"
    return p.read_text() if p.exists() else f"# Guideline '{topic}' not found."

@mcp.resource("pbm://precedent/{drug}")
def read_precedents(drug: str) -> str:
    items: List[Dict[str, Any]] = []
    if PRECEDENTS_PATH.exists():
        with PRECEDENTS_PATH.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("drug", "").lower() == drug.lower():
                        items.append(rec)
                except json.JSONDecodeError:
                    continue
    return json.dumps(items, indent=2)

# --------------- SMALL UTIL TOOLS ---------------

@mcp.tool(title="list_plans")
def list_plans() -> List[str]:
    data = _load_json(FORMULARY_PATH)
    return sorted(list(data.get("plans", {}).keys()))

@mcp.tool(title="list_drugs")
def list_drugs(plan: str) -> List[str]:
    data = _load_json(FORMULARY_PATH)
    pa = data.get("plans", {}).get(plan, {}).get("PA", {})
    return sorted(list(pa.keys()))

# --------------- EXPLAINABLE EVALUATOR ---------------

@mcp.tool(title="evaluate_pa")
def evaluate_pa(
    plan: str,
    drug: str,
    diagnosis_text: str,
    patient: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Returns an explainable decision:
      - decision_code: APPROVE | DENY
      - outcome: one line
      - reasons: list[str]
      - criteria_evaluation: [{criterion, met, note}]
      - missing: list[str]
      - step_therapy_* and exclusions info
      - references (plan, version, limits)
    """
    data = _load_json(FORMULARY_PATH)
    plan_blob = data.get("plans", {}).get(plan, {})
    version = data.get("version")
    policy = plan_blob.get("PA", {}).get(drug, {})
    plan_type = plan_blob.get("planType")

    def has_phrase(needle: str, hay: List[str]) -> bool:
        return any(needle.lower() in str(s).lower() for s in hay)

    criteria_results: List[Dict[str, Any]] = []
    missing: List[str] = []
    reasons: List[str] = []

    # -------------------- ADDED: hard requirements + sane thresholds --------------------
    # Minimal, self-contained checks that run before policy parsing.
    AGE_MIN = 1                      # treat <1 or None as missing/invalid
    A1C_MIN_VALID, A1C_MAX_VALID = 4.0, 15.0
    BMI_MIN_VALID, BMI_MAX_VALID = 10.0, 80.0
    T2DM_A1C_THRESHOLD = 7.5
    OBESITY_BMI_PRIMARY = 30.0       # ≥30 OR ≥27 + comorbidity
    OBESITY_BMI_SECONDARY = 27.0

    diag_texts = [diagnosis_text] + list(patient.get("diagnoses", []) or [])
    age = patient.get("age", None)
    a1c = (patient.get("labs", {}) or {}).get("A1c", None)
    bmi = (patient.get("labs", {}) or {}).get("BMI", None)
    comorbid = patient.get("comorbidities", []) or []

    required_missing = []
    if not any(str(x).strip() for x in diag_texts):
        required_missing.append("Diagnosis text")
    if age is None or int(age) < AGE_MIN:
        required_missing.append("Age")
    if a1c is None:
        required_missing.append("A1C")
    if bmi is None:
        required_missing.append("BMI")

    if required_missing:
        for r in required_missing:
            criteria_results.append({"criterion": f"{r} provided", "met": False, "note": f"{r} is missing or invalid"})
        missing.extend([f"Provide {r}" for r in required_missing])

    # Valid ranges
    a1c_ok = False
    if a1c is not None:
        try:
            a1c_val = float(a1c)
        except Exception:
            a1c_val = -1.0
        a1c_ok = (A1C_MIN_VALID <= a1c_val <= A1C_MAX_VALID)
        criteria_results.append({
            "criterion": f"A1C within valid range ({A1C_MIN_VALID}-{A1C_MAX_VALID})",
            "met": bool(a1c_ok),
            "note": f"A1C={a1c}"
        })
        if not a1c_ok:
            missing.append(f"A1C must be between {A1C_MIN_VALID} and {A1C_MAX_VALID}")

    bmi_ok = False
    if bmi is not None:
        try:
            bmi_val = float(bmi)
        except Exception:
            bmi_val = -1.0
        bmi_ok = (BMI_MIN_VALID <= bmi_val <= BMI_MAX_VALID)
        criteria_results.append({
            "criterion": f"BMI within valid range ({BMI_MIN_VALID}-{BMI_MAX_VALID})",
            "met": bool(bmi_ok),
            "note": f"BMI={bmi}"
        })
        if not bmi_ok:
            missing.append(f"BMI must be between {BMI_MIN_VALID} and {BMI_MAX_VALID}")

    # Indication-specific thresholds
    lower_diag = " ".join([str(x).lower() for x in diag_texts if x]).strip()
    is_obesity = ("obesity" in lower_diag) or drug.lower().startswith("wegovy")
    is_t2dm = ("type 2" in lower_diag) or ("t2dm" in lower_diag) or ("e11" in lower_diag)

    if is_obesity and bmi is not None:
        obesity_met = (bmi_val >= OBESITY_BMI_PRIMARY) or (bmi_val >= OBESITY_BMI_SECONDARY and len(comorbid) > 0)
        criteria_results.append({
            "criterion": f"Obesity BMI policy (≥{OBESITY_BMI_PRIMARY} or ≥{OBESITY_BMI_SECONDARY} + comorbidity)",
            "met": bool(obesity_met),
            "note": f"BMI={bmi_val if bmi is not None else bmi}, comorbidities={comorbid}"
        })
        if not obesity_met:
            missing.append(
                f"Obesity BMI policy not met (BMI≥{OBESITY_BMI_PRIMARY} or BMI≥{OBESITY_BMI_SECONDARY} with comorbidity)"
            )

    if is_t2dm and a1c is not None:
        t2dm_met = (a1c_val >= T2DM_A1C_THRESHOLD)
        criteria_results.append({
            "criterion": f"T2DM A1C threshold (≥{T2DM_A1C_THRESHOLD})",
            "met": bool(t2dm_met),
            "note": f"A1C={a1c_val if a1c is not None else a1c}"
        })
        if not t2dm_met:
            missing.append(f"A1C must be ≥ {T2DM_A1C_THRESHOLD} for T2DM policy")
    # -------------------- end ADDED block --------------------

    for c in policy.get("criteria", []):
        ok, note = True, ""
        cl = c.lower()

        if "age" in cl and ">=" in cl:
            m = re.search(r"age\s*>=\s*(\d+)", cl)
            age_val = int(patient.get("age", 0))
            thr = int(m.group(1)) if m else 0
            passed = age_val >= thr
            ok &= passed
            note = f"age={age_val} (need ≥ {thr})"
            if not passed: missing.append("Age threshold not met")

        if "diagnosis" in cl:
            if "type 2" in cl:
                passed = has_phrase("type 2 diabetes", patient.get("diagnoses", []))
            elif "obesity" in cl or "e66" in cl:
                passed = has_phrase("obesity", patient.get("diagnoses", [])) or has_phrase("e66", patient.get("diagnoses", []))
            else:
                passed = has_phrase("diagnosis", patient.get("diagnoses", []))
            ok &= passed
            note += ("; " if note else "") + f"diagnoses={patient.get('diagnoses')}"
            if not passed: missing.append("Diagnosis does not match policy")

        if "a1c" in cl and ">=" in cl:
            m = re.search(r"a1c\s*>=\s*(\d+(?:\.\d+)?)", cl)
            a1c_cur = patient.get("labs", {}).get("A1c")
            thr = float(m.group(1)) if m else None
            if a1c_cur is None or thr is None:
                ok = False
                note += ("; " if note else "") + "A1c missing"
                missing.append("Add recent A1c value")
            else:
                passed = float(a1c_cur) >= thr
                ok &= passed
                note += ("; " if note else "") + f"A1c={a1c_cur} (need ≥ {thr})"
                if not passed: missing.append(f"A1c must be ≥ {thr}")

        if "bmi" in cl:
            bmi_cur = patient.get("labs", {}).get("BMI")
            if "or bmi >=" in cl:
                if bmi_cur is None:
                    ok = False
                    note += ("; " if note else "") + "BMI missing"
                    missing.append("Add BMI and/or comorbidity")
                else:
                    passed = (bmi_cur >= 30) or (bmi_cur >= 27 and len(patient.get("comorbidities", [])) > 0)
                    ok &= passed
                    note += ("; " if note else "") + f"BMI={bmi_cur}, comorbidities={patient.get('comorbidities')}"
                    if not passed: missing.append("BMI/comorbidity threshold not met")

        if "lifestyle" in cl:
            months = int(patient.get("lifestyle_months", 0))
            passed = months >= 3
            ok &= passed
            note += ("; " if note else "") + f"lifestyle_months={months} (need ≥ 3)"
            if not passed: missing.append("Document ≥3 months lifestyle program")

        if "tried/failed" in cl:
            tried = [s.lower() for s in patient.get("tried_failed", [])]
            need_met = "metformin" in cl
            need_other = "additional" in cl or "either" in cl
            synonyms = {
                "empagliflozin": "sglt2", "dapagliflozin": "sglt2", "canagliflozin": "sglt2",
                "sulfonylurea": "su", "glipizide": "su", "glimepiride": "su",
                "sitagliptin": "dpp-4", "linagliptin": "dpp-4", "alogliptin": "dpp-4"
            }
            classes = {synonyms.get(x, x) for x in tried}
            other_ok = any(cls in classes for cls in ["sglt2", "dpp-4", "su"])
            passed = ((not need_met or "metformin" in classes) and (not need_other or other_ok))
            ok &= passed
            note += ("; " if note else "") + f"tried_failed={patient.get('tried_failed')}"
            if not passed: missing.append("Step therapy: need metformin + one additional class")

        if "contraindication" in cl:
            c_list = [s.lower() for s in patient.get("contraindications", [])]
            passed = len(c_list) == 0
            ok &= passed
            note += ("; " if note else "") + f"contraindications={patient.get('contraindications')}"
            if not passed: reasons.append("Safety: contraindication present")

        if "pancreatitis" in cl:
            c_list = [s.lower() for s in patient.get("contraindications", [])]
            passed = ("pancreatitis" not in c_list)
            ok &= passed
            note += ("; " if note else "") + f"contraindications={patient.get('contraindications')}"
            if not passed: reasons.append("Safety: active pancreatitis")

        criteria_results.append({"criterion": c, "met": bool(ok), "note": note})

    # Step therapy groups
    step_groups = policy.get("step_therapy", [])
    tried_lower = {s.strip().lower() for s in patient.get("tried_failed", [])}
    def group_ok(g: str) -> bool:
        if "|" in g:
            return any(opt.strip().lower() in tried_lower for opt in g.split("|"))
        return g.strip().lower() in tried_lower
    step_ok = all(group_ok(g) for g in step_groups) if step_groups else True
    if not step_ok:
        missing.append(f"Step therapy groups not satisfied: {step_groups}")

    # Exclusions
    exclusions = policy.get("exclusions", [])
    exclusions_hit = []
    for ex in exclusions:
        if "type 1" in ex.lower() and has_phrase("type 1", patient.get("diagnoses", [])):
            exclusions_hit.append(ex)
        if "medullary thyroid carcinoma" in ex.lower() and has_phrase("medullary", patient.get("contraindications", [])):
            exclusions_hit.append(ex)
        if "pregnancy" in ex.lower() and has_phrase("pregnan", patient.get("contraindications", [])):
            exclusions_hit.append(ex)

    # Final
    criteria_all_met = all(r["met"] for r in criteria_results)
    meets_criteria = criteria_all_met and step_ok and (len(exclusions_hit) == 0)

    if meets_criteria:
        decision_code = "APPROVE"
        outcome = "Meets all formulary criteria and step therapy; no exclusions present."
        reasons.insert(0, "All criteria satisfied")
        if step_groups:
            reasons.append(f"Step therapy satisfied: {step_groups}")
    else:
        decision_code = "DENY"
        if not missing and (not step_ok or exclusions_hit):
            if not step_ok: missing.append("Step therapy not met")
            if exclusions_hit: reasons.append(f"Exclusion triggered: {', '.join(exclusions_hit)}")
        outcome = "Does not meet criteria — insufficient evidence to satisfy one or more formulary criteria."

    references = {
        "plan": plan,
        "plan_type": plan_type,
        "policy_version": version,
        "policy_drug": drug,
        "quantity_limit": policy.get("quantity_limit"),
        "exclusions": exclusions
    }

    rationale_bullets = []
    for row in criteria_results:
        emoji = "✅" if row["met"] else "❌"
        rationale_bullets.append(f"{emoji} {row['criterion']} — {row['note']}")

    return {
        "plan": plan,
        "plan_type": plan_type,
        "drug": drug,
        "diagnosis_text": diagnosis_text,
        "criteria_evaluation": criteria_results,
        "criteria_all_met": bool(criteria_all_met),
        "step_therapy_required": step_groups,
        "step_therapy_satisfied": bool(step_ok),
        "quantity_limit": policy.get("quantity_limit"),
        "exclusions": exclusions,
        "exclusions_hit": exclusions_hit,
        "decision_code": decision_code,
        "outcome": outcome,
        "meets_criteria": bool(meets_criteria),
        "missing": list(dict.fromkeys(missing)),
        "reasons": reasons,
        "rationale_bullets": rationale_bullets,
        "references": references
    }

if __name__ == "__main__":
    mcp.run()
