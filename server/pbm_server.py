"""
pbm_server.py — MCP server for Pharmacy Benefits Prior Authorization (PA) evaluation.

This server exposes:
- RESOURCES:
    • pbm://formulary/{plan}  -> JSON for a specific plan or ALL
    • pbm://guideline/{topic} -> Markdown guideline text for a topic
    • pbm://precedent/{drug}  -> JSONL precedents for a drug
- TOOLS:
    • list_plans()            -> List of plan names from formulary.json
    • list_drugs(plan)        -> List of PA-covered drugs for a plan
    • evaluate_pa(...)        -> Explainable evaluation for a prior auth request

The evaluation pipeline:
1) **Hard prechecks**: require Diagnosis text, Age, A1C, BMI. Validate numeric ranges for A1C and BMI.
2) **Indication thresholds**:
   - Type 2 Diabetes Mellitus (T2DM): A1C ≥ 7.5
   - Obesity: BMI ≥ 30 OR BMI ≥ 27 with ≥1 comorbidity
3) **Policy criteria** (from formulary.json "criteria"): age thresholds, diagnosis matches, A1C/BMI checks, step therapy,
   contraindications, etc.
4) **Step therapy groups** (policy.step_therapy): each group must be satisfied (supports OR with "a|b").
5) **Exclusions** (policy.exclusions): if triggered, will deny.
6) **Final decision** (APPROVE or DENY) with reasons, missing items, rationale bullets, and references.
"""

from __future__ import annotations
import json, os, pathlib, re
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Load environment variables (from repo root or current cwd)
load_dotenv()

# Resolve base data directory (default: <repo>/data)
DEFAULT_BASE = pathlib.Path(__file__).resolve().parents[1] / "data"
BASE = pathlib.Path(os.getenv("PBM_DATA_DIR") or DEFAULT_BASE)

# Core data paths
FORMULARY_PATH = BASE / "formulary.json"
GUIDELINES_DIR = BASE / "guidelines"
PRECEDENTS_PATH = BASE / "precedents.jsonl"

# Create MCP server instance
mcp = FastMCP("PharmacyBenefitsMCP")


def _load_json(path: pathlib.Path) -> dict:
    """Load and parse JSON from a file path."""
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# RESOURCES — static content endpoints (read-only)
# ---------------------------------------------------------------------------

@mcp.resource("pbm://formulary/{plan}")
def read_formulary(plan: str) -> str:
    """
    Return formulary JSON for a given plan; or ALL plans if plan == "ALL".

    Args:
        plan: Plan name or "ALL".

    Returns:
        Pretty-printed JSON string.
    """
    data = _load_json(FORMULARY_PATH)
    plans = data.get("plans", {})
    if plan == "ALL":
        return json.dumps(plans, indent=2)
    return json.dumps(plans.get(plan, {}), indent=2)


@mcp.resource("pbm://guideline/{topic}")
def read_guideline(topic: str) -> str:
    """
    Return Markdown guideline for a given topic if available.

    Args:
        topic: Guideline topic name.

    Returns:
        Markdown text; or a "not found" message.
    """
    p = GUIDELINES_DIR / f"{topic}.md"
    return p.read_text() if p.exists() else f"# Guideline '{topic}' not found."


@mcp.resource("pbm://precedent/{drug}")
def read_precedents(drug: str) -> str:
    """
    Return precedent cases (JSONL) filtered by drug.

    Args:
        drug: Drug name filter.

    Returns:
        Pretty-printed JSON string list of precedent dicts.
    """
    items: List[Dict[str, Any]] = []
    if PRECEDENTS_PATH.exists():
        with PRECEDENTS_PATH.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("drug", "").lower() == drug.lower():
                        items.append(rec)
                except json.JSONDecodeError:
                    # Skip corrupted lines
                    continue
    return json.dumps(items, indent=2)


# ---------------------------------------------------------------------------
# SMALL TOOLS — formulary lookup helpers
# ---------------------------------------------------------------------------

@mcp.tool(title="list_plans")
def list_plans() -> List[str]:
    """
    Return all available plan names from formulary.json.

    Returns:
        Sorted list of plan names.
    """
    data = _load_json(FORMULARY_PATH)
    return sorted(list(data.get("plans", {}).keys()))


@mcp.tool(title="list_drugs")
def list_drugs(plan: str) -> List[str]:
    """
    Return all PA drugs for a given plan.

    Args:
        plan: Plan name.

    Returns:
        Sorted list of drug names that have a PA policy under the plan.
    """
    data = _load_json(FORMULARY_PATH)
    pa = data.get("plans", {}).get(plan, {}).get("PA", {})
    return sorted(list(pa.keys()))


# ---------------------------------------------------------------------------
# MAIN TOOL — evaluate prior authorization (PA) request
# ---------------------------------------------------------------------------

@mcp.tool(title="evaluate_pa")
def evaluate_pa(
    plan: str,
    drug: str,
    diagnosis_text: str,
    patient: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a prior authorization request with explainable output.

    Args:
        plan: Insurance plan name.
        drug: Requested drug name.
        diagnosis_text: Clinical diagnosis free-text supplied with the request.
        patient: Dict with fields like:
            {
              "age": int,
              "diagnoses": [str, ...],
              "labs": {"A1c": number, "BMI": number},
              "comorbidities": [str, ...],
              "lifestyle_months": int,
              "tried_failed": [str, ...],
              "contraindications": [str, ...]
            }

    Returns:
        A dict with keys:
          - decision_code: "APPROVE" or "DENY"
          - outcome: one-line summary
          - reasons: list[str] with rationale
          - criteria_evaluation: list[{criterion, met, note}]
          - missing: list[str] of missing items / unmet conditions
          - step_therapy_required: list[str] (from policy)
          - step_therapy_satisfied: bool
          - exclusions / exclusions_hit
          - references: plan/plan_type/policy_version/policy_drug/quantity_limit/exclusions
          - rationale_bullets: user-friendly bullets of criteria results
    """
    # Load formulary data and policy for this plan+drug
    data = _load_json(FORMULARY_PATH)
    plan_blob = data.get("plans", {}).get(plan, {})
    version = data.get("version")
    policy = plan_blob.get("PA", {}).get(drug, {})
    plan_type = plan_blob.get("planType")

    def has_phrase(needle: str, hay: List[str]) -> bool:
        """Case-insensitive substring search across a string list."""
        return any(needle.lower() in str(s).lower() for s in hay)

    # Prepare containers for evaluation results
    criteria_results: List[Dict[str, Any]] = []
    missing: List[str] = []
    reasons: List[str] = []

    # -----------------------------------------------------------------------
    # 1) HARD PRECHECKS (always required)
    #    - Ensure mandatory fields are present
    #    - Validate numeric ranges (A1C, BMI)
    #    - Enforce indication thresholds (T2DM, Obesity)
    # -----------------------------------------------------------------------
    AGE_MIN = 10
    A1C_MIN_VALID, A1C_MAX_VALID = 4.0, 15.0      # physiological guardrails
    BMI_MIN_VALID, BMI_MAX_VALID = 13.0, 80.0     # physiological guardrails
    T2DM_A1C_THRESHOLD = 7.5                      # threshold for T2DM control
    OBESITY_BMI_PRIMARY = 30.0                    # ≥30 OR ≥27 with comorbidity
    OBESITY_BMI_SECONDARY = 27.0

    # Normalize input fields
    diag_texts = [diagnosis_text] + list(patient.get("diagnoses", []) or [])
    age = patient.get("age", None)
    a1c = (patient.get("labs", {}) or {}).get("A1c", None)
    bmi = (patient.get("labs", {}) or {}).get("BMI", None)
    comorbid = patient.get("comorbidities", []) or []

    # Required inputs must be provided
    required_missing = []
    if not any(str(x or "").strip() for x in diag_texts):
        required_missing.append("Diagnosis text")
    if age is None or int(age) < AGE_MIN:
        required_missing.append("Age")
    if a1c is None:
        required_missing.append("A1C")
    if bmi is None:
        required_missing.append("BMI")

    if required_missing:
        # Report each missing required item as a failed criterion
        for r in required_missing:
            criteria_results.append({
                "criterion": f"{r} provided",
                "met": False,
                "note": f"{r} missing"
            })
        missing.extend([f"Provide {r}" for r in required_missing])

    # Numeric range checks (add criteria rows whether pass/fail)
    a1c_ok, bmi_ok = False, False
    a1c_val, bmi_val = None, None

    if a1c is not None:
        try:
            a1c_val = float(a1c)
        except Exception:
            a1c_val = -1.0
        a1c_ok = (A1C_MIN_VALID <= a1c_val <= A1C_MAX_VALID)
        criteria_results.append({
            "criterion": f"A1C in {A1C_MIN_VALID}-{A1C_MAX_VALID}",
            "met": a1c_ok,
            "note": f"A1C={a1c}"
        })
        if not a1c_ok:
            missing.append(f"A1C must be {A1C_MIN_VALID}-{A1C_MAX_VALID}")

    if bmi is not None:
        try:
            bmi_val = float(bmi)
        except Exception:
            bmi_val = -1.0
        bmi_ok = (BMI_MIN_VALID <= bmi_val <= BMI_MAX_VALID)
        criteria_results.append({
            "criterion": f"BMI in {BMI_MIN_VALID}-{BMI_MAX_VALID}",
            "met": bmi_ok,
            "note": f"BMI={bmi}"
        })
        if not bmi_ok:
            missing.append(f"BMI must be {BMI_MIN_VALID}-{BMI_MAX_VALID}")

    # Indication identification and thresholds
    lower_diag = " ".join([str(x).lower() for x in diag_texts if x]).strip()
    is_obesity = ("obesity" in lower_diag) or drug.lower().startswith("wegovy")
    is_t2dm = ("type 2" in lower_diag) or ("t2dm" in lower_diag) or ("e11" in lower_diag)

    if is_obesity and bmi_val is not None:
        obesity_met = (bmi_val >= OBESITY_BMI_PRIMARY) or (bmi_val >= OBESITY_BMI_SECONDARY and len(comorbid) > 0)
        criteria_results.append({
            "criterion": "Obesity BMI policy (≥30 OR ≥27 with comorbidity)",
            "met": obesity_met,
            "note": f"BMI={bmi_val}, comorbidities={comorbid}"
        })
        if not obesity_met:
            missing.append("Obesity BMI policy not met")

    if is_t2dm and a1c_val is not None:
        t2dm_met = (a1c_val >= T2DM_A1C_THRESHOLD)
        criteria_results.append({
            "criterion": f"T2DM A1C ≥ {T2DM_A1C_THRESHOLD}",
            "met": t2dm_met,
            "note": f"A1C={a1c_val}"
        })
        if not t2dm_met:
            missing.append(f"A1C must be ≥ {T2DM_A1C_THRESHOLD} for T2DM policy")

    # -----------------------------------------------------------------------
    # 2) POLICY CRITERIA (from formulary.json)
    #    We keep your original parsing & checks so plan policies still apply.
    # -----------------------------------------------------------------------
    for c in policy.get("criteria", []):
        ok, note = True, ""
        cl = c.lower()

        # Age criterion like: "Age >= 18"
        if "age" in cl and ">=" in cl:
            m = re.search(r"age\s*>=\s*(\d+)", cl)
            age_in = int(patient.get("age", 0))
            thr = int(m.group(1)) if m else 0
            passed = age_in >= thr
            ok &= passed
            note = f"age={age_in} (need ≥ {thr})"
            if not passed:
                missing.append("Age threshold not met")

        # Diagnosis criterion e.g., "Diagnosis: Type 2 diabetes"
        if "diagnosis" in cl:
            if "type 2" in cl:
                passed = has_phrase("type 2 diabetes", patient.get("diagnoses", [])) or ("type 2" in lower_diag)
            elif "obesity" in cl or "e66" in cl:
                passed = has_phrase("obesity", patient.get("diagnoses", [])) or has_phrase("e66", patient.get("diagnoses", [])) \
                         or ("obesity" in lower_diag)
            else:
                # generic diagnosis presence
                passed = any(str(x).strip() for x in patient.get("diagnoses", [])) or bool(lower_diag)
            ok &= passed
            note += ("; " if note else "") + f"diagnoses={patient.get('diagnoses')}"
            if not passed:
                missing.append("Diagnosis does not match policy")

        # A1C thresholds embedded in policy text e.g., "A1C >= 7.5"
        if "a1c" in cl and ">=" in cl:
            m = re.search(r"a1c\s*>=\s*(\d+(?:\.\d+)?)", cl)
            a1c_in = (patient.get("labs", {}) or {}).get("A1c")
            thr = float(m.group(1)) if m else None
            if a1c_in is None or thr is None:
                ok = False
                note += ("; " if note else "") + "A1c missing"
                missing.append("Add recent A1c value")
            else:
                passed = float(a1c_in) >= thr
                ok &= passed
                note += ("; " if note else "") + f"A1c={a1c_in} (need ≥ {thr})"
                if not passed:
                    missing.append(f"A1c must be ≥ {thr}")

        # BMI policy including combined thresholds (BMI ≥ 30 or BMI ≥ 27 with comorbidity)
        if "bmi" in cl:
            bmi_in = (patient.get("labs", {}) or {}).get("BMI")
            if "or bmi >=" in cl:
                if bmi_in is None:
                    ok = False
                    note += ("; " if note else "") + "BMI missing"
                    missing.append("Add BMI and/or comorbidity")
                else:
                    passed = (bmi_in >= 30) or (bmi_in >= 27 and len(patient.get("comorbidities", [])) > 0)
                    ok &= passed
                    note += ("; " if note else "") + f"BMI={bmi_in}, comorbidities={patient.get('comorbidities')}"
                    if not passed:
                        missing.append("BMI/comorbidity threshold not met")

        # Lifestyle duration check (e.g., "≥ 3 months")
        if "lifestyle" in cl:
            months = int(patient.get("lifestyle_months", 0))
            passed = months >= 3
            ok &= passed
            note += ("; " if note else "") + f"lifestyle_months={months} (need ≥ 3)"
            if not passed:
                missing.append("Document ≥3 months lifestyle program")

        # Step therapy: "tried/failed" classes
        if "tried/failed" in cl:
            tried = [s.lower() for s in patient.get("tried_failed", [])]
            need_met = "metformin" in cl
            need_other = "additional" in cl or "either" in cl
            # Simple class synonyms
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
            if not passed:
                missing.append("Step therapy: need metformin + one additional class")

        # Contraindications present?
        if "contraindication" in cl:
            c_list = [s.lower() for s in patient.get("contraindications", [])]
            passed = len(c_list) == 0
            ok &= passed
            note += ("; " if note else "") + f"contraindications={patient.get('contraindications')}"
            if not passed:
                reasons.append("Safety: contraindication present")

        # Specific: pancreatitis is a contraindication
        if "pancreatitis" in cl:
            c_list = [s.lower() for s in patient.get("contraindications", [])]
            passed = ("pancreatitis" not in c_list)
            ok &= passed
            note += ("; " if note else "") + f"contraindications={patient.get('contraindications')}"
            if not passed:
                reasons.append("Safety: active pancreatitis")

        # Record per-criterion outcome for transparency
        criteria_results.append({"criterion": c, "met": bool(ok), "note": note})

    # -----------------------------------------------------------------------
    # 3) STEP THERAPY GROUPS (policy.step_therapy) and EXCLUSIONS
    # -----------------------------------------------------------------------
    step_groups = policy.get("step_therapy", [])
    tried_lower = {s.strip().lower() for s in patient.get("tried_failed", [])}

    def group_ok(g: str) -> bool:
        """
        Accepts either a single item or an OR group like "empagliflozin|dapagliflozin".
        Returns True if any item in the group is present in tried_failed.
        """
        if "|" in g:
            return any(opt.strip().lower() in tried_lower for opt in g.split("|"))
        return g.strip().lower() in tried_lower

    step_ok = all(group_ok(g) for g in step_groups) if step_groups else True
    if step_groups:
        # Capture step therapy as an explicit row for transparency
        criteria_results.append({
            "criterion": f"Step therapy satisfied: {step_groups}",
            "met": bool(step_ok),
            "note": f"tried_failed={sorted(list(tried_lower))}"
        })
    if not step_ok:
        missing.append(f"Step therapy groups not satisfied: {step_groups}")

    # Exclusions: if any trigger matches, will be listed and can deny
    exclusions = policy.get("exclusions", [])
    exclusions_hit: List[str] = []
    for ex in exclusions:
        ex_l = ex.lower()
        if "type 1" in ex_l and has_phrase("type 1", patient.get("diagnoses", [])):
            exclusions_hit.append(ex)
        if "medullary thyroid carcinoma" in ex_l and has_phrase("medullary", patient.get("contraindications", [])):
            exclusions_hit.append(ex)
        if "pregnancy" in ex_l and has_phrase("pregnan", patient.get("contraindications", [])):
            exclusions_hit.append(ex)

    # -----------------------------------------------------------------------
    # 4) Final decision assembly
    # -----------------------------------------------------------------------
    # All criteria rows (including hard prechecks and policy checks) must pass.
    criteria_all_met = all(r["met"] for r in criteria_results) if criteria_results else False

    # Overall meets_criteria requires:
    # - All criteria rows met
    # - Step therapy satisfied (if any)
    # - No exclusions triggered
    meets_criteria = bool(criteria_all_met and step_ok and (len(exclusions_hit) == 0))

    if meets_criteria:
        decision_code = "APPROVE"
        outcome = "Meets all formulary criteria and step therapy; no exclusions present."
        reasons.insert(0, "All criteria satisfied.")
        if step_groups:
            reasons.append(f"Step therapy satisfied: {step_groups}")
    else:
        decision_code = "DENY"
        # If nothing landed in 'missing' but step/exclusions failed, make that explicit
        if not missing and (not step_ok or exclusions_hit):
            if not step_ok:
                missing.append("Step therapy not met")
            if exclusions_hit:
                reasons.append(f"Exclusion triggered: {', '.join(exclusions_hit)}")
        outcome = "Does not meet criteria — insufficient evidence to satisfy one or more formulary criteria."

    # References block for downstream traceability
    references = {
        "plan": plan,
        "plan_type": plan_type,
        "policy_version": version,
        "policy_drug": drug,
        "quantity_limit": policy.get("quantity_limit"),
        "exclusions": exclusions
    }

    # Human-friendly bullet list for quick review
    rationale_bullets: List[str] = []
    for row in criteria_results:
        emoji = "✅" if row["met"] else "❌"
        rationale_bullets.append(f"{emoji} {row['criterion']} — {row['note']}")

    # Full explainable response
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
        "missing": list(dict.fromkeys(missing)),  # de-duplicate while preserving order
        "reasons": reasons,
        "rationale_bullets": rationale_bullets,
        "references": references
    }


# Entry point for MCP server
if __name__ == "__main__":
    mcp.run()
