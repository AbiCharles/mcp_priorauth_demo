"""
pbm_server.py

Prior Authorization (PA) policy service with optional LLM assistance.
Deterministic decisioning uses formulary.json in /app/data; LLM is only for
rationales/extraction (never to flip APPROVE/DENY).

Run:
  uvicorn pbm_server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import json
import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, validator

# ----------------------- Logging ----------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("pbm_server")

# ----------------------- LLM config (Together) --------------------------------
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()

TOGETHER_AVAILABLE = False
llm_client = None
try:
    if TOGETHER_API_KEY:
        from together import Together  # type: ignore
        llm_client = Together(api_key=TOGETHER_API_KEY)
        TOGETHER_AVAILABLE = True
        log.info("Together client initialized; LLM features ENABLED.")
    else:
        log.warning("TOGETHER_API_KEY not set; LLM features DISABLED.")
except Exception:
    log.exception("Failed to initialize Together client; LLM features DISABLED.")

# ----------------------- Data Loading -----------------------------------------
DATA_DIR = Path("/app/data").resolve()
FORMULARY_PATH = DATA_DIR / "formulary.json"
PRECEDENTS_PATH = DATA_DIR / "precedents.jsonl"

def load_json_or_empty(path: Path) -> Dict[str, Any]:
    """Return JSON contents from `path`, or `{}` if the file is missing."""
    if not path.exists():
        log.error("%s not found at %s", path.name, path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_precedents(path: Path) -> List[Dict[str, Any]]:
    """Return precedent rows from a JSONL file, ignoring malformed lines."""
    if not path.exists():
        log.info("precedents.jsonl not found at %s; continuing without precedents.", path)
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed precedents line.")
    return items

FORMULARY = load_json_or_empty(FORMULARY_PATH)
if FORMULARY:
    log.info("Loaded formulary.json from %s", FORMULARY_PATH)

PRECEDENTS = load_precedents(PRECEDENTS_PATH)
if PRECEDENTS:
    log.info("Loaded %d precedent(s) from %s", len(PRECEDENTS), PRECEDENTS_PATH)

# ----------------------- Models -----------------------------------------------
class Patient(BaseModel):
    """Structured representation of the member associated with a PA request."""

    age: Optional[int] = None
    diagnoses: List[str] = Field(default_factory=list)
    tried_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    # Structured labs preferred over parsing notes (e.g., {"A1c": 8.1, "BMI": 31.0})
    labs: Optional[Dict[str, float]] = Field(
        default=None,
        description="Structured labs (e.g., {'A1c': 8.1, 'BMI': 31.0})"
    )

class EvaluatePARequest(BaseModel):
    """Request payload accepted by the `/evaluate_pa` FastAPI route."""

    plan: str
    drug: str
    patient: Patient
    use_llm: bool = True

    @validator("plan", "drug")
    def norm(cls, v: str) -> str:
        """Normalize whitespace so formulary lookup succeeds."""
        return v.strip()

class Decision(BaseModel):
    """Deterministic decision surfaced to the client UI."""

    status: str  # APPROVE | DENY
    missing_criteria: List[str] = Field(default_factory=list)
    quantity_limit: Optional[str] = None
    policy_criteria: List[str] = Field(default_factory=list)

class EvaluatePAResponse(BaseModel):
    """Envelope for the decision, optional rationale, and debug metadata."""

    decision: Decision
    rationale: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None

class ExtractFactsRequest(BaseModel):
    """Wrapper for free text we want to mine for basic clinical entities."""

    text: str

class ExtractFactsResponse(BaseModel):
    """Heuristic fact extraction returned by the `/extract_patient_facts` route."""

    medications: List[str] = Field(default_factory=list)
    med_classes: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    procedures: List[str] = Field(default_factory=list)
    durations: List[str] = Field(default_factory=list)
    raw_chunks: Optional[List[str]] = None

class ExplainPARequest(BaseModel):
    """Input schema for `/explain_pa`, including the prior deterministic decision."""

    plan: str
    drug: str
    patient: Patient
    decision: Decision

class ExplainPAResponse(BaseModel):
    """LLM-generated rationale string returned from `/explain_pa`."""

    rationale: str

# ----------------------- Policy Lookup ----------------------------------------
def _ci_get_key(d: Dict[str, Any], key: str) -> Optional[str]:
    """Return the dict key matching `key` case-insensitively, or None."""
    if key in d:
        return key
    lk = key.lower()
    for k in d.keys():
        if str(k).lower() == lk:
            return k
    return None

def get_policy(plan: str, drug: str) -> Tuple[List[str], Optional[str], Dict[str, Any], Dict[str, str]]:
    """Look up formulary metadata for a `plan`/`drug` combination.

    Handles either of the formulary shapes expected by the client:
        {"plans": { PLAN: { "PA": { DRUG: {...} } } }}
        { PLAN: { "drugs": { DRUG: {...} } } }

    Returns
    -------
    criteria, quantity_limit, entry, resolved_names
        Criteria list, the quantity limit (if any), the raw formulary entry, and
        a mapping describing the exact plan/drug keys used after case folding.
    """
    root = FORMULARY or {}
    plans = root.get("plans", root) if isinstance(root, dict) else {}
    if not isinstance(plans, dict):
        return [], None, {}, {}

    plan_key = _ci_get_key(plans, plan)
    if not plan_key:
        return [], None, {}, {}
    plan_obj = plans[plan_key] if isinstance(plans[plan_key], dict) else {}

    drug_blocks = None
    if "PA" in plan_obj and isinstance(plan_obj["PA"], dict):
        drug_blocks = plan_obj["PA"]
    elif "drugs" in plan_obj and isinstance(plan_obj["drugs"], dict):
        drug_blocks = plan_obj["drugs"]
    if not isinstance(drug_blocks, dict):
        return [], None, {}, {"resolved_plan": plan_key}

    drug_key = _ci_get_key(drug_blocks, drug)
    if not drug_key:
        return [], None, {}, {"resolved_plan": plan_key}

    entry = drug_blocks.get(drug_key, {})
    criteria = [str(c) for c in entry.get("criteria", [])]
    ql = entry.get("quantity_limit")
    resolved = {"resolved_plan": plan_key, "resolved_drug": drug_key}
    return criteria, (str(ql) if ql is not None else None), entry, resolved

# ----------------------- Criteria Evaluation ----------------------------------
SGLT2 = {"empagliflozin", "canagliflozin", "dapagliflozin", "ertugliflozin", "sglt2"}
SULFONYLUREAS = {"glipizide", "glyburide", "glimepiride", "su", "sulfonylurea", "sulfonylureas"}

def _extract_a1c(text: str) -> Optional[float]:
    """Parse the first numeric A1c value from unstructured text."""
    if not text:
        return None
    m = re.search(r"\bA1c\b[^0-9]{0,10}(\d{1,2}(?:\.\d+)?)", text, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def _extract_bmi(text: str) -> Optional[float]:
    """Parse the first numeric BMI value from unstructured text."""
    if not text:
        return None
    m = re.search(r"\bBMI\b[^0-9]{0,10}(\d{1,2}(?:\.\d+)?)", text, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def _get_a1c_from_patient(patient: Patient) -> Optional[float]:
    """Prefer structured labs['A1c'] if provided; otherwise parse from notes."""
    if patient.labs:
        for k, v in patient.labs.items():
            if str(k).lower() == "a1c":
                try:
                    return float(v)
                except Exception:
                    pass
    return _extract_a1c(patient.notes or "")

def _get_bmi_from_patient(patient: Patient) -> Optional[float]:
    """Prefer structured labs['BMI'] if provided; otherwise parse from notes."""
    if patient.labs:
        for k, v in patient.labs.items():
            if str(k).lower() == "bmi":
                try:
                    return float(v)
                except Exception:
                    pass
    return _extract_bmi(patient.notes or "")

def _token_set(text: str) -> set:
    """Return tokens that look like medication names from the provided text."""
    return set(t.lower() for t in re.findall(r"\b([A-Za-z][A-Za-z0-9\-]{2,})\b", text or ""))

def _meds_set(patient: Patient) -> set:
    """Merge structured medications with tokens mined from free-text notes."""
    meds = set(m.lower() for m in patient.tried_medications)
    meds |= _token_set(patient.notes or "")
    return meds

def evaluate_against_policy(patient: Patient, entry: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (met_criteria, missing_criteria) for a formulary entry.

    Patterns handled include age gates, targeted diagnoses, A1c thresholds,
    BMI cutoffs, combination-therapy requirements, and contraindication checks
    (optionally supplemented by `entry['exclusions']`).

    When no structured pattern matches, we fall back to a simple substring
    search across the patient's notes and diagnoses.
    """
    criteria: List[str] = [str(c) for c in entry.get("criteria", [])]
    exclusions = [str(x).lower() for x in entry.get("exclusions", [])]
    diag_text = " ".join(patient.diagnoses or [])
    corpus = f"{patient.notes or ''} || {diag_text}".lower()
    meds = _meds_set(patient)

    met, missing = [], []
    for c in criteria:
        cl = c.lower()
        ok = False

        if cl.startswith("age >="):
            m = re.search(r"age\s*>=\s*(\d+)", cl)
            th = int(m.group(1)) if m else None
            ok = (patient.age is not None and th is not None and patient.age >= th)

        elif cl.startswith("diagnosis:"):
            target = c.split(":", 1)[1].strip().lower()
            ok = (target in corpus)

        elif cl.startswith("a1c >="):
            m = re.search(r"a1c\s*>=\s*([0-9.]+)", cl)
            th = float(m.group(1)) if m else None
            a1c = _get_a1c_from_patient(patient)  # labs first, then notes
            ok = (a1c is not None and th is not None and a1c >= th and ("metformin" in meds))

        elif cl.startswith("bmi"):
            # Support "BMI >= 27", "BMI > 27", or Unicode ≥ — policy-level check
            m = re.search(r"bmi\s*(>=|>|≥)\s*([0-9.]+)", cl, flags=re.I)
            bmi = _get_bmi_from_patient(patient)  # labs first, then notes
            ok = False
            if bmi is not None:
                if m:
                    op, val = m.group(1), float(m.group(2))
                    th = max(val, 28.0)  # enforce a minimum threshold of 28
                    if op in (">=", "≥"):
                        ok = bmi >= th
                    else:  # ">"
                        ok = bmi > th
                else:
                    ok = bmi >= 28.0  # generic BMI criterion -> require >= 28

        elif cl.startswith("tried/failed"):
            has_metformin = "metformin" in meds
            has_extra = bool(SGLT2 & meds) or bool(SULFONYLUREAS & meds)
            ok = has_metformin and has_extra

        elif cl.startswith("no contraindication"):
            ok = True
            for ex in exclusions:
                if ex and ex in corpus:
                    ok = False
                    break

        else:
            # Fallback: loose substring
            ok = (cl in corpus)

        (met if ok else missing).append(c)

    return met, missing

# ----------------------- LLM helpers ------------------------------------------
SYSTEM_EXPLAINER = (
    "You are a concise clinical PA assistant. "
    "Write short, clinician-facing rationales. "
    "Never change the decision outcome; only explain it and advise on next steps."
)

def call_llm(messages: List[Dict[str, str]], max_tokens: int = 350, temperature: float = 0.1) -> Optional[str]:
    """Call the Together chat completion API, returning stripped content."""
    if not TOGETHER_AVAILABLE:
        return None
    try:
        resp = llm_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        log.exception("LLM call failed")
        return None

def llm_rationale(plan: str, drug: str, criteria: List[str], patient: Patient, decision: Decision) -> Optional[str]:
    """Generate bullet-point rationales grounded in deterministic results."""

    bullets = "\n- ".join(criteria) if criteria else "(no explicit criteria found)"
    patient_json = patient.model_dump_json(indent=2)
    decision_json = decision.model_dump_json(indent=2)

    grounding_snips = []
    for p in PRECEDENTS[:3]:
        try:
            grounding_snips.append(json.dumps(p, ensure_ascii=False))
        except Exception:
            pass
    grounding_text = "\n\n".join(grounding_snips) if grounding_snips else "(none)"

    prompt = f"""
Plan: {plan}
Drug: {drug}

Policy Criteria:
- {bullets}

Patient (JSON):
{patient_json}

Deterministic Decision (JSON):
{decision_json}

If DENY: list exact missing items and what documentation/evidence would satisfy them.
If APPROVE: cite which criteria are met and mention any quantity/step limits.

Return 2–5 crisp bullet points. Avoid filler. Do not alter the decision.

Relevant precedents (if helpful):
{grounding_text}
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_EXPLAINER},
        {"role": "user", "content": prompt},
    ]
    return call_llm(messages)

# ----------------------- FastAPI app ------------------------------------------
app = FastAPI(title="PBM PA Policy Service", version="1.1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    """Redirect API root requests to FastAPI's interactive documentation."""
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def health() -> Dict[str, str]:
    """Simple liveness probe that also reports LLM availability."""
    return {"status": "ok", "llm": "enabled" if TOGETHER_AVAILABLE else "disabled"}

@app.get("/debug/formulary")
def debug_formulary() -> Dict[str, Any]:
    """Enumerate the drugs configured under each plan for quick inspection."""
    out = {}
    plans = FORMULARY.get("plans", FORMULARY) if isinstance(FORMULARY, dict) else {}
    if isinstance(plans, dict):
        for p, pobj in plans.items():
            drugs = []
            if isinstance(pobj, dict):
                if isinstance(pobj.get("PA"), dict):
                    drugs = list(pobj["PA"].keys())
                elif isinstance(pobj.get("drugs"), dict):
                    drugs = list(pobj["drugs"].keys())
            out[p] = sorted(drugs)
    return out

@app.get("/debug/formulary/{plan}")
def debug_formulary_plan(plan: str) -> Dict[str, Any]:
    """Return granular formulary details for a specific plan."""
    criteria, ql, entry, resolved = get_policy(plan, next(iter((FORMULARY.get("plans") or FORMULARY).get(plan, {}).get("PA", {})), ""))
    return {
        "resolved": resolved,
        "has_entry": bool(entry),
        "quantity_limit": ql,
        "criteria": entry.get("criteria", []) if entry else [],
    }

@app.post("/evaluate_pa", response_model=EvaluatePAResponse)
def evaluate_pa(req: EvaluatePARequest) -> EvaluatePAResponse:
    """Run deterministic policy checks and optionally augment with LLM output."""
    criteria, ql, entry, resolved = get_policy(req.plan, req.drug)
    if not criteria:
        decision = Decision(
            status="DENY",
            missing_criteria=["No explicit criteria found for this plan/drug in formulary.json."],
            quantity_limit=None,
            policy_criteria=[],
        )
        rationale = llm_rationale(req.plan, req.drug, [], req.patient, decision) if req.use_llm else None
        return EvaluatePAResponse(
            decision=decision,
            rationale=rationale,
            debug={"reason": "missing_policy", **resolved}
        )

    met, missing = evaluate_against_policy(req.patient, entry)

    # ---------- GLOBAL BMI GATE (applies to all plans/drugs) ----------
    # Enforce an absolute BMI floor so sub-threshold cases never auto-approve,
    # even if the formulary entry omits an explicit weight requirement.
    bmi = _get_bmi_from_patient(req.patient)
    if bmi is not None and bmi < 28.0:
        # ensure the UI shows this in requirements/missing
        bmi_req = "BMI ≥ 28 (global)"
        if bmi_req not in missing:
            missing.append(bmi_req)
        if bmi_req not in criteria:
            criteria = criteria + [bmi_req]
    # ------------------------------------------------------------------

    status = "APPROVE" if not missing else "DENY"
    decision = Decision(
        status=status,
        missing_criteria=missing,
        quantity_limit=ql,
        policy_criteria=criteria,
    )

    rationale = None
    if req.use_llm and TOGETHER_AVAILABLE:
        rationale = llm_rationale(req.plan, req.drug, criteria, req.patient, decision)

    debug = {
        "met_count": len([c for c in criteria if c not in set(missing)]),
        "missing_count": len(missing),
        "resolved_plan": resolved.get("resolved_plan"),
        "resolved_drug": resolved.get("resolved_drug"),
        "together_model": TOGETHER_MODEL if TOGETHER_AVAILABLE else None,
    }
    return EvaluatePAResponse(decision=decision, rationale=rationale, debug=debug)

@app.post("/extract_patient_facts", response_model=ExtractFactsResponse)
def extract_patient_facts(req: ExtractFactsRequest) -> ExtractFactsResponse:
    """Regex-based backup for extracting meds/conditions when LLM is disabled."""
    # Simple heuristic extraction; LLM version available if needed.
    meds = sorted(set(re.findall(r"\b([A-Z][a-zA-Z0-9\-]{2,})\b", req.text)))
    conditions = sorted(set(re.findall(r"\b(arthritis|asthma|diabetes|psoriasis|migraine|copd|ra|t2dm|type 2 diabetes)\b", req.text, flags=re.I)))
    durations = sorted(set(re.findall(r"\b(\d+\s*(?:days?|weeks?|months?|years?))\b", req.text, flags=re.I)))

    return ExtractFactsResponse(
        medications=meds[:50],
        med_classes=[],
        conditions=conditions[:50],
        procedures=[],
        durations=durations[:50],
        raw_chunks=[c.strip() for c in re.split(r"[;\n]", req.text) if c.strip()],
    )

@app.post("/explain_pa", response_model=ExplainPAResponse)
def explain_pa(req: ExplainPARequest) -> ExplainPAResponse:
    """Call the LLM to narrate how the decision was reached for clinicians."""
    if not TOGETHER_AVAILABLE:
        raise HTTPException(status_code=400, detail="LLM is not configured. Set TOGETHER_API_KEY.")
    rationale = llm_rationale(req.plan, req.drug, req.decision.policy_criteria, req.patient, req.decision)
    if not rationale:
        raise HTTPException(status_code=500, detail="Failed to generate rationale.")
    return ExplainPAResponse(rationale=rationale)

# ----------------------- Main --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("pbm_server:app", host="0.0.0.0", port=port, reload=bool(os.getenv("DEV_RELOAD")))
