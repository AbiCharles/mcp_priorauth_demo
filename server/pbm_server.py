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
    if not path.exists():
        log.error("%s not found at %s", path.name, path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_precedents(path: Path) -> List[Dict[str, Any]]:
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
    age: Optional[int] = None
    diagnoses: List[str] = Field(default_factory=list)
    tried_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    # Prefer structured labs over parsing notes (e.g., {"A1c": 8.1, "BMI": 31.0})
    labs: Optional[Dict[str, float]] = Field(default=None, description="Structured labs (e.g., {'A1c': 8.1, 'BMI': 31.0})")

class EvaluatePARequest(BaseModel):
    plan: str
    drug: str
    patient: Patient
    use_llm: bool = True

    @validator("plan", "drug")
    def norm(cls, v: str) -> str:
        return v.strip()

class Decision(BaseModel):
    status: str  # APPROVE | DENY
    missing_criteria: List[str] = Field(default_factory=list)
    quantity_limit: Optional[str] = None
    policy_criteria: List[str] = Field(default_factory=list)

class EvaluatePAResponse(BaseModel):
    decision: Decision
    rationale: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None

class ExtractFactsRequest(BaseModel):
    text: str

class ExtractFactsResponse(BaseModel):
    medications: List[str] = Field(default_factory=list)
    med_classes: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    procedures: List[str] = Field(default_factory=list)
    durations: List[str] = Field(default_factory=list)
    raw_chunks: Optional[List[str]] = None

class ExplainPARequest(BaseModel):
    plan: str
    drug: str
    patient: Patient
    decision: Decision

class ExplainPAResponse(BaseModel):
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
    """
    Support schemas:
      {"plans": { PLAN: { "PA": { DRUG: {...} } } }}
    and also fallback to { PLAN: { "drugs": { DRUG: {...} } } }
    Returns (criteria, quantity_limit, entry, resolved_names)
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
    return set(t.lower() for t in re.findall(r"\b([A-Za-z][A-Za-z0-9\-]{2,})\b", text or ""))

def _meds_set(patient: Patient) -> set:
    meds = set(m.lower() for m in patient.tried_medications)
    meds |= _token_set(patient.notes or "")
    return meds

def evaluate_against_policy(patient: Patient, entry: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Heuristics for common criteria seen in your formulary:
      - "Age >= N"
      - "Diagnosis: <text>"
      - "A1c >= X despite metformin"
      - "BMI >= Y" or "BMI > Y"
      - "Tried/failed: metformin and one additional agent (SGLT2 or SU)"
      - "No contraindication to <...>"  (+ optional entry['exclusions'])
    Fallback: substring match.
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
            # Support "BMI >= 27", "BMI > 27", or Unicode ≥
            m = re.search(r"bmi\s*(>=|>|≥)\s*([0-9.]+)", cl, flags=re.I)
            if m:
                op, val = m.group(1), float(m.group(2))
                bmi = _get_bmi_from_patient(patient)  # labs first, then notes
                if bmi is not None:
                    if op in (">=", "≥"):
                        ok = bmi >= val
                    else:  # ">"
                        ok = bmi > val
            else:
                # If a generic BMI criterion appears without comparator, enforce > 27 by default
                bmi = _get_bmi_from_patient(patient)
                ok = (bmi is not None and bmi > 27.0)

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
app = FastAPI(title="PBM PA Policy Service", version="1.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok", "llm": "enabled" if TOGETHER_AVAILABLE else "disabled"}

@app.get("/debug/formulary")
def debug_formulary() -> Dict[str, Any]:
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
    criteria, ql, entry, resolved = get_policy(plan, next(iter((FORMULARY.get("plans") or FORMULARY).get(plan, {}).get("PA", {})), ""))
    return {
        "resolved": resolved,
        "has_entry": bool(entry),
        "quantity_limit": ql,
        "criteria": entry.get("criteria", []) if entry else [],
    }

@app.post("/evaluate_pa", response_model=EvaluatePAResponse)
def evaluate_pa(req: EvaluatePARequest) -> EvaluatePAResponse:
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
        "met_count": len(met),
        "missing_count": len(missing),
        "resolved_plan": resolved.get("resolved_plan"),
        "resolved_drug": resolved.get("resolved_drug"),
        "together_model": TOGETHER_MODEL if TOGETHER_AVAILABLE else None,
    }
    return EvaluatePAResponse(decision=decision, rationale=rationale, debug=debug)

@app.post("/extract_patient_facts", response_model=ExtractFactsResponse)
def extract_patient_facts(req: ExtractFactsRequest) -> ExtractFactsResponse:
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
