# client/app.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import requests  # HTTP client for FastAPI calls

# ---- MCP client import (robust, lazy connect) ----
MCP_AVAILABLE = False
MCP_STATUS = "not initialized"
try:
    from client.mcp_client import MCPClient  # when launched as a package
except Exception:
    try:
        from mcp_client import MCPClient     # when launched as a script
    except Exception as e:
        MCPClient = None
        MCP_STATUS = f"import failed: {e!s}"

# Don't probe tools at import time. Just prepare the client and mark available.
mcp_client = MCPClient() if MCPClient else None
if MCPClient:
    MCP_AVAILABLE = True
    MCP_STATUS = "client ready (lazy connect)"
# -----------------------------------------------

# ------------------------------------
DATA_DIR = Path("/app/data")
FORMULARY_PATH = DATA_DIR / "formulary.json"

# Preferred defaults (can be overridden by env) — prefilled to an APPROVE case
DEFAULT_PLAN = os.environ.get("DEFAULT_PLAN", "AcmeCommercial")
DEFAULT_DRUG = os.environ.get("DEFAULT_DRUG", "Semaglutide")

# FastAPI policy server base URL (pbm_server.py)
PBM_API_BASE = os.getenv("PBM_API_BASE", "http://localhost:7860").rstrip("/")

DEFAULTS = {
    "plan": DEFAULT_PLAN,
    "drug": DEFAULT_DRUG,
    "diagnosis_text": "Type 2 diabetes mellitus",  # meets Diagnosis criterion
    "icd10": "E11.65",
    "age": "52",                                   # meets Age >= 18
    "a1c": "8.1",                                  # used by local fallback
    "bmi": "31.0",
    # API path reads A1c & step therapy from NOTES (plus tried meds list)
    "notes": "A1c 8.1% despite 6 months of metformin and empagliflozin; no contraindications.",
    "indication": "Type 2 diabetes",
}

# -----------------------------
# Load formulary (local fallback)
# -----------------------------
def load_formulary_map() -> Dict[str, List[str]]:
    try:
        with open(FORMULARY_PATH, "r") as f:
            data = json.load(f)
        plans_blob = data.get("plans", {})
        mapping: Dict[str, List[str]] = {}
        for plan, blob in plans_blob.items():
            drugs = sorted(list((blob or {}).get("PA", {}).keys()))
            mapping[plan] = drugs
        if mapping:
            return mapping
    except Exception:
        pass
    # sensible stub
    return {
        "AcmeCommercial": ["Semaglutide", "Tirzepatide"],
        "AcmeMA": ["Semaglutide"],
        "AcmeMedicaid": ["Wegovy"],
    }

FORMULARY = load_formulary_map()
ALL_PLANS = sorted(FORMULARY.keys())

def initial_defaults_from_formulary() -> Tuple[str, str]:
    plan = DEFAULTS["plan"] if DEFAULTS["plan"] in FORMULARY else (ALL_PLANS[0] if ALL_PLANS else "")
    drugs = FORMULARY.get(plan, [])
    drug = DEFAULTS["drug"] if DEFAULTS["drug"] in drugs else (drugs[0] if drugs else "")
    return plan, drug

INIT_PLAN, INIT_DRUG = initial_defaults_from_formulary()

# -----------------------------
# Utils
# -----------------------------
def to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default

def to_int(s: str, default: int = 0) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return default

def fmt_json(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

def _api_alive() -> bool:
    try:
        r = requests.get(f"{PBM_API_BASE}/healthz", timeout=(2, 5))
        return r.ok
    except Exception:
        return False

# Map common brand names → generic keys used in policy
_BRAND_TO_GENERIC = {
    "ozempic": "Semaglutide",
    "rybelsus": "Semaglutide",
    "wegovy": "Wegovy",
    "mounjaro": "Tirzepatide",
    "trulicity": "Dulaglutide",
    "victoza": "Liraglutide",
}
def _normalize_drug_for_api(drug: str) -> str:
    """
    Prefer generic (e.g., 'Ozempic (semaglutide)' -> 'Semaglutide').
    Falls back to small brand map or original string.
    """
    m = re.search(r"\(([^)]+)\)", drug or "")
    if m:
        return m.group(1).strip().title()
    low = (drug or "").lower().strip()
    for brand, generic in _BRAND_TO_GENERIC.items():
        if brand in low:
            return generic
    return (drug or "").strip()

# -----------------------------
# Local evaluator (fallback)
# -----------------------------
def local_evaluate(plan: str, drug: str, diagnosis_text: str, fields: Dict[str, str]) -> Dict:
    # Parse
    age = to_int(fields.get("age", ""), 0)
    a1c = to_float(fields.get("a1c", ""), -1.0)
    bmi = to_float(fields.get("bmi", ""), -1.0)
    diagnosis = (fields.get("diagnosis_text") or "").strip()

    # Hard requirements
    reqs = [
        "Diagnosis text provided",
        "Age provided",
        "A1C provided and valid",
        "BMI provided and valid",
    ]
    meets = []
    missing = []

    # Required presence checks
    if diagnosis:
        meets.append("Diagnosis text provided")
    else:
        missing.append("Provide Diagnosis text")

    if age >= 1:
        meets.append("Age provided")
    else:
        missing.append("Provide Age")

    # Valid ranges
    A1C_MIN_VALID, A1C_MAX_VALID = 4.0, 15.0
    BMI_MIN_VALID, BMI_MAX_VALID = 10.0, 80.0

    if A1C_MIN_VALID <= a1c <= A1C_MAX_VALID:
        meets.append("A1C provided and valid")
    else:
        missing.append(f"A1C must be between {A1C_MIN_VALID} and {A1C_MAX_VALID}")

    if BMI_MIN_VALID <= bmi <= BMI_MAX_VALID:
        meets.append("BMI provided and valid")
    else:
        missing.append(f"BMI must be between {BMI_MIN_VALID} and {BMI_MAX_VALID}")

    # Indication-specific thresholds
    lower_diag = diagnosis.lower()
    is_obesity = ("obesity" in lower_diag) or (drug.lower().startswith("wegovy"))
    is_t2dm = ("type 2" in lower_diag) or ("t2dm" in lower_diag) or ("e11" in lower_diag)

    T2DM_A1C_THRESHOLD = 7.5
    OBESITY_BMI_PRIMARY = 30.0
    OBESITY_BMI_SECONDARY = 27.0

    if is_t2dm:
        reqs.append(f"T2DM A1C threshold (≥{T2DM_A1C_THRESHOLD})")
        if a1c >= T2DM_A1C_THRESHOLD:
            meets.append(f"T2DM A1C threshold (≥{T2DM_A1C_THRESHOLD})")
        else:
            missing.append(f"A1C must be ≥ {T2DM_A1C_THRESHOLD} for T2DM policy")

    if is_obesity:
        reqs.append(f"Obesity BMI policy (≥{OBESITY_BMI_PRIMARY} or ≥{OBESITY_BMI_SECONDARY} + comorbidity)")
        # We don't model comorbidities here; enforce primary BMI as minimal safeguard
        if bmi >= OBESITY_BMI_PRIMARY:
            meets.append(f"Obesity BMI policy (≥{OBESITY_BMI_PRIMARY} or ≥{OBESITY_BMI_SECONDARY} + comorbidity)")
        else:
            missing.append(f"Obesity BMI policy not met (BMI≥{OBESITY_BMI_PRIMARY} or BMI≥{OBESITY_BMI_SECONDARY} with comorbidity)")

    # Formulary + step (basic local heuristics)
    on_formulary = drug in FORMULARY.get(plan, [])
    notes = (fields.get("notes") or "").lower()
    metformin_fail = "metformin" in notes and any(k in notes for k in ["fail", "failed", "inadequate", "intoler"])

    reqs.append("On plan formulary")
    if on_formulary:
        meets.append("On plan formulary")
    else:
        missing.append("Drug not on plan formulary")

    reqs.append("Step: tried/failed metformin")
    if metformin_fail:
        meets.append("Step: tried/failed metformin")
    else:
        # only enforce this if T2DM
        if is_t2dm:
            missing.append("Step therapy not met: document tried/failed metformin")

    approved = (len(missing) == 0)
    return {
        "requirements": reqs,
        "meets": meets,
        "missing": list(dict.fromkeys(missing)),
        "approved": approved,
        "source": "local",
    }

# -----------------------------
# Transform MCP response -> UI sections
# -----------------------------
def mcp_to_sections(resp: Dict[str, any]) -> Dict[str, any]:
    criteria = resp.get("criteria_evaluation", []) or []
    step_required = resp.get("step_therapy_required", []) or []
    step_ok = bool(resp.get("step_therapy_satisfied", True))
    exclusions = resp.get("exclusions", []) or []
    exclusions_hit = resp.get("exclusions_hit", []) or []

    requirements: List[str] = []
    meets: List[str] = []

    for row in criteria:
        crit = str(row.get("criterion", "")).strip()
        met = bool(row.get("met", False))
        if crit:
            requirements.append(crit)
            if met:
                meets.append(crit)

    if step_required:
        step_label = f"Step therapy satisfied: {step_required}"
        requirements.append(step_label)
        if step_ok:
            meets.append(step_label)

    if exclusions:
        excl_label = "No exclusions present"
        requirements.append(excl_label)
        if len(exclusions_hit) == 0:
            meets.append(excl_label)

    missing = list(resp.get("missing", []))
    if step_required and not step_ok:
        missing.append("Step therapy not met")
    if exclusions_hit:
        missing.append(f"Exclusion triggered: {', '.join(exclusions_hit)}")

    approved = str(resp.get("decision_code", "")).upper() == "APPROVE"
    return {
        "requirements": requirements,
        "meets": meets,
        "missing": missing or [],
        "approved": approved,
    }

# -----------------------------
# MCP discovery (optional)
# -----------------------------
def try_mcp_lists() -> Tuple[List[str], Dict[str, List[str]]]:
    if not MCP_AVAILABLE:
        return ALL_PLANS, FORMULARY
    try:
        client = MCPClient()
        plans = client.list_plans()
        mapping: Dict[str, List[str]] = {}
        for p in plans:
            mapping[p] = client.list_drugs(p) or []
        for k in mapping:
            mapping[k] = sorted(mapping[k])
        return sorted(plans), mapping
    except Exception:
        return ALL_PLANS, FORMULARY

# -----------------------------
# MCP/API orchestration
# -----------------------------
def evaluate_via_mcp(
    plan: str,
    drug: str,
    diagnosis_text: str,
    fields: Dict[str, str],
    use_llama: bool,   # controls LLaMA rationale on API
) -> Tuple[Dict, str, List[str], Dict, Dict, str]:
    """
    Returns (sections, provenance, steps, patient_payload, raw_response, llm_rationale)
    Preference order:
      1) FastAPI (pbm_server) if PBM_API_BASE is alive
      2) MCP server if available
      3) Local heuristic fallback
    """
    steps: List[str] = []
    patient_payload: Dict = {}
    raw_resp: Dict = {}
    llm_rationale: str = ""

    # Build shared patient payload from UI fields
    notes = fields.get("notes", "") or ""
    notes_low = notes.lower()
    tried_failed = []
    if "metformin" in notes_low:
        tried_failed.append("metformin")
    for kw in [
        "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "sglt2",
        "sulfonylurea", "glipizide", "glimepiride", "glyburide",
        "sitagliptin", "linagliptin", "alogliptin", "dpp-4"
    ]:
        if kw in notes_low:
            tried_failed.append(kw)

    # ------------------ Try API (uses LLaMA) ------------------
    if _api_alive():
        steps.append(f"Detected FastAPI at {PBM_API_BASE}/healthz.")
        api_drug = _normalize_drug_for_api(drug)
        steps.append(f"Using API drug key: '{api_drug}' (from '{drug}').")
        steps.append(f"LLaMA rationale: {'enabled' if use_llama else 'disabled'} (use_llm flag).")

        api_payload = {
            "plan": plan,
            "drug": api_drug,
            "use_llm": bool(use_llama),  # drives LLaMA usage server-side
            "patient": {
                "age": to_int(fields.get("age", "0"), 0),
                "diagnoses": [diagnosis_text, fields.get("icd10", ""), fields.get("indication", "")],
                "tried_medications": tried_failed,
                "allergies": [],
                "notes": notes,
                "labs": {
                    "A1c": to_float(fields.get("a1c", "0"), 0.0),
                    "BMI": to_float(fields.get("bmi", "0"), 0.0),
                },
            },
        }
        patient_payload = api_payload["patient"]

        try:
            r = requests.post(
                f"{PBM_API_BASE}/evaluate_pa",
                headers={"Content-Type": "application/json"},
                data=json.dumps(api_payload),
                timeout=(5, 30),
            )
            r.raise_for_status()
            raw_resp = r.json()
            steps.append("Called API /evaluate_pa and received response.")

            dec = raw_resp.get("decision", {}) or {}
            status = str(dec.get("status", "")).upper()
            policy_criteria = dec.get("policy_criteria", []) or []
            missing = dec.get("missing_criteria", []) or []
            llm_rationale = raw_resp.get("rationale") or ""

            # Approximate 'meets' = criteria - missing
            meets = [c for c in policy_criteria if c not in set(missing)]
            sections = {
                "requirements": policy_criteria,
                "meets": meets,
                "missing": missing,
                "approved": (status == "APPROVE"),
            }
            sections["_raw"] = raw_resp
            return sections, "Evaluated via FastAPI (pbm_server)", steps, patient_payload, raw_resp, llm_rationale

        except Exception as e:
            steps.append(f"API evaluation failed: {e}. Falling back to MCP/local.")

    # ------------------ MCP path (unchanged) ------------------
    if MCP_AVAILABLE:
        try:
            client = MCPClient()
            steps.append("Started MCP client process.")
            try:
                tools = client.list_tools()
                steps.append(f"Discovered tools: {', '.join(sorted([t.get('name','') for t in tools if isinstance(t, dict)])) or '(none)'}")
            except Exception as e:
                steps.append(f"Failed to list tools: {e}")

            patient_payload = {
                "age": to_int(fields.get("age", "0"), 0),
                "diagnoses": [fields.get("diagnosis_text", ""), fields.get("icd10", ""), fields.get("indication", "")],
                "labs": {
                    "A1c": to_float(fields.get("a1c", "0"), 0.0),
                    "BMI": to_float(fields.get("bmi", "0"), 0.0),
                },
                "comorbidities": [],
                "lifestyle_months": 3,
                "tried_failed": tried_failed,
                "contraindications": [],
            }
            steps.append("Built patient payload.")

            raw_resp = client.evaluate_pa(plan, drug, diagnosis_text, patient_payload)
            steps.append(f"Called MCP tool `evaluate_pa(plan={plan}, drug={drug})` and received response.")

            sections = mcp_to_sections(raw_resp)
            steps.append("Translated MCP response → UI sections (requirements/meets/missing/approved).")
            sections["_raw"] = raw_resp
            return sections, "Evaluated via MCP server", steps, patient_payload, raw_resp, llm_rationale

        except Exception as e:
            steps.append(f"MCP evaluation failed: {e} → local fallback used.")

    # ------------------ Local fallback ------------------
    local = local_evaluate(plan, drug, diagnosis_text, fields)
    return local, "Evaluated locally (no API/MCP)", steps, patient_payload, raw_resp, llm_rationale

# -----------------------------
# UI
# -----------------------------
MCP_PLANS, MCP_FORMULARY = try_mcp_lists()
ALL_PLANS = MCP_PLANS or ALL_PLANS
FORMULARY = MCP_FORMULARY or FORMULARY
INIT_PLAN, INIT_DRUG = (
    (DEFAULTS["plan"], DEFAULTS["drug"])
    if DEFAULTS["plan"] in FORMULARY and DEFAULTS["drug"] in FORMULARY[DEFAULTS["plan"]]
    else initial_defaults_from_formulary()
)

with gr.Blocks(title="MCP + Together AI • Pharmacy Benefits Prior Authorization") as demo:
    gr.Markdown("# MCP + Together AI • Pharmacy Benefits Prior Authorization Demo")

    with gr.Row():
        with gr.Column(scale=1):
            plan_dd = gr.Dropdown(label="Plan", choices=ALL_PLANS, value=INIT_PLAN, interactive=True)
        with gr.Column(scale=1):
            drug_dd = gr.Dropdown(
                label="Drug",
                choices=FORMULARY.get(INIT_PLAN, []),
                value=INIT_DRUG,
                interactive=True,
            )

    with gr.Row():
        diagnosis_tb = gr.Textbox(label="Diagnosis text", value=DEFAULTS["diagnosis_text"])
        icd10_tb = gr.Textbox(label="ICD-10", value=DEFAULTS["icd10"])

    with gr.Row():
        age_tb = gr.Textbox(label="Age (years)", value=DEFAULTS["age"])
        a1c_tb = gr.Textbox(label="A1C (%)", value=DEFAULTS["a1c"])
        bmi_tb = gr.Textbox(label="BMI", value=DEFAULTS["bmi"])

    notes_tb = gr.Textbox(label="Notes (e.g., trials/failures, risks, comments)", value=DEFAULTS["notes"])
    indication_tb = gr.Textbox(label="Indication (for 'Meets Indication')", value=DEFAULTS["indication"])

    with gr.Row():
        use_mcp_chk = gr.Checkbox(label="Use MCP/API (if available)", value=True)
        use_llama_chk = gr.Checkbox(label="Use LLaMA rationale (server)", value=True)  # drives API use_llm
        evaluate_btn = gr.Button("Evaluate", variant="primary")
        reload_defaults_btn = gr.Button("Reload defaults")

    with gr.Row():
        with gr.Column(scale=1):
            full_req_md = gr.Markdown("### Full Requirements\n- *(none yet)*")
        with gr.Column(scale=1):
            meets_md = gr.Markdown("### Patient Meets\n- *(none yet)*")
    missing_md = gr.Markdown("### Missing Requirements\n- *(none yet)*")
    approval_banner = gr.HTML("")
    provenance_md = gr.Markdown("")
    llm_md = gr.Markdown("### LLM Rationale\n*(none yet)*")  # shows LLaMA explanation if enabled

    # Trace & Debug tab (kept)
    with gr.Tabs():
        with gr.Tab("Trace & Debug"):
            steps_md = gr.Markdown("### Trace\n*(no steps yet)*")
            patient_code = gr.Code(label="Patient payload sent", value="{}", language="json")
            raw_code = gr.Code(label="Raw response", value="{}", language="json")

    def on_plan_change(plan: str) -> dict:
        choices = FORMULARY.get(plan, [])
        value = DEFAULT_DRUG if DEFAULT_DRUG in choices else (choices[0] if choices else "")
        return gr.update(choices=choices, value=value)

    plan_dd.change(fn=on_plan_change, inputs=[plan_dd], outputs=[drug_dd], queue=False)

    def on_reload_defaults():
        plan, drug = initial_defaults_from_formulary()
        return (
            gr.update(value=plan, choices=ALL_PLANS),
            gr.update(value=drug, choices=FORMULARY.get(plan, [])),
            gr.update(value=DEFAULTS["diagnosis_text"]),
            gr.update(value=DEFAULTS["icd10"]),
            gr.update(value=DEFAULTS["age"]),
            gr.update(value=DEFAULTS["a1c"]),
            gr.update(value=DEFAULTS["bmi"]),
            gr.update(value=DEFAULTS["notes"]),
            gr.update(value=DEFAULTS["indication"]),
        )

    reload_defaults_btn.click(
        fn=on_reload_defaults,
        inputs=[],
        outputs=[plan_dd, drug_dd, diagnosis_tb, icd10_tb, age_tb, a1c_tb, bmi_tb, notes_tb, indication_tb],
        queue=False,
    )

    def render_sections(evaluation: Dict) -> Tuple[str, str, str, str]:
        reqs = evaluation.get("requirements", [])
        meets = set(evaluation.get("meets", []))
        missing = evaluation.get("missing", [])
        approved = bool(evaluation.get("approved", False))

        req_md = ["### Full Requirements"]
        req_md += [f"- {r}" for r in reqs] if reqs else ["- *(No requirements returned)*"]
        meet_md = ["### Patient Meets"]
        meet_md += [f"- ✅ {r}" for r in reqs if r in meets] if meets else ["- *(None)*"]
        miss_md = ["### Missing Requirements"]
        miss_md += [f"- ❌ {m}" for m in missing] if missing else ["- **None missing**"]

        if approved:
            banner = """
            <div style="margin-top:16px;padding:18px;border-radius:12px;background:#ecfdf5;border:1px solid #10b981;display:flex;align-items:center;gap:12px;">
              <div style="font-size:28px;">✅</div>
              <div>
                <div style="font-weight:700;font-size:20px;color:#065f46;">APPROVED</div>
                <div style="font-size:14px;color:#065f46;">All requirements satisfied.</div>
              </div>
            </div>
            """
        else:
            banner = """
            <div style="margin-top:16px;padding:18px;border-radius:12px;background:#fef2f2;border:1px solid #ef4444;display:flex;align-items:center;gap:12px;">
              <div style="font-size:28px;">❌</div>
              <div>
                <div style="font-weight:700;font-size:20px;color:#991b1b;">NOT APPROVED</div>
                <div style="font-size:14px;color:#991b1b;">Some requirements are missing.</div>
              </div>
            </div>
            """
        return "\n".join(req_md), "\n".join(meet_md), "\n".join(miss_md), banner

    def on_evaluate(
        plan: str,
        drug: str,
        diagnosis_text: str,
        icd10: str,
        age: str,
        a1c: str,
        bmi: str,
        notes: str,
        indication_text: str,
        use_remote: bool,
        use_llama: bool,
    ):
        fields = {
            "diagnosis_text": diagnosis_text,
            "icd10": icd10,
            "age": age,
            "a1c": a1c,
            "bmi": bmi,
            "notes": notes,
            "indication": indication_text,
        }

        if use_remote:
            evaluation, prov, steps, patient_payload, raw_resp, llm_rationale = evaluate_via_mcp(
                plan, drug, diagnosis_text, fields, use_llama
            )
        else:
            # local path but still show what we'd send and steps
            steps = ["Remote (API/MCP) disabled by user → local evaluator used."]
            patient_payload = {
                "age": to_int(fields.get("age", "0"), 0),
                "diagnoses": [fields.get("diagnosis_text", ""), fields.get("icd10", ""), fields.get("indication", "")],
                "labs": {"A1c": to_float(fields.get("a1c", "0"), 0.0), "BMI": to_float(fields.get("bmi", "0"), 0.0)},
                "comorbidities": [],
                "lifestyle_months": 3,
                "tried_failed": [],
                "contraindications": [],
            }
            evaluation, prov = local_evaluate(plan, drug, diagnosis_text, fields), "Evaluated locally (remote disabled)"
            raw_resp = {}
            llm_rationale = ""

        if "requirements" not in evaluation:
            evaluation = mcp_to_sections(evaluation)

        full_req, meets, missing, banner = render_sections(evaluation)

        # Trace markdown
        steps_md_text = "### Trace\n" + "\n".join([f"- {s}" for s in steps]) if steps else "### Trace\n- *(no steps)*"

        # If API provided model name, surface it
        model_name = ""
        try:
            dbg = (raw_resp or {}).get("debug") or {}
            model_name = dbg.get("together_model") or ""
        except Exception:
            pass
        if model_name:
            steps_md_text += f"\n\n*LLM model:* `{model_name}`"

        # LLM rationale panel
        llm_text = f"### LLM Rationale\n{llm_rationale}" if llm_rationale else "### LLM Rationale\n*(disabled or not available)*"

        return (
            full_req,
            meets,
            missing,
            banner,
            f"_{prov}_",
            llm_text,
            steps_md_text,
            fmt_json(patient_payload),
            fmt_json(raw_resp if raw_resp else {"note": "No remote response (local path)."}),
        )

    evaluate_btn.click(
        fn=on_evaluate,
        inputs=[
            plan_dd, drug_dd, diagnosis_tb, icd10_tb, age_tb, a1c_tb, bmi_tb, notes_tb, indication_tb,
            use_mcp_chk, use_llama_chk
        ],
        outputs=[
            full_req_md, meets_md, missing_md, approval_banner, provenance_md,
            llm_md, steps_md, patient_code, raw_code
        ],
        queue=False,
        concurrency_limit=3,
    )

# Keep UI on 7861 so API can own 7860
if __name__ == "__main__":
    GR_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
    GR_PORT = int(os.getenv("GRADIO_PORT", "7861"))
    GR_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    print(f"[boot] Gradio starting on {GR_HOST}:{GR_PORT} (share={GR_SHARE})")

    # NOTE: We avoid demo.queue(); events use queue=False above.
    demo.launch(
        server_name=GR_HOST,
        server_port=GR_PORT,
        share=GR_SHARE,
        show_api=False,
        quiet=True,
        max_threads=8,
    )
