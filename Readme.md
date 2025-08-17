# Pharmacy Benefits Prior Authorization — MCP Demo

A small, end‑to‑end demo that uses **Model Context Protocol (MCP)** to evaluate
**prior authorization** requests for pharmacy benefits. It ships with:

- A **Gradio UI** (`client/app.py`) for entering patient/context data and viewing a
  requirements summary, what the patient meets, any missing items, and a large approval/denial banner.
- An **MCP server** (`server/pbm_server.py`) built with `FastMCP` that exposes:
  - `list_plans` — list available plans from the formulary
  - `list_drugs(plan)` — list drugs for a plan that require PA
  - `evaluate_pa(plan, drug, diagnosis_text, patient)` — explainable decision
- A **thin MCP client adapter** (`client/mcp_client.py`) that spawns the server over **stdio**
  and calls MCP tools from the UI.
- Demo **policy data** in `data/` (formulary, guidelines, precedents).

> The UI also includes a **Trace & Debug** tab showing the exact steps taken,
> the **patient payload** sent to the MCP server, and the **raw MCP response**.

---

## How it works (MCP in this project)

MCP (Model Context Protocol) defines a standard way for a **client** to call **tools**
on a **server** over a stream (stdio here). In this project:

1. **UI (client/app.py)** builds a patient payload from form inputs.
2. It uses **`client/mcp_client.py`** to start the server process via stdio:
   - Command resolved from `MCP_SERVER_CMD` or `MCP_SERVER_PY` (default: `/app/server/pbm_server.py`).
   - A proper `StdioServerParameters` is passed to the official `mcp` Python client.
3. The client calls MCP tools implemented by **`server/pbm_server.py`**:
   - `list_plans`
   - `list_drugs(plan)`
   - `evaluate_pa(plan, drug, diagnosis_text, patient)`
4. The server loads policy rules from `data/formulary.json` and evaluates criteria like:
   - age thresholds,
   - diagnosis match (e.g., “Type 2 diabetes” or “Obesity”/ICD‑10 E66),
   - A1c and BMI thresholds,
   - lifestyle months,
   - step therapy (e.g., tried/failed metformin and an additional class),
   - exclusions (e.g., pancreatitis).
5. The **UI translates** the MCP response into:
   - **Full Requirements**
   - **Patient Meets**
   - **Missing Requirements**
   - **Approval/Denial banner**
   plus a provenance note and the raw response in the **Trace & Debug** tab.

If the MCP server is not reachable, the UI transparently **falls back** to a local
simpler evaluator and tells you it did so.

---

## Repository layout (minimal essentials)

- client/
- app.py # Gradio UI (uses MCP if available; else local fallback)
- mcp_client.py # Spawns MCP server via stdio & calls tools
- server/
-pbm_server.py # FastMCP server (tools + resources)
 data/
- formulary.json # Sample plans, PA policies & criteria
- guidelines/ # (optional) Markdown refs used as resources
- precedents.jsonl # (optional) Prior cases used as resources
- Dockerfile
- docker-compose.yml
- requirements.txt
- .env (you create this)

```

.
├── client/
│   ├── app.py                 # Gradio UI (uses MCP if available; local fallback otherwise)
│   ├── mcp_client.py          # StdIO MCP client adapter (spawns and calls the MCP server)
│   └── __init__.py            # (optional) makes 'client' a package for imports like 'from client.mcp_client ...'
│
├── server/
│   ├── pbm_server.py          # FastMCP server exposing tools: list_plans, list_drugs, evaluate_pa
│   └── __init__.py            # Makes 'server' a package
│
├── data/
│   ├── formulary.json         # Plans, PA criteria, step therapy, exclusions, quantity limits
│   ├── guidelines/            # Markdown clinical references exposed as MCP resources
│   │   ├── diabetes.md        # Diabetes clinical references
│   │   └── obesity.md         # Obesity clinical references
│   └── precedents.jsonl       # Prior cases; one JSON object per line
│
├── Dockerfile                 # Container image: installs deps, copies code/data, runs client/app.py
├── docker-compose.yml         # One service: exposes Gradio on host:7860 → container:7860
├── requirements.txt           # Pinned deps (gradio 4.44.x, mcp>=1.1.0,<2, fastmcp, etc.)
├── README.md                  # Project description, run instructions, and test cases
├── .env                       # (local-only) runtime vars: GRADIO_HOST/PORT, MCP_SERVER_PY, defaults, etc.

---

## Configuration (env vars)

Create a `.env` in the repo root (Compose will load it). Example:

```env
# Gradio serving
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
GRADIO_SHARE=false

# MCP server location (Python)
MCP_SERVER_PY=/app/server/pbm_server.py
# Alternatively, provide a full command (Node/TS server would also work)
# MCP_SERVER_CMD=node /app/server_node/pbm_server.mjs

# Server data dir override (defaults to /app/data inside container)
PBM_DATA_DIR=/app/data

# UI defaults (plan + drug must exist in formulary)
DEFAULT_PLAN=AcmeCommercial
DEFAULT_DRUG=Semaglutide
```

---

## Quick Start (Docker)
Run with Docker (recommended)
> **Prereqs:** Docker Desktop (or Docker Engine) and Docker Compose v2.

1. **Clone the repo**
   ```bash
   git clone https://github.com/AbiCharles/mcp_priorauth_demo.git
   cd mcp_priorauth_demo

docker compose build --no-cache
docker compose up --force-recreate


Once you see a log like:

- [boot] Gradio starting on 0.0.0.0:7860 (share=false)

Open http://localhost:7860 in your browser.

If you need a shareable link for a demo, set GRADIO_SHARE=true in .env and restart.

To verify the app is responding, try:

- curl -s http://localhost:7860/config | jq .enable_queue

## Common Docker issues & fixes

- App not reachable / “0.0.0.0 refused to connect”
Use http://localhost:7860 on your host (not 0.0.0.0). Ensure Compose has:
ports: - "7860:7860".

- Queue errors (pending_message_lock NoneType, etc.)
The app enables Gradio’s queue (demo = demo.queue()), which is required for
/queue/join. Do not remove it.

- MCP fallback message
If you see “MCP not available → local fallback evaluator used.”:

- Ensure mcp package is installed and pinned (see below).

- Ensure MCP_SERVER_PY or MCP_SERVER_CMD points to a valid server.

- Check container logs for Python import errors in server/pbm_server.py.

Run locally (without Docker)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
# env vars for local run
export GRADIO_HOST=0.0.0.0
export GRADIO_PORT=7860
export MCP_SERVER_PY="$(pwd)/server/pbm_server.py"

python client/app.py


Open http://localhost:7860.

Version pins you should keep

In requirements.txt:

# Gradio tested on 4.44.x in this repo
gradio==4.44.1

# Crucial for the stdio client API we use:
mcp>=1.1.0,<2.0.0

# Plus: fastapi, uvicorn, python-dotenv and other deps you already have


Our client/mcp_client.py uses StdioServerParameters to avoid signature
mismatches with older mcp client versions.
```
## **Using the UI**
* **Select a Plan and Drug.** (Plans and drugs come from `data/formulary.json` via MCP.)
* **Fill in patient data** (defaults provided).
* **Notes** are parsed for tried/failed classes (e.g., “metformin”, “dapagliflozin”, “SGLT2”, “sulfonylurea”…).
* Use **Indication** to support the “Meets Indication” checkpoint.
* Keep **“Use MCP server (if available)”** checked to use the server policy engine.
    * Uncheck to exercise the local fallback (minimal rules).
* Click **Evaluate**.
* **Review**:
    * Full Requirements (policy’s criteria + step therapy + exclusions)
    * Patient Meets (✅ items)
    * Missing Requirements (❌ items or None missing)
    * Approval banner (green/approved or red/denied).
* Open **Trace & Debug** tab to review:
    * Steps taken (tool discovery, payload build, tool call)
    * Patient payload sent to MCP (JSON)
    * Raw MCP response (JSON).

---

## **Data & Policy Customization**
* Edit `data/formulary.json` to change plans, drugs, and policy criteria.

The MCP server reads:

```json
{
  "version": "2024-10",
  "plans": {
    "AcmeCommercial": {
      "planType": "HMO",
      "PA": {
        "Semaglutide": {
          "criteria": [
            "Age >= 18",
            "Diagnosis: Type 2 diabetes",
            "A1c >= 7.0",
            "Tried/failed metformin and an additional class",
            "Lifestyle >= 3 months"
          ],
          "step_therapy": ["metformin", "sglt2|dpp-4|su"],
          "exclusions": ["Type 1 diabetes", "Pancreatitis"],
          "quantity_limit": "4 pens / 28 days"
        }
      }
    }
  }
}
- Add Markdown files under data/guidelines/ and records under data/precedents.jsonl

- If you want to demonstrate the resource endpoints exposed by the server:

   - pbm://formulary/{plan}

   - pbm://guideline/{topic}

   - pbm://precedent/{drug}
```
## Test cases

  - Tip: Keep “Use MCP server” checked for these. If you deliberately test the
  local fallback, only two rules are applied:
  - (1) Drug must be on the plan formulary and (2) tried/failed metformin (from Notes).

| Plan | Drug | Diagnosis text | ICD‑10 | Age | A1c | BMI | Notes (tried/failed…) | Indication | Expected (MCP) | Why |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | AcmeCommercial | Semaglutide | Type 2 diabetes mellitus, poorly controlled | E11.6 | 55 | 8.4 | 33.1 | Tried/failed metformin; dapagliflozin (SGLT2) | Type 2 diabetes | **APPROVE** | Meets age, diagnosis, A1c, step therapy (metformin + SGLT2). |
| **2** | AcmeCommercial | Semaglutide | Type 2 diabetes mellitus | E11.9 | 45 | 8.0 | 29.0 | Tried/failed metformin only (no second class) | Type 2 diabetes | **DENY (miss)** | Fails step therapy: needs an additional class. |
| **3** | AcmeCommercial | Semaglutide | Type 2 diabetes mellitus | E11.9 | 61 | 6.2 | 31.0 | Metformin + SGLT2 | Type 2 diabetes | **DENY (miss)** | A1c threshold unmet if policy requires ≥7.0. |
| **4** | AcmeCommercial | Semaglutide | Obesity with comorbidity (HTN) | E66.9 | 50 | — | 34.0 | Lifestyle program 4 months | Obesity | **APPROVE** | If policy includes obesity path (BMI ≥30), lifestyle ≥3 mo. |
| **5** | AcmeCommercial | Semaglutide | Type 2 diabetes mellitus | E11.9 | 39 | 8.2 | 28.0 | Metformin + DPP‑4 | Type 2 diabetes | **APPROVE** | Age ≥18, A1c ≥7, step therapy met with DPP‑4. |
| **6** | AcmeCommercial | Semaglutide | Type 2 diabetes mellitus | E11.9 | 66 | — | 32.0 | Metformin + SGLT2 | Type 2 diabetes | **DENY (miss)** | Missing A1c when required → “Add recent A1c value”. |