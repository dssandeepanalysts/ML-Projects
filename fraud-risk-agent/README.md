# Fraud Risk Agent (FRA)

A small, runnable implementation of the Fraud Risk Agent that Pega's **Claim Assist** case calls (A2A skill `assess_fraud_risk`) during Automated Assessment of a motor own-damage claim. It returns a risk score, a confidence, the fraud indicators that fired and a neutral explanation for the claims adjuster. It is **advisory only**: it never approves, rejects or routes a claim.

It is built from four source files:

| Source | Role in this build |
|---|---|
| *Fraud-Risk-Agent-Build-Specification.docx* (v1.0, 2 Oct 2026) | **Binding contract.** Request and response schema, the 5 indicators, formulas, acceptance tests AT-1 to AT-8 |
| *Claim-Assist-Pega-Blueprint-Source-Document v1.5* | Pega-side context: governance thresholds, decision tables DT-01 to DT-07, test scenarios TS-01 to TS-18 |
| *Fraud Risk Agent – Build Specification for Claude Code* (PDF) | Larger production design (ECS, DAA join, L2/L3 models, 26 indicators). The docx §13 explicitly scopes most of it out; it is used here only for robustness ideas (injection handling, template fallback, shadow-model roadmap) |
| *fra_synthetic_claims_1500.csv* + *fra_data_dictionary.csv* | 1,500 labelled synthetic claims, used for ingestion, regression tests and evaluation |

---

## 1. Agent architecture overview

```
 Pega Claim Assistant                         Fraud Risk Agent (this folder)
 ────────────────────                         ──────────────────────────────
 Connect Agent ── A2A JSON-RPC ──▶ server.py ─▶ FraudRiskAgent.assess(payload)   (fra_agent.py)
 (or Connect-REST /v1/assess)       bearer auth   │
                                                  ├─1 validate_request ───── invalid ──▶ FAILED + field name
                                                  ├─2 fired_indicators + risk_score      (rules, pure code)
                                                  ├─3 compute_confidence                 (data availability)
                                                  ├─4 template_reasoning                 (always computed)
                                                  ├─5 llm_reasoning  (optional) ──▶ LangChain ─▶ ChatOllama
                                                  │     facts only: no description,        (local llama3.2)
                                                  │     no hashes; guardrails fail ─▶ keep template
                                                  └─6 response_problem (self-check) ── bad ──▶ FAILED
 ◀──────── {claim_id, status, risk_score, confidence, risk_flags, reasoning} ────────
```

End to end, for one claim:

1. **Ingest.** Pega sends one JSON payload: hashed policy and VIN, loss facts, four claim-history facts and two data-availability flags. For offline work, `row_to_request()` maps a CSV row onto the same contract and drops the expected outputs and the label.
2. **Validate.** Missing fields, wrong types, unknown `loss.cause`, bad dates and personal data all return `FAILED` with a reason that names the field. Unknown optional fields are ignored.
3. **Score.** Five deterministic rules fire or don't. Points are summed, capped at 100 and raised to the highest floor among the fired rules.
4. **Confidence.** Computed from data availability only (100, 80 or 60); it never depends on the score.
5. **Explain.** A template sentence is always built; it reproduces the dataset's `reasoning` column exactly. If a local LLM is available and at least one indicator fired, LangChain asks Ollama to reword **the facts only**. The output must pass eight guardrail checks or the template is used instead.
6. **Self-check and respond.** Out-of-range values become `FAILED`, never a malformed `COMPLETED`. The agent is stateless and has no side effects, so Pega's retries are safe.

**Why not a tool-calling (ReAct) agent?** The spec requires that every number and flag come from plain code (§7, §12.4), that repeat calls return identical results (AT-8), and that each call answer inside Pega's 15-second timeout. A local 3B model choosing tools would add nondeterminism, latency and an injection surface without improving the score. So this is an agent in the A2A sense: an autonomous service with a published Agent Card and skill. The LLM's job is limited to wording.

---

## 2. Required tools & integrations

| Component | Version tested | Free / OSS | Used for | Where |
|---|---|---|---|---|
| Python | 3.11 (needs 3.10+) | ✓ | runtime | all |
| `langchain-core` | 1.6 | ✓ MIT | prompt template, runnable chain, output parser | `fra_agent.py` |
| `langchain-ollama` | 1.1 | ✓ MIT | `ChatOllama` client for the local model | `fra_agent.py` |
| **Ollama** + `llama3.2` (3B) | any recent release | ✓ MIT / Llama 3.2 licence | local LLM for explanation wording (optional) | runtime |
| `fastapi` + `uvicorn` | 0.142 / 0.50 | ✓ MIT / BSD | A2A JSON-RPC and REST endpoints, Agent Card | `server.py` |
| `pandas`, `scikit-learn` | 3.0 / 1.9 | ✓ BSD | batch evaluation, shadow challenger model | `evaluate.py` |
| `pytest`, `httpx` | 9.1 / 0.28 | ✓ MIT / BSD | test suite, FastAPI TestClient | `tests/` |

**External services:** none are required. Ollama runs locally, and without it the agent uses the template wording. No paid APIs, no cloud calls.

```bash
cd fraud-risk-agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: local LLM for explanation wording
# 1) install Ollama from https://ollama.com/download   (Linux: curl -fsSL https://ollama.com/install.sh | sh)
ollama pull llama3.2          # ~2 GB; any Ollama chat model works (set FRA_OLLAMA_MODEL)
ollama serve                  # if it is not already running as a service
```

Configuration (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `FRA_OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama URL |
| `FRA_LLM_TIMEOUT` | `8` | seconds before falling back to the template (Pega allows 15 s per attempt) |
| `FRA_USE_LLM` | `1` | server only: `0` gives template-only wording |
| `FRA_API_TOKEN` | *(unset → all POSTs return 401)* | bearer token the caller must send |
| `FRA_PUBLIC_URL` | `http://localhost:8000` | URL advertised in the Agent Card |
| `FRA_LOG_LEVEL` | `INFO` | log level |

---

## 3. Decision logic & rules

### 3.1 Fraud indicators (spec §7.1). These are the only rules that change the score.

| Code | Fires when (request field) | Points | Floor |
|---|---|---|---|
| `VEHICLE_NOT_INSURED` | `vehicle_policy_mismatch == true` | +30 | 45 |
| `DUPLICATE_PATTERN` | `claim_history.potential_duplicate == true` | +40 | 70 |
| `MULTIPLE_RECENT_CLAIMS` | `claim_history.claims_last_24_months >= 2` | +25 | – |
| `EARLY_POLICY_CLAIM` | `claim_history.days_since_policy_start < 30` | +20 | – |
| `LATE_REPORTED` | `claim_history.late_reported == true` | +10 | – |

```text
fired       = indicators whose trigger is true          (returned in the order above)
risk_score  = max( min(100, Σ points(fired)),  max(floor(fired), default 0) )
confidence  = round( (policy_c + history_c) / 2 ),  each component 100 if its data was available, else 60
```

| Worked example (acceptance tests) | Fired | Score |
|---|---|---|
| AT-1 clean claim | – | 0 |
| AT-2 duplicate only | DUPLICATE_PATTERN | max(40, 70) = **70** |
| AT-3 vehicle not on policy | VEHICLE_NOT_INSURED | max(30, 45) = **45** |
| D: 2 recent claims + policy 18 days old | MULTIPLE_RECENT_CLAIMS, EARLY_POLICY_CLAIM | 25 + 20 = **45** |
| AT-6 all five | all | max(min(100, 125), 70) = **100** |

| policy available | history available | confidence |
|---|---|---|
| ✓ | ✓ | 100 |
| ✓ | ✗ | 80 |
| ✗ | ✓ | 80 |
| ✗ | ✗ | 60 (Pega escalates anything below 70 to a Senior Adjuster) |

**What Pega does with the score.** The agent never computes this; the table is for context (Blueprint v1.5 governance defaults). Low < 30 · Medium 30–59 · ≥ 60 escalate to Senior and recommend *Investigate* · ≥ 80 create a Fraud Investigation child case. A score is never a reason to reject a claim.

### 3.2 Contract rules → `FAILED`

The agent returns `FAILED` (with no score, confidence or flags) when any of these happen. The reason always names the offending field:

* a required field is missing, or has the wrong JSON type (`1` is not a boolean, `true` is not an integer)
* `loss.cause` is not one of `COLLISION, THEFT, VANDALISM, FIRE, NATURAL, GLASS, ANIMAL, OTHER`
* a date is not `YYYY-MM-DD`; `claim_id` is not `AC-nnnn`; `correlation_id` is not a UUID; a hash is not hex; a count is negative
* personal or damage data appears anywhere in the payload (name, DOB, phone, email, address, make/model/year, photos, plain policy number or VIN, damage estimate or severity)
* an unexpected exception occurs, or the self-check finds an out-of-range value

### 3.3 LLM-based reasoning: wording only, behind guardrails

* **Input:** plain text: the cause of loss, one line per fired indicator (the template sentence) and, if any data was missing, the incomplete-data sentence. The system prompt includes one worked example. The LLM **never** receives `loss.description` or the hashes, so the 8 prompt-injection rows in the dataset cannot reach it. Those rows are still detected by a regex and logged as a warning.
* **Settings:** `temperature=0`, `seed=42`, `num_predict=200`, 8-second timeout.
* **Called only when at least one indicator fired.** A clean claim gets the spec's sentence verbatim.
* **Output is rejected (and the template used instead) if it:** is shorter than 20 or longer than 800 characters; contains markup or a list; uses accusatory, judgemental or advisory words (*fraudulent, suspicious, unusual, warrants review, investigate, should…*); talks about the prompt itself (*key phrase, data note, input…*); misses the key phrase of a fired indicator; mentions an indicator that did not fire; contains **any number not present in the facts** (which blocks invented scores or counts); contains a hash fragment; or repeats 6 or more consecutive words of the claimant's description.
* Ollama down, model missing, timeout: all fall back to the template. Numbers and flags are identical either way.
* **Real-model check (CI job `ollama-smoke`):** runs `llama3.2` on 15 flagged claims. The first run showed why the strict checks matter: 11 of 15 answers talked about the prompt ("…which is a key phrase") or judged the claim ("warrant further review"). These patterns are now rejected and kept as regression tests. On GitHub's 4-core CPU runner each explanation took **5–12 s**, so with the 8 s timeout some calls fall back to the template; use a GPU or a smaller model (`llama3.2:1b`) if you want LLM wording on every claim.

### 3.4 How good are the rules? (`python evaluate.py`)

```text
Contract reproduction: 1500/1500 rows identical (risk_score, confidence, risk_flags, reasoning)

Rule engine vs fraud_confirmed (base rate 20.0%, 300 positives)
  PR-AUC 0.577   ROC-AUC 0.819
  score >= 30  Medium band or above            13.3% of claims   precision  69.3%   recall  46.0%
  score >= 60  Senior escalation (DT-01)        6.1% of claims   precision  70.7%   recall  21.7%
  score >= 80  Fraud Investigation referral     2.1% of claims   precision 100.0%   recall  10.7%

Shadow challenger (not used in responses), out-of-time test set: 450 claims reported 2025-11-06 to 2026-06-30
  rules       PR-AUC 0.549   ROC-AUC 0.794
  challenger  PR-AUC 0.668   ROC-AUC 0.836
```

**Reading:** the five rules are precise, but they catch only **46 %** of confirmed fraud at Medium or above, against the design's 80 % recall target (F-M05). A logistic-regression challenger that also uses `loss_cause` and the description text does better on held-out, later claims. The spec deliberately ships rules only until real SIU-confirmed labels exist. The challenger is therefore reported for comparison and never changes a response. Promoting it would need real labels, a fairness review and a contract change.

---

## 4. Implementation plan in phases

Each phase ends with a working artifact and a command that proves it. Later phases only add to earlier ones.

| Phase | Build | Working artifact | Exit criteria (command) |
|---|---|---|---|
| **0. Setup** | venv, `requirements.txt`, data in `data/` | importable package | `python -c "import fra_agent"` |
| **1. Ingestion & contract** | `row_to_request`, `read_csv_rows`, `validate_request`, `FAILED` responses | CLI that rejects bad payloads with the field name | `python -m pytest -k "invalid or personal or missing or non_object or unknown"` · `python fra_agent.py --no-llm --request examples/request_missing_field.json` |
| **2. Rule engine & confidence** | `fired_indicators`, `risk_score`, `compute_confidence`, self-check | deterministic scorer | `python -m pytest -k "acceptance or confidence or boundaries or at8"`: AT-1…AT-8 exact |
| **3. Reasoning (template, then LLM)** | `template_reasoning`; LangChain prompt → `ChatOllama`; `narrative_problems` guardrails; fallback | explanations for every claim, with or without Ollama | `python -m pytest -k "llm or wording"` (fake LLM, no Ollama needed) · `python fra_agent.py --request examples/request_all_indicators.json` (real Ollama) |
| **4. Testing & evaluation** | full pytest suite; `evaluate.py` (contract reproduction, metrics, shadow challenger) | regression suite + evaluation report | `python -m pytest` → 55 passed · `python evaluate.py` → 1500/1500 identical · CI runs both on every PR, plus the real-model smoke job |
| **5. A2A service** | `server.py`: Agent Card, bearer auth, JSON-RPC `message/send`, REST `/v1/assess`, `/health` | HTTP service Pega can call | `python -m pytest tests/test_server.py` · `uvicorn server:app` + the curl calls below |

**Before production** (out of scope for this build, per spec §13 and §15): validate real OAuth 2.0 JWTs instead of a static token; confirm the A2A version and DataPart shape with Pega (O-1); review LLM wording quality on a sample with adjusters (target: zero factual errors, F-M09); start collecting SIU-confirmed outcomes so a model can be trained and tested later.

---

## 5. Complete code

| File | Lines | What it is |
|---|---|---|
| [`fra_agent.py`](fra_agent.py) | ~500 | **The agent**: contract validation, rule engine, confidence, template and LLM reasoning with guardrails, self-check, CLI. Self-contained; this is the only file needed to score a claim. |
| [`server.py`](server.py) | ~130 | A2A / REST front door (FastAPI): Agent Card, bearer auth, JSON-RPC `message/send` |
| [`evaluate.py`](evaluate.py) | ~100 | Batch evaluation and the shadow scikit-learn challenger |
| [`tests/`](tests) | ~320 | 55 pytest tests: acceptance AT-1…AT-8, boundaries, validation, 1,500-row reproduction, LLM guardrails (fake model), server |
| [`../.github/workflows/fraud-risk-agent.yml`](../.github/workflows/fraud-risk-agent.yml) | ~70 | CI: tests + evaluation, and the `ollama-smoke` job with the real model |
| [`examples/`](examples) | – | Request payloads: clean, duplicate, all indicators, missing field |
| [`data/`](data) | – | Synthetic claims CSV and data dictionary |

### Run it

```bash
# Tests (no Ollama needed)
python -m pytest

# Score claims from the command line (add --no-llm to skip Ollama)
python fra_agent.py --request examples/request_duplicate.json
python fra_agent.py --csv data/fra_synthetic_claims_1500.csv --claim-id AC-3047   # an injection-test row
python fra_agent.py --csv data/fra_synthetic_claims_1500.csv --limit 5

# Evaluate the rules on all 1,500 claims
python evaluate.py

# Serve over A2A / REST
export FRA_API_TOKEN=change-me
uvicorn server:app --port 8000
curl -s localhost:8000/.well-known/agent.json
curl -s -X POST localhost:8000/v1/assess -H "Authorization: Bearer $FRA_API_TOKEN" \
     -H "Content-Type: application/json" -d @examples/request_duplicate.json
curl -s -X POST localhost:8000/a2a -H "Authorization: Bearer $FRA_API_TOKEN" -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0", "id": 1, "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "m-1",
             "parts": [{"kind": "data", "data": '"$(cat examples/request_duplicate.json)"'}]}}}'
```

Example response (AT-2):

```json
{
  "claim_id": "AC-1002",
  "status": "COMPLETED",
  "risk_score": 70,
  "confidence": 100,
  "risk_flags": ["DUPLICATE_PATTERN"],
  "reasoning": "One indicator was found: a potential duplicate of this claim was already flagged against the same policy and vehicle. No other indicators were found."
}
```

---

## Assumptions

Each of these was needed because the source files are silent or disagree:

1. **Which document wins.** The Build Specification docx is the contract. Where the PDF describes more (ECS service, DAA join, L2/L3 models, 26 indicators, four-part confidence, rescore/explain skills), this build follows the docx, whose §13 rejects those items explicitly.
2. **`claim_id` is echoed in the response.** Spec §5.2 and the data dictionary say to echo it, but the §6.2 response table lists only five fields. If Pega's schema rejects extra properties, delete the one line in `FraudRiskAgent.assess` that adds it.
3. **`FAILED` omits `risk_flags`.** Spec §6.2 says flags are "always" present, while §6.2.1 and AT-7 say to omit them on failure. The more specific rule and the acceptance test are followed.
4. **Risk bands** come from Blueprint v1.5 governance defaults (30 / 60 / 80), not the PDF's 40 / 70. Bands are Pega's job; they appear here only in the evaluation.
5. **Forbidden-field names** are not specified, so a deny-list of likely key names is used (`FORBIDDEN_KEYS`). A future optional field that happens to use one of those names (e.g. `model`) would be rejected; edit the list if Pega adds one.
6. **Formats.** `AC-nnnn` means `AC-` followed by 4 or more digits. Hashes must be hex with 16 or more characters (SHA-256 gives 64, per O-2). Date ordering and description length (V-02, V-19) are validated by Pega, not re-checked here.
7. **`LATE_REPORTED` wording** quotes the day gap computed from the two dates, with `LateReportDays = 30`. The flag itself always comes from Pega's boolean.
8. **Authentication.** A static bearer token (`FRA_API_TOKEN`) stands in for OAuth 2.0 client credentials. A2A is implemented as protocol 0.3 JSON-RPC `message/send` with a DataPart (open item O-1). The business payload is decoupled from the envelope, so either can change independently.
9. **LLM.** The default is `llama3.2` (3B); any Ollama chat model works. With `temperature=0` and a fixed seed, wording repeats on the same model and hardware. Byte-identical repeat responses (AT-8) are strictly guaranteed in template mode (`--no-llm` / `FRA_USE_LLM=0`), while scores and flags are identical in every mode. On CPU-only hardware the LLM adds 5–12 s per flagged claim, above the spec's 5 s target for a normal request (§10.1); the template-only mode meets it easily.
10. **Labels are synthetic.** `fraud_confirmed` is generated data, so the evaluation numbers show the method, not real-world performance.
