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
 (or Connect-REST /v1/assess)       OAuth 2.0 JWT │
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
5. **Explain.** A template sentence is always built; it reproduces the dataset's `reasoning` column exactly. If a local LLM is available and at least one indicator fired, LangChain asks Ollama to reword **the facts only**. The output must pass the guardrail checks in §3.3, within a hard time limit, or the template is used instead.
6. **Self-check and respond.** Out-of-range values become `FAILED`, never a malformed `COMPLETED`. The agent is stateless and has no side effects, so Pega's retries are safe.

**Why not a tool-calling (ReAct) agent?** The spec requires that every number and flag come from plain code (§7, §12.4), that repeat calls return identical results (AT-8), and that each call answer inside Pega's 15-second timeout. A local 3B model choosing tools would add nondeterminism, latency and an injection surface without improving the score. So this is an agent in the A2A sense: an autonomous service with a published Agent Card and skill. The LLM's job is limited to wording.

---

## 2. Required tools & integrations

| Component | Version tested | Free / OSS | Used for | Where |
|---|---|---|---|---|
| Python | 3.11 (needs 3.10+) | ✓ | runtime | all |
| `langchain-core` | 1.6 | ✓ MIT | chat prompt template, streaming interface the agent drives | `fra_agent.py` |
| `langchain-ollama` | 1.1 | ✓ MIT | `ChatOllama` client for the local model | `fra_agent.py` |
| **Ollama** + `llama3.2` (3B) | any recent release | ✓ MIT / Llama 3.2 licence | local LLM for explanation wording (optional) | runtime |
| `fastapi` + `uvicorn` | 0.142 / 0.50 | ✓ MIT / BSD | A2A JSON-RPC and REST endpoints, Agent Card | `server.py` |
| `PyJWT[crypto]` | 2.13 / 2.15 | ✓ MIT | checks OAuth 2.0 access tokens: signature (provider's JWKS), expiry, issuer, audience, scope | `server.py` |
| `pyngrok` + **ngrok** (free account) | 8.1 | ✓ MIT / free tier | public HTTPS address for the Pega test notebook (optional) | `pega_a2a_ngrok.ipynb` |
| **Keycloak** (Docker) | 26.8 | ✓ Apache 2.0 | local identity provider to try OAuth end to end (optional, testing only) | `docker-compose.yml` |
| `pandas`, `scikit-learn` | 3.0 / 1.9 | ✓ BSD | batch evaluation, shadow challenger model | `evaluate.py` |
| `pytest`, `httpx` | 9.1 / 0.28 | ✓ MIT / BSD | test suite, FastAPI TestClient | `tests/` |

**External services:** none are required. Ollama runs locally, and without it the agent uses the template wording. In production the agent fetches signing keys from your identity provider (OAuth 2.0). No paid APIs, no cloud calls.

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
| `FRA_LLM_TIMEOUT` | `8` | total seconds one LLM explanation may take before the template is used (Pega allows 15 s per attempt) |
| `FRA_USE_LLM` | `0` | server only: `1` turns on Ollama wording (the CLI uses it unless `--no-llm`) |
| `FRA_OAUTH_ISSUER`, `FRA_OAUTH_AUDIENCE`, `FRA_OAUTH_JWKS_URL` | *(unset)* | server: OAuth 2.0 mode, set all three; see [DEPLOYMENT.md §4](DEPLOYMENT.md#4-oauth-20-login) |
| `FRA_OAUTH_SCOPE`, `FRA_OAUTH_TOKEN_URL` | `fraud.assess`, *(unset)* | server: scope the token must carry; token URL shown in the Agent Card |
| `FRA_OAUTH_ALLOW_HTTP` | *(unset)* | server: `1` allows a plain-http key URL, for the local Keycloak only (production needs https) |
| `FRA_API_TOKEN` | *(unset)* | server, local use only: static bearer token, ignored in OAuth mode. With neither, the server refuses to start |
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

The agent returns `FAILED` (with no score, confidence or flags) when any of these happen. For a contract violation the reason names the offending field; internal errors and self-check failures give a generic reason (spec §11). Every contract violation is also logged, by field name only:

* a required field is missing, or has the wrong JSON type (`1` is not a boolean, `true` is not an integer)
* `loss.cause` is not one of `COLLISION, THEFT, VANDALISM, FIRE, NATURAL, GLASS, ANIMAL, OTHER`
* a date is not `YYYY-MM-DD`; `claim_id` is not `AC-nnnn`; `correlation_id` is not a UUID; a hash is not hex; a count is negative
* a key for personal or damage data appears anywhere in the payload, matched in any case style (`date_of_birth`, `dateOfBirth`, `DateOfBirth`): name, first/last name, date of birth, phone, email, address, vehicle make/model/year, photos, plain policy number or VIN, the damage agent's estimate, severity or confidence score
* an unexpected exception occurs, or the self-check finds an out-of-range value

### 3.3 LLM-based reasoning: wording only, behind guardrails

* **Input:** plain text: the cause of loss (left out when it is OTHER, so the model cannot write "this other claim"), the number of indicators, one line per fired indicator (the template sentence) and, if any data was missing, the incomplete-data sentence to copy at the end. The system prompt includes two worked examples. The LLM **never** receives `loss.description` or the hashes, so the 8 prompt-injection rows in the dataset cannot reach it. Those rows are still detected by a regex and logged as a warning.
* **Settings:** `temperature=0`, `seed=42`, `num_predict=200`, and an 8-second limit on the whole explanation (not just on each streamed chunk).
* **Called only when at least one indicator fired.** A clean claim gets the spec's sentence verbatim.
* **Output is rejected (and the template used instead) unless it has the template's structure:** an optional opener stating the right count (*"One indicator was found on this theft claim:"*, the cause of loss allowed only there); every fired indicator in its **exact template wording** (so each number stays attached to its own fact); optionally *"No other indicators were found"*, but only after the last indicator; and, exactly when data was missing, the missing-data sentence word for word as the **final** sentence. Once those parts are removed, only joining words may remain (*and, also, additionally, in addition, while…*). This closed structure is what stops negated indicators, swapped numbers, invented facts or indicators, a wrong count or cause, and scores. Further checks reject: any character outside plain prose (letters, digits, `, . ; : -`), so other scripts, look-alike letters, quotes, `?`, `!`, bullets and emoji are out; under 20 or over 800 characters; lists; accusatory, judgemental or advisory wording matched by word stem (*fraud…, suspect…, fake…, investigate, must, deny, reject…*); talk about the prompt itself; a number not in the facts; a hash fragment; 6 or more consecutive words from the claimant's description. The template's own wording passes all of these for every flagged claim in the dataset, which is checked by a test.
* Ollama down, model missing, too slow: all fall back to the template, and a call that times out while still queued never reaches the model. Numbers and flags are identical either way.
* **Real-model check (CI job `ollama-smoke`):** runs `llama3.2` on 15 flagged claims. The first run showed why the strict checks matter: 11 of 15 answers talked about the prompt ("…which is a key phrase") or judged the claim ("warrant further review"). A second run, with a new prompt, fixed that but showed the model miscounting ("Two indicators were found" when one fired) in 6 of 15 answers, so stated counts are now checked too. Every pattern seen is kept as a regression test, and the 15 correct answers from the latest run are kept as tests that must keep passing. On GitHub's 4-core CPU runner each explanation took **3–9 s** (17 s on a cold start), so with the 8 s limit some calls fall back to the template; use a GPU or a smaller model (`llama3.2:1b`) if you want LLM wording on every claim.

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
| **1. Ingestion & contract** | `row_to_request`, `read_csv_rows`, `validate_request`, `FAILED` responses | CLI that rejects bad payloads with the field name | `python -m pytest -k "invalid or personal or missing_field or non_object or unknown or nested"` (31 tests) · `python fra_agent.py --no-llm --request examples/request_missing_field.json` |
| **2. Rule engine & confidence** | `fired_indicators`, `risk_score`, `compute_confidence`, self-check | deterministic scorer | `python -m pytest -k "acceptance or confidence or boundaries or at7 or at8 or out_of_range or internal_error"` (21 tests): AT-1…AT-8 exact |
| **3. Reasoning (template, then LLM)** | `template_reasoning`; LangChain prompt → `ChatOllama`; `narrative_problems` guardrails; fallback | explanations for every claim, with or without Ollama | `python -m pytest -k "llm or llama or wording or missing_data or template_wording or chunks or queued or stream"` (72 tests, fake LLM, no Ollama needed) · `python fra_agent.py --request examples/request_all_indicators.json` (real Ollama) |
| **4. Testing & evaluation** | full pytest suite; `evaluate.py` (contract reproduction, metrics, shadow challenger) | regression suite + evaluation report | `python -m pytest` → 232 passed · `python evaluate.py` → 1500/1500 identical (exits non-zero otherwise) · CI runs both on every PR, plus the real-model smoke job |
| **5. A2A service** | `server.py`: Agent Card, OAuth 2.0 JWT checks (or a static token locally), JSON-RPC `message/send`, REST `/v1/assess`, `/health` | HTTP service Pega can call | `python -m pytest tests/test_server.py tests/test_oauth.py` (106 tests) · `uvicorn server:app` + the curl calls below |
| **6. Deployment** | `Dockerfile` (non-root, health check, graceful shutdown), `docker-compose.yml` (optional Ollama and Keycloak profiles), `.env.example`, [DEPLOYMENT.md](DEPLOYMENT.md) | container Pega can reach | `docker compose up -d --build --wait` → `(healthy)`; CI job `docker` builds the image and runs the stack, incl. the Ollama profile with the real model; CI job `oauth` calls the agent with a real Keycloak token |

**Before production** (out of scope for this build, per spec §13 and §15; steps in [DEPLOYMENT.md §5](DEPLOYMENT.md#5-before-production-steps-that-need-your-infrastructure-details)): point the `FRA_OAUTH_*` settings at your identity provider; put HTTPS in front; confirm the A2A version and DataPart shape with Pega (O-1); review LLM wording quality on a sample with adjusters (target: zero factual errors, F-M09); start collecting SIU-confirmed outcomes so a model can be trained and tested later.

---

## 5. Complete code

| File | Lines | What it is |
|---|---|---|
| [`fra_agent.py`](fra_agent.py) | ~640 | **The agent**: contract validation, rule engine, confidence, template and LLM reasoning with guardrails, self-check, CLI. Self-contained; this is the only file needed to score a claim. |
| [`server.py`](server.py) | ~640 | A2A / REST front door (FastAPI): Agent Card, OAuth 2.0 JWT checks (static token for local use), JSON-RPC `message/send` |
| [`evaluate.py`](evaluate.py) | ~100 | Batch evaluation and the shadow scikit-learn challenger |
| [`tests/`](tests) | ~1,250 | 232 pytest tests: acceptance AT-1…AT-8, boundaries, validation, self-check, log hygiene, 1,500-row reproduction, LLM guardrails (fake model, real llama3.2 outputs, deadline), server, OAuth (forged, expired and wrong-audience tokens, key rotation, provider outages against a real local key endpoint, startup check) |
| [`../.github/workflows/fraud-risk-agent.yml`](../.github/workflows/fraud-risk-agent.yml) | ~155 | CI: tests + evaluation, the `ollama-smoke` job with the real model, the `docker` job (builds the image and runs the Compose stack, incl. the Ollama profile) and the `oauth` job (real Keycloak token) |
| [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml), [`.env.example`](.env.example) | – | Container image (non-root, health check) and Compose stack with optional Ollama and Keycloak; see [DEPLOYMENT.md](DEPLOYMENT.md) |
| [`keycloak/`](keycloak) | – | Test realm for the local Keycloak (`claims-realm.json`) and `get-token.sh`, which fetches a token the way Pega will |
| [`pega_a2a_ngrok.ipynb`](pega_a2a_ngrok.ipynb) | – | Jupyter/Colab notebook for a quick Pega test: runs the agent, gives it a public HTTPS address with ngrok (free account), and issues the OAuth 2.0 Client ID, Client Secret, Access Token Endpoint and Scope that Pega needs |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | – | Short deployment guide: run, optional LLM, health checks, OAuth 2.0 (Keycloak, Entra ID, Okta), production checklist (HTTPS, Pega) |
| [`examples/`](examples) | – | Request payloads: clean, duplicate, all indicators, missing field |
| [`data/`](data) | – | Synthetic claims CSV and data dictionary |

### Run it

```bash
# Tests (no Ollama needed)
python -m pytest

# Score claims from the command line (add --no-llm to skip Ollama). With no options at all
# (e.g. an IDE's Run button), it scores the example requests in examples/.
python fra_agent.py --request examples/request_duplicate.json
python fra_agent.py --csv data/fra_synthetic_claims_1500.csv --claim-id AC-3047   # an injection-test row
python fra_agent.py --csv data/fra_synthetic_claims_1500.csv --limit 5

# Evaluate the rules on all 1,500 claims
python evaluate.py

# Serve over A2A / REST (or with Docker: see DEPLOYMENT.md, which also covers OAuth 2.0)
export FRA_API_TOKEN=change-me          # local use only; production uses the FRA_OAUTH_* settings
uvicorn server:app --port 8000          # template wording; FRA_USE_LLM=1 uvicorn ... to use Ollama
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
5. **Forbidden-field names** are not specified, so a deny-list of likely key names is used (`FORBIDDEN_KEYS`), compared after normalising case and separators, and including the Blueprint's own property names (`PolicyNumber`, `DateOfBirth`, `ConfidenceScore`). A future optional field that happens to use one of those names (e.g. `model`) would be rejected; edit the list if Pega adds one. `correlation_id` must be a canonical UUID v4 and `claim_id` uses ASCII digits only.
6. **Formats.** `AC-nnnn` means `AC-` followed by 4 or more digits. Hashes must be hex with 16 or more characters (SHA-256 gives 64, per O-2). Date ordering and description length (V-02, V-19) are validated by Pega, not re-checked here.
7. **`LATE_REPORTED` wording** quotes the day gap computed from the two dates, with `LateReportDays = 30`. The flag itself always comes from Pega's boolean.
8. **Authentication.** Pega gets tokens with OAuth 2.0 client credentials, and the agent checks each one: signature against the provider's published keys, expiry, issuer, audience and the `fraud.assess` scope (spec §4 names OAuth 2.0 but no provider, scope or audience, so these are configurable). A static bearer token (`FRA_API_TOKEN`) remains for local use only. A2A is implemented as protocol 0.3 JSON-RPC `message/send` (open item O-1). The claim may arrive as a DataPart, as JSON text, or as one `name: value` line per field, which is how a Pega AI agent sent it in testing (flat names, day-first dates); the older `"type"` key also works. Replies carry the result as a DataPart and as JSON text. The business payload is decoupled from the envelope, so either can change independently.
9. **LLM.** The default is `llama3.2` (3B); any Ollama chat model works. With `temperature=0` and a fixed seed, wording repeats on the same model and hardware. Byte-identical repeat responses (AT-8) are strictly guaranteed in template mode (`--no-llm`, and the server default), while scores and flags are identical in every mode. On CPU-only hardware the model takes about 3–9 s per flagged claim (more on a cold start); the agent caps this at `FRA_LLM_TIMEOUT` (8 s) and then uses the template. That is above the spec's 5 s target for a normal request (§10.1), while the template-only mode meets it easily, so the server runs template-only unless `FRA_USE_LLM=1` (e.g. on a GPU host).
10. **Labels are synthetic.** `fraud_confirmed` is generated data, so the evaluation numbers show the method, not real-world performance.
