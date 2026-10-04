"""
A2A front door for the Fraud Risk Agent (spec s3-s4). Business logic lives in fra_agent.py;
this file only handles transport, so a change of A2A envelope never touches scoring (spec O-1).

    export FRA_API_TOKEN=change-me        # bearer token the caller must send
    uvicorn server:app --port 8000

Endpoints
    GET  /.well-known/agent.json   Agent Card (public, for A2A discovery)
    POST /a2a                      A2A JSON-RPC 2.0, method message/send, claim in a DataPart
    POST /v1/assess                Plain REST binding of the same payload (Connect-REST fallback)
    GET  /health                   Liveness check

Environment: FRA_API_TOKEN (required for POSTs), FRA_PUBLIC_URL, FRA_USE_LLM=1 to turn on
Ollama wording (off by default: template replies in ~1 ms, a CPU-only LLM adds seconds), plus the FRA_OLLAMA_MODEL / OLLAMA_HOST / FRA_LLM_TIMEOUT settings in fra_agent.py.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from fra_agent import AGENT_VERSION, LLM_TIMEOUT, FraudRiskAgent, build_ollama_llm, failed

logging.basicConfig(level=os.getenv("FRA_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per claim is enough
log = logging.getLogger("fra.server")

agent = FraudRiskAgent(llm=build_ollama_llm() if os.getenv("FRA_USE_LLM", "0") == "1" else None)
app = FastAPI(title="Fraud Risk Agent", version=AGENT_VERSION)

if not os.getenv("FRA_API_TOKEN"):
    log.warning("FRA_API_TOKEN is not set: every assessment call will be rejected with 401")

PUBLIC_URL = os.getenv("FRA_PUBLIC_URL", "http://localhost:8000")
AGENT_CARD = {
    "protocolVersion": "0.3.0",
    "name": "Fraud Risk Agent",
    "description": "Advisory fraud-indicator scoring for Claim Assist auto claims. "
    "It never approves, rejects or resolves a claim.",
    "url": f"{PUBLIC_URL}/a2a",
    "preferredTransport": "JSONRPC",
    "version": AGENT_VERSION,  # Pega stores this as AgentCardVersion
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
    "security": [{"bearer": []}],
    "skills": [
        {
            "id": "assess_fraud_risk",
            "name": "Assess fraud risk",
            "description": "Scores five history-based fraud indicators and returns risk_score, "
            "confidence, risk_flags and a neutral explanation.",
            "tags": ["fraud", "claims", "insurance"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
        }
    ],
}

bearer = HTTPBearer(auto_error=False)


async def require_token(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    """Reject unauthenticated calls outright (spec s4). Production: validate the OAuth 2.0 JWT instead."""
    expected = os.getenv("FRA_API_TOKEN")
    # Compare bytes: compare_digest raises on non-ASCII str, which would turn a bad token into a 500.
    if not expected or creds is None or not secrets.compare_digest(creds.credentials.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.",
                            headers={"WWW-Authenticate": "Bearer"})


def rpc_error(rpc_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception) -> JSONResponse:
    """Last resort (spec s11): never a bare 500 or a dropped connection; Pega needs a parseable body."""
    log.error("unhandled %s on %s", type(exc).__name__, request.url.path)
    if request.url.path == "/a2a":
        return JSONResponse(rpc_error(None, -32603, "Internal error"), status_code=500)
    return JSONResponse(failed("An internal error occurred while processing this request."), status_code=500)


@app.get("/health")
async def health() -> dict:  # async: never queues behind assessments in the thread pool
    """Liveness for Docker/Kubernetes. Stays "ok" when Ollama is down: the template still answers."""
    return {"status": "ok", "version": AGENT_VERSION, "llm_wording": "on" if agent.llm is not None else "off"}


@app.get("/.well-known/agent.json")
@app.get("/.well-known/agent-card.json")  # path used by newer A2A versions
def agent_card() -> dict:
    return AGENT_CARD


# Pega allows 15 s per attempt. A request that waited this long for a free worker thread is
# answered with the template (same score and flags), so the LLM cannot push it past 15 s.
QUEUE_BUDGET = max(0.0, 14.0 - LLM_TIMEOUT)
template_agent = FraudRiskAgent()
if agent.llm is not None and QUEUE_BUDGET < 2:
    log.warning("FRA_LLM_TIMEOUT=%.0f s leaves %.0f s of Pega's 15 s for queueing: most replies will use the "
                "template. Keep it at 10 s or less.", LLM_TIMEOUT, QUEUE_BUDGET)


def _assess(payload, arrived: float) -> dict:
    return (template_agent if time.monotonic() - arrived > QUEUE_BUDGET else agent).assess(payload)


@app.post("/a2a", dependencies=[Depends(require_token)])
async def a2a(request: Request) -> dict:
    arrived = time.monotonic()
    try:
        rpc = await request.json()
    except (ValueError, RecursionError):  # invalid or too deeply nested JSON
        return rpc_error(None, -32700, "Parse error")
    if not isinstance(rpc, dict) or rpc.get("jsonrpc") != "2.0" or "method" not in rpc:
        return rpc_error(rpc.get("id") if isinstance(rpc, dict) else None, -32600, "Invalid Request")
    if rpc["method"] != "message/send":
        return rpc_error(rpc.get("id"), -32601, "Method not found")
    try:
        message = rpc["params"]["message"]
        claim = next(p["data"] for p in message["parts"] if p.get("kind") == "data")
    except (KeyError, TypeError, AttributeError, StopIteration):
        return rpc_error(rpc.get("id"), -32602, "Invalid params: expected a message with a data part")

    result = await run_in_threadpool(_assess, claim, arrived)
    message_id = message.get("messageId")
    return {
        "jsonrpc": "2.0",
        "id": rpc.get("id"),
        "result": {
            "kind": "message",
            "role": "agent",
            # Derived from the request's messageId, so a repeat call gets an identical reply.
            "messageId": str(uuid.uuid5(uuid.NAMESPACE_URL, message_id if isinstance(message_id, str) else "")),
            "parts": [{"kind": "data", "data": result}],
        },
    }


@app.post("/v1/assess", dependencies=[Depends(require_token)])
async def assess(request: Request) -> dict:
    arrived = time.monotonic()
    try:
        payload = await request.json()
    except (ValueError, RecursionError):  # invalid or too deeply nested JSON
        payload = None  # the agent answers FAILED: "The request body must be a JSON object."
    return await run_in_threadpool(_assess, payload, arrived)
