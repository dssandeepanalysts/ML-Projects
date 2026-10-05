"""
A2A front door for the Fraud Risk Agent (spec s3-s4). Business logic lives in fra_agent.py;
this file only handles transport, so a change of A2A envelope never touches scoring (spec O-1).

    export FRA_API_TOKEN=change-me        # local use; production uses OAuth 2.0 (FRA_OAUTH_* below)
    uvicorn server:app --port 8000

Endpoints
    GET  /.well-known/agent.json   Agent Card (public, for A2A discovery)
    POST /a2a                      A2A JSON-RPC 2.0, method message/send, claim in a DataPart
    POST /v1/assess                Plain REST binding of the same payload (Connect-REST fallback)
    GET  /health                   Liveness check

Authentication (spec s4), one of:
    OAuth 2.0 (production)  FRA_OAUTH_ISSUER, FRA_OAUTH_AUDIENCE, FRA_OAUTH_JWKS_URL, and optionally
                            FRA_OAUTH_SCOPE (default fraud.assess) and FRA_OAUTH_TOKEN_URL (Agent Card only)
    Static token (local)    FRA_API_TOKEN, used only when OAuth is not configured
The server refuses to start with neither, or with OAuth half configured.

Other environment: FRA_PUBLIC_URL, FRA_USE_LLM=1 to turn on Ollama wording (off by default: template
replies in ~1 ms, a CPU-only LLM adds seconds), plus the FRA_OLLAMA_MODEL / OLLAMA_HOST /
FRA_LLM_TIMEOUT settings in fra_agent.py.
"""
from __future__ import annotations

import functools
import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from fra_agent import AGENT_VERSION, LLM_TIMEOUT, FraudRiskAgent, build_ollama_llm, failed

logging.basicConfig(level=os.getenv("FRA_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per claim is enough
log = logging.getLogger("fra.server")

agent = FraudRiskAgent(llm=build_ollama_llm() if os.getenv("FRA_USE_LLM", "0") == "1" else None)

# ---------------------------------------------------------------- authentication (spec s4)
OAUTH_REQUIRED = ("FRA_OAUTH_ISSUER", "FRA_OAUTH_AUDIENCE", "FRA_OAUTH_JWKS_URL")
# Asymmetric only: rules out "none" and HS256 signed with the public key (algorithm confusion).
OAUTH_ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"]
JWKS_CACHE_SECONDS = 600  # reuse the provider's signing keys this long
JWKS_REFRESH_SECONDS = 60  # a token naming an unknown key refetches them at most this often


def oauth_settings() -> dict | None:
    """The OAuth settings, or None when OAuth is not (fully) configured. Read per call, like FRA_API_TOKEN."""
    if not all(os.getenv(name) for name in OAUTH_REQUIRED):
        return None
    return {
        "issuer": os.environ["FRA_OAUTH_ISSUER"],
        "audience": os.environ["FRA_OAUTH_AUDIENCE"],
        "jwks_url": os.environ["FRA_OAUTH_JWKS_URL"],
        "scope": os.getenv("FRA_OAUTH_SCOPE") or "fraud.assess",
        "token_url": os.getenv("FRA_OAUTH_TOKEN_URL", ""),
    }


def auth_problem() -> str | None:
    """Why the current settings cannot authenticate anyone, or None if they can."""
    missing = [name for name in OAUTH_REQUIRED if not os.getenv(name)]
    if 0 < len(missing) < len(OAUTH_REQUIRED):
        return f"OAuth is half configured: also set {', '.join(missing)}"
    if not missing and urlparse(os.environ["FRA_OAUTH_JWKS_URL"]).scheme not in ("http", "https"):
        return "FRA_OAUTH_JWKS_URL must be an http(s) URL"
    if missing and not os.getenv("FRA_API_TOKEN"):
        return "no authentication configured: set the FRA_OAUTH_* settings, or FRA_API_TOKEN for local use"
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Refuse to start rather than run a server that rejects (or worse, accepts) every call."""
    problem = auth_problem()
    if problem:
        raise RuntimeError(f"Refusing to start: {problem} (see DEPLOYMENT.md)")
    settings = oauth_settings()
    if settings and os.getenv("FRA_API_TOKEN"):
        log.warning("OAuth is configured, so FRA_API_TOKEN is ignored")
    if settings and urlparse(settings["jwks_url"]).scheme == "http":
        log.warning("FRA_OAUTH_JWKS_URL uses plain http: fine for local testing, use https in production")
    log.info("authentication: %s", "OAuth 2.0 JWT" if settings else "static token (local use only)")
    yield


@functools.lru_cache(maxsize=4)
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url, lifespan=JWKS_CACHE_SECONDS, timeout=3)


_last_refresh: dict[str, float] = {}
_refresh_lock = threading.Lock()


def signing_key(url: str, kid) -> jwt.PyJWK:
    """The provider's public key named by the token's kid. Only the configured JWKS URL is used, never
    a URL from the token. An unknown kid (the provider rotated its keys) refetches the keys, at most
    once per JWKS_REFRESH_SECONDS, so forged tokens cannot make us flood the provider."""
    client = _jwks_client(url)
    for refresh in (False, True):
        if refresh:
            with _refresh_lock:
                now = time.monotonic()
                if now - _last_refresh.get(url, float("-inf")) < JWKS_REFRESH_SECONDS:
                    break
                _last_refresh[url] = now
        key = next((k for k in client.get_signing_keys(refresh=refresh) if k.key_id == kid), None)
        if key is not None:
            return key
    raise jwt.InvalidTokenError("no signing key matches the token's kid")


def decode_access_token(token: str, settings: dict) -> dict:
    """Verify signature, expiry, issuer and audience; return the claims. Raises jwt.PyJWTError."""
    header = jwt.get_unverified_header(token)
    if header.get("alg") not in OAUTH_ALGORITHMS:  # checked before any key fetch
        raise jwt.InvalidAlgorithmError("algorithm not allowed")
    return jwt.decode(
        token,
        signing_key(settings["jwks_url"], header.get("kid")),
        algorithms=OAUTH_ALGORITHMS,
        audience=settings["audience"],
        issuer=settings["issuer"],
        leeway=30,  # seconds of clock skew between us and the provider
        options={"require": ["exp", "iss", "aud"]},
    )


def granted_scopes(claims: dict) -> set[str]:
    """Scopes as Keycloak ("scope": "a b"), Okta ("scp": [...]) and Entra ID app roles ("roles": [...]) send them."""
    granted = set()
    for name in ("scope", "scp", "roles"):
        value = claims.get(name)
        if isinstance(value, str):
            granted.update(value.split())
        elif isinstance(value, list):
            granted.update(v for v in value if isinstance(v, str))
    return granted


def _reject(status: int, detail: str, error: str = "invalid_token") -> HTTPException:
    return HTTPException(status_code=status, detail=detail, headers={"WWW-Authenticate": f'Bearer error="{error}"'})


bearer = HTTPBearer(auto_error=False)


async def authenticate(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> float:
    """Reject unauthenticated calls outright (spec s4). Returns the arrival time, so time spent here
    (a key fetch, a wait for a worker thread) counts against Pega's 15 s."""
    arrived = time.monotonic()
    problem = auth_problem()
    if problem:  # the startup check normally stops this earlier
        log.error("cannot authenticate: %s", problem)
        raise _reject(503, "Authentication is not configured.", "temporarily_unavailable")
    settings = oauth_settings()
    if creds is None:
        raise _reject(401, "Missing or invalid bearer token.")
    if settings is None:
        # Compare bytes: compare_digest raises on non-ASCII str, which would turn a bad token into a 500.
        if not secrets.compare_digest(creds.credentials.encode(), os.environ["FRA_API_TOKEN"].encode()):
            raise _reject(401, "Missing or invalid bearer token.")
        return arrived
    try:  # in a worker thread: a key fetch must not block the event loop
        claims = await run_in_threadpool(decode_access_token, creds.credentials, settings)
    except (jwt.PyJWKClientError, jwt.PyJWKSetError) as exc:  # the provider's key endpoint, not the token
        log.error("cannot fetch signing keys from FRA_OAUTH_JWKS_URL: %s", type(exc).__name__)
        raise _reject(503, "Cannot verify tokens right now.", "temporarily_unavailable") from None
    except jwt.PyJWTError as exc:  # the reason is logged, never the token or the caller's text
        log.info("token rejected: %s", type(exc).__name__)
        raise _reject(401, "Missing or invalid bearer token.") from None
    except Exception as exc:  # anything unexpected while checking: fail closed
        log.error("token check failed: %s", type(exc).__name__)
        raise _reject(503, "Cannot verify tokens right now.", "temporarily_unavailable") from None
    if settings["scope"] not in granted_scopes(claims):
        log.info("token rejected: missing scope %s", settings["scope"])
        raise _reject(403, "The token does not grant the required scope.", "insufficient_scope")
    return arrived


# ---------------------------------------------------------------- app
app = FastAPI(title="Fraud Risk Agent", version=AGENT_VERSION, lifespan=lifespan)
PUBLIC_URL = os.getenv("FRA_PUBLIC_URL", "http://localhost:8000")


def security_schemes() -> dict:
    """How callers authenticate, as the Agent Card advertises it (A2A securitySchemes, OpenAPI shapes)."""
    settings = oauth_settings()
    if settings is None:
        return {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}, "security": [{"bearer": []}]}
    if settings["token_url"]:
        flows = {"clientCredentials": {"tokenUrl": settings["token_url"],
                                       "scopes": {settings["scope"]: "Assess the fraud risk of a claim"}}}
        scheme = {"type": "oauth2", "flows": flows}
    else:  # no token URL configured: point at the provider's discovery document instead
        scheme = {"type": "openIdConnect",
                  "openIdConnectUrl": settings["issuer"].rstrip("/") + "/.well-known/openid-configuration"}
    return {"securitySchemes": {"oauth2": scheme}, "security": [{"oauth2": [settings["scope"]]}]}


def build_agent_card() -> dict:
    return {
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
        **security_schemes(),
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
    return build_agent_card()


# Pega allows 15 s per attempt. A request that waited this long for a free worker thread is
# answered with the template (same score and flags), so the LLM cannot push it past 15 s.
QUEUE_BUDGET = max(0.0, 14.0 - LLM_TIMEOUT)
template_agent = FraudRiskAgent()
if agent.llm is not None and QUEUE_BUDGET < 2:
    log.warning("FRA_LLM_TIMEOUT=%.0f s leaves %.0f s of Pega's 15 s for queueing: most replies will use the "
                "template. Keep it at 10 s or less.", LLM_TIMEOUT, QUEUE_BUDGET)


def _assess(payload, arrived: float) -> dict:
    return (template_agent if time.monotonic() - arrived > QUEUE_BUDGET else agent).assess(payload)


@app.post("/a2a")
async def a2a(request: Request, arrived: float = Depends(authenticate)) -> dict:
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


@app.post("/v1/assess")
async def assess(request: Request, arrived: float = Depends(authenticate)) -> dict:
    try:
        payload = await request.json()
    except (ValueError, RecursionError):  # invalid or too deeply nested JSON
        payload = None  # the agent answers FAILED: "The request body must be a JSON object."
    return await run_in_threadpool(_assess, payload, arrived)
