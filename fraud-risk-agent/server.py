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
    OAuth 2.0 (production)  FRA_OAUTH_ISSUER, FRA_OAUTH_AUDIENCE, FRA_OAUTH_JWKS_URL (https), and optionally
                            FRA_OAUTH_SCOPE (default fraud.assess), FRA_OAUTH_TOKEN_URL (Agent Card only)
                            and FRA_OAUTH_ALLOW_HTTP=1 (an http JWKS URL, for local testing only)
    Static token (local)    FRA_API_TOKEN, used only when OAuth is not configured
The server refuses to start with neither, or with OAuth half configured.

Other environment: FRA_PUBLIC_URL, FRA_USE_LLM=1 to turn on Ollama wording (off by default: template
replies in ~1 ms, a CPU-only LLM adds seconds), plus the FRA_OLLAMA_MODEL / OLLAMA_HOST /
FRA_LLM_TIMEOUT settings in fra_agent.py.
"""
from __future__ import annotations

import asyncio
import datetime
import functools
import json
import logging
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from fra_agent import AGENT_VERSION, LLM_TIMEOUT, REQUIRED_FIELDS, FraudRiskAgent, build_ollama_llm, failed

logging.basicConfig(level=os.getenv("FRA_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per claim is enough
log = logging.getLogger("fra.server")

agent = FraudRiskAgent(llm=build_ollama_llm() if os.getenv("FRA_USE_LLM", "0") == "1" else None)

# ---------------------------------------------------------------- authentication (spec s4)
OAUTH_REQUIRED = ("FRA_OAUTH_ISSUER", "FRA_OAUTH_AUDIENCE", "FRA_OAUTH_JWKS_URL")
# Asymmetric only: rules out "none" and HS256 signed with the public key (algorithm confusion).
OAUTH_ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"]
LEEWAY_SECONDS = 30  # clock skew allowed between us and the provider (exp, nbf, iat)


def _env(name: str) -> str:
    """A setting with stray whitespace removed (a pasted newline, a space): blank counts as unset."""
    return os.getenv(name, "").strip()


def oauth_settings() -> dict | None:
    """The OAuth settings, or None when OAuth is not (fully) configured. Read per call, like FRA_API_TOKEN."""
    if not all(_env(name) for name in OAUTH_REQUIRED):
        return None
    return {
        "issuer": _env("FRA_OAUTH_ISSUER"),
        "audience": _env("FRA_OAUTH_AUDIENCE"),
        "jwks_url": _env("FRA_OAUTH_JWKS_URL"),
        "scope": _env("FRA_OAUTH_SCOPE") or "fraud.assess",
        "token_url": _env("FRA_OAUTH_TOKEN_URL"),
    }


def auth_problem() -> str | None:
    """Why the current settings cannot authenticate anyone safely, or None if they can."""
    missing = [name for name in OAUTH_REQUIRED if not _env(name)]
    if 0 < len(missing) < len(OAUTH_REQUIRED):
        return f"OAuth is half configured: also set {', '.join(missing)}"
    if missing:
        return None if _env("FRA_API_TOKEN") else \
            "no authentication configured: set the FRA_OAUTH_* settings, or FRA_API_TOKEN for local use"
    url = urlparse(_env("FRA_OAUTH_JWKS_URL"))
    if url.scheme not in ("http", "https") or not url.hostname:
        return "FRA_OAUTH_JWKS_URL must be an https URL"
    # Whoever can change the keys in transit can mint tokens, so plain http needs an explicit opt-in.
    if url.scheme == "http" and _env("FRA_OAUTH_ALLOW_HTTP") != "1":
        return "FRA_OAUTH_JWKS_URL uses plain http: use https (FRA_OAUTH_ALLOW_HTTP=1 allows http for local testing)"
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Refuse to start rather than run a server that rejects (or worse, accepts) every call."""
    problem = auth_problem()
    if problem:
        raise RuntimeError(f"Refusing to start: {problem} (see DEPLOYMENT.md)")
    settings = oauth_settings()
    if settings is None:
        log.warning("authentication: static token (FRA_API_TOKEN), for local use only; "
                    "set the FRA_OAUTH_* settings in production")
    else:
        log.info("authentication: OAuth 2.0 JWT (issuer %s, audience %s)", settings["issuer"], settings["audience"])
        if _env("FRA_API_TOKEN"):
            log.warning("OAuth is configured, so FRA_API_TOKEN is ignored")
        if urlparse(settings["jwks_url"]).scheme == "http":
            log.warning("FRA_OAUTH_JWKS_URL uses plain http (FRA_OAUTH_ALLOW_HTTP=1): for local testing only")
    yield


# ---------------------------------------------------------------- the provider's signing keys (JWKS)
JWKS_FRESH_SECONDS = 600  # keys older than this are refetched (in the background while they still work)
JWKS_RETRY_SECONDS = 10  # at most one fetch per this interval, whatever asks for it
JWKS_STALE_SECONDS = 3600  # while the provider is unreachable, keep using the last keys this long
JWKS_TIMEOUT = 3.0  # seconds a request waits for a fetch (also urllib's limit for each network step)
JWKS_FETCH_SECONDS = 10  # a whole download is abandoned after this long, however slowly the bytes arrive
JWKS_MAX_BYTES = 1_000_000

# One fetch at a time, in its own thread. Requests wait for it on the event loop, so a slow key endpoint
# never ties up the worker threads that answer Pega.
_jwks_fetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fra-jwks")


class KeysUnavailable(Exception):
    """The provider's keys could not be fetched: the provider's problem, not the token's (503)."""


class UnknownSigningKey(jwt.InvalidTokenError):
    """The token names a key (kid) that the provider does not publish (401)."""


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # use the provider's jwks_uri exactly as configured
        return None


def fetch_signing_keys(url: str) -> dict[str, dict]:
    """Download the JWK Set from the configured URL; return {kid: JWK} for its usable signing keys.
    Raises KeysUnavailable with a reason fit for the log (no token data can reach it).
    The default opener honours HTTPS_PROXY / NO_PROXY and SSL_CERT_FILE."""
    if urlparse(url).scheme not in ("http", "https"):  # urllib would also open file:// and ftp://
        raise KeysUnavailable("not an http(s) URL")
    deadline = time.monotonic() + JWKS_FETCH_SECONDS
    try:
        with urllib.request.build_opener(_NoRedirects).open(url, timeout=JWKS_TIMEOUT) as response:
            body = b""
            while len(body) <= JWKS_MAX_BYTES:
                if time.monotonic() > deadline:
                    raise KeysUnavailable(f"the key endpoint took longer than {JWKS_FETCH_SECONDS:g} s")
                chunk = response.read1(65536)  # what has arrived so far
                if not chunk:
                    break
                body += chunk
    except urllib.error.HTTPError as exc:  # includes redirects, which are not followed
        exc.close()
        raise KeysUnavailable(f"HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError) as exc:  # DNS, connection, TLS certificate, timeout
        raise KeysUnavailable(str(getattr(exc, "reason", exc))) from None
    if len(body) > JWKS_MAX_BYTES:
        raise KeysUnavailable("the response is larger than 1 MB")
    try:
        candidates = [k for k in json.loads(body)["keys"] if isinstance(k, dict)]
    except (ValueError, TypeError, KeyError, RecursionError):
        raise KeysUnavailable("the response is not a JSON Web Key Set") from None
    keys = {}
    for jwk in candidates:
        if isinstance(jwk.get("kid"), str) and jwk.get("use", "sig") == "sig":
            try:
                jwt.PyJWK(jwk)  # skip key types we cannot use
            except (jwt.PyJWTError, ValueError, TypeError, KeyError):
                continue
            keys[jwk["kid"]] = jwk
    if not keys:
        raise KeysUnavailable("the key set has no usable signing keys")
    return keys


class ProviderKeys:
    """The provider's signing keys, cached. They come only from FRA_OAUTH_JWKS_URL, never from a URL in a
    token. At most one fetch runs at a time and one starts per JWKS_RETRY_SECONDS, so forged tokens cannot
    make the agent flood the provider; and the last good keys are kept while the provider is unreachable."""

    def __init__(self, url: str):
        self.url = url
        self.keys: dict[str, dict] = {}
        self.fetched_at = float("-inf")  # last successful fetch
        self.attempted_at = float("-inf")  # last fetch started
        self.fetch: Future | None = None
        self.slow_fetch: Future | None = None  # the stalled fetch already logged
        self.last_error = "no keys fetched yet"
        self.lock = threading.Lock()

    async def get(self, kid) -> dict:
        """The JWK for this kid. Raises UnknownSigningKey or KeysUnavailable."""
        with self.lock:
            age = time.monotonic() - self.fetched_at
            key = self.keys.get(kid) if age < JWKS_STALE_SECONDS else None
            fetch = self._start_fetch() if key is None or age >= JWKS_FRESH_SECONDS else None
        if key is not None:
            return key  # if the keys are getting old, the fetch just started runs in the background
        if fetch is not None:  # a new kid (the provider rotated its keys) or no keys yet: wait for the fetch
            waiter = asyncio.wrap_future(fetch)  # waiting on the event loop costs no worker thread
            waiter.add_done_callback(lambda w: w.cancelled() or w.exception())  # the fetch logs its own failure
            done, _ = await asyncio.wait([waiter], timeout=JWKS_TIMEOUT)  # never cancels the shared fetch
            if not done:
                reason = f"the key endpoint did not answer within {JWKS_TIMEOUT:g} s"
                with self.lock:
                    first, self.slow_fetch = self.slow_fetch is not fetch, fetch
                if first:  # once per stalled fetch, not once per waiting request
                    log.error("cannot fetch signing keys from FRA_OAUTH_JWKS_URL: %s", reason)
                raise KeysUnavailable(reason)
            # A failed fetch falls through: the keys fetched earlier may still be usable.
        with self.lock:
            usable = time.monotonic() - self.fetched_at < JWKS_STALE_SECONDS
            key = self.keys.get(kid) if usable else None
        if key is not None:
            return key
        if usable:  # the provider answered recently and does not publish this kid
            raise UnknownSigningKey("no signing key matches the token's kid")
        raise KeysUnavailable(self.last_error)

    def _start_fetch(self) -> Future | None:  # with self.lock held
        if self.fetch is not None and not self.fetch.done():
            return self.fetch  # share the fetch in progress
        if time.monotonic() - self.attempted_at < JWKS_RETRY_SECONDS:
            return None
        self.attempted_at = time.monotonic()
        self.fetch = _jwks_fetcher.submit(self._refresh)
        return self.fetch

    def _refresh(self) -> None:
        try:
            keys = fetch_signing_keys(self.url)
        except Exception as exc:  # KeysUnavailable, or anything unexpected: keep the last good keys
            reason = str(exc) if isinstance(exc, KeysUnavailable) else type(exc).__name__
            with self.lock:
                self.last_error, age = reason, time.monotonic() - self.fetched_at
            still = f"; still using keys fetched {age:.0f} s ago" if age < JWKS_STALE_SECONDS else ""
            log.error("cannot fetch signing keys from FRA_OAUTH_JWKS_URL: %s%s", reason, still)
            raise KeysUnavailable(reason) from None
        with self.lock:
            self.keys, self.fetched_at = keys, time.monotonic()


@functools.lru_cache(maxsize=4)
def provider_keys(url: str) -> ProviderKeys:
    return ProviderKeys(url)


async def decode_access_token(token: str, settings: dict) -> dict:
    """Verify signature, expiry, issuer and audience; return the claims.
    Raises jwt.PyJWTError (or ValueError for a malformed token) for a bad token, KeysUnavailable otherwise."""
    try:
        header = jwt.get_unverified_header(token)
    except (ValueError, TypeError, RecursionError):  # e.g. a deeply nested header: still just a bad token
        raise jwt.DecodeError("malformed token header") from None
    if header.get("alg") not in OAUTH_ALGORITHMS:  # checked before any key fetch
        raise jwt.InvalidAlgorithmError("algorithm not allowed")
    jwk = await provider_keys(settings["jwks_url"]).get(header.get("kid"))
    return jwt.decode(
        token,
        jwt.PyJWK(jwk, jwk.get("alg") or header["alg"]),  # "alg" is optional in a JWK (RFC 7517 s4.4)
        algorithms=OAUTH_ALGORITHMS,
        audience=settings["audience"],
        issuer=settings["issuer"],
        leeway=LEEWAY_SECONDS,
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


def _reject(status: int, detail: str, error: str | None = "invalid_token") -> HTTPException:
    # RFC 6750 s3.1: no error code when the request carried no credentials at all.
    return HTTPException(status_code=status, detail=detail,
                         headers={"WWW-Authenticate": f'Bearer error="{error}"' if error else "Bearer"})


def _unavailable(detail: str) -> HTTPException:
    """503: the call may succeed if Pega retries (the provider's keys could not be fetched)."""
    return HTTPException(status_code=503, detail=detail, headers={"Retry-After": str(JWKS_RETRY_SECONDS)})


bearer = HTTPBearer(auto_error=False)


async def authenticate(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> float:
    """Reject unauthenticated calls outright (spec s4). Returns the arrival time, so time spent here
    (a key fetch, a wait for a worker thread) counts against Pega's 15 s."""
    arrived = time.monotonic()
    problem = auth_problem()
    if problem:  # the startup check normally stops this earlier
        log.error("cannot authenticate: %s", problem)
        raise _unavailable("Authentication is not configured.")
    settings = oauth_settings()
    if creds is None:
        raise _reject(401, "Missing or invalid bearer token.", error=None)
    if settings is None:
        # Compare bytes: compare_digest raises on non-ASCII str, which would turn a bad token into a 500.
        if not secrets.compare_digest(creds.credentials.encode(), _env("FRA_API_TOKEN").encode()):
            raise _reject(401, "Missing or invalid bearer token.")
        return arrived
    try:  # on the event loop: a signature check takes well under 1 ms, and waiting for keys costs no thread
        claims = await decode_access_token(creds.credentials, settings)
    except KeysUnavailable:  # the fetch logs why
        raise _unavailable("Cannot verify tokens right now.") from None
    except (jwt.PyJWTError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        log.info("token rejected: %s", type(exc).__name__)  # the reason, never the token or the caller's text
        raise _reject(401, "Missing or invalid bearer token.") from None
    except Exception as exc:  # anything unexpected while checking: fail closed
        log.error("token check failed: %s", type(exc).__name__)
        raise _unavailable("Cannot verify tokens right now.") from None
    if settings["scope"] not in granted_scopes(claims):
        log.info("token rejected: missing scope %s", settings["scope"])
        raise _reject(403, "The token does not grant the required scope.", "insufficient_scope")
    return arrived


# ---------------------------------------------------------------- app
app = FastAPI(title="Fraud Risk Agent", version=AGENT_VERSION, lifespan=lifespan)
PUBLIC_URL = os.getenv("FRA_PUBLIC_URL", "http://localhost:8000")


# ---------------------------------------------------------------- A2A message <-> claim (spec s3, open item O-1)
# Pega AI agents call external agents through their language model, which writes the claim as text: JSON, or
# one "name: value" line per field with flat names (loss_cause, late_reported: false) and dates like 30-06-2024.
# Other callers send a data part. All of these work, and so does the older A2A "type" key. Reading them is
# plain code, never a language model, and every value is still validated by fra_agent.validate_request.
EXAMPLE_CLAIM = {
    "claim_id": "AC-1002",
    "correlation_id": "5b0e9a7c-2f3d-4e8a-9c1b-6d7e8f9a0b1c",
    "policy_ref_hash": "dcec50fe0267059456d6f6872195e28bc3b2cdfd425dccd8a115ee7d2255f220",
    "vin_hash": "1473895b7794b967de1cd38c9f5636569a118f099b0f630d7d3dc315e924e2fd",
    "vehicle_policy_mismatch": False,
    "loss": {"cause": "COLLISION", "date": "2026-09-20", "reported_on": "2026-09-21",
             "description": "Rear-ended at a signal while waiting for the light to change."},
    "claim_history": {"claims_last_24_months": 0, "days_since_policy_start": 262, "late_reported": False,
                      "potential_duplicate": True},
    "policy_data_available": True,
    "history_data_available": True,
}
HOW_TO_SEND = (
    "Send the claim as one JSON object (in a data part or as text), or as one 'name: value' line per field, "
    "with these fields: claim_id, correlation_id (a UUID v4), policy_ref_hash, vin_hash, "
    "vehicle_policy_mismatch (true/false), loss {cause, date, reported_on, description}, claim_history "
    "{claims_last_24_months, days_since_policy_start, late_reported, potential_duplicate}, "
    "policy_data_available, history_data_available. Write dates as YYYY-MM-DD. "
    "The Agent Card's skill has a complete example."
)
TEXT_SCAN_LIMIT = 65_536  # characters of a text part searched for the claim


def _json_object_in(text: str):
    """The first JSON object in a text: the whole text, or one inside prose or a ``` block."""
    decoder, text = json.JSONDecoder(), text[:TEXT_SCAN_LIMIT]
    start = text.find("{")
    for _ in range(20):  # bounded, so a text full of stray braces cannot keep a worker busy
        if start < 0:
            break
        try:
            value, _ = decoder.raw_decode(text, start)
            if isinstance(value, dict):
                return value
        except (ValueError, RecursionError):
            pass
        start = text.find("{", start + 1)
    return None


# Flat field names, as in the data dictionary's CSV columns ("loss_cause"), mapped to request paths.
FLAT_NAMES = {path.split(".")[-1]: path for path in REQUIRED_FIELDS} | {
    path.replace(".", "_"): path for path in REQUIRED_FIELDS if path.startswith("loss.")}
FLAT_NAMES.pop("date")  # "date" alone is too vague; loss_date / loss.date are accepted
FIELD_LINE = re.compile(r"\s*(?:[-*\u2022]+|\d+[.)])?\s*[*_`]*([A-Za-z][\w .-]*?)[*_`]*\s*[:=]\s*(.*?)\s*,?\s*")
NUMERIC_DATE = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})")
YEAR_FIRST_DATE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")


def _typed(value, kind):
    """A text value as the type the request needs ("false" -> False, "3" -> 3). Anything else is left
    as it is, so validation names the field."""
    if isinstance(value, str):
        value = value.strip().strip("\"'`").strip()
        if kind is bool and value.lower() in ("true", "yes", "1", "false", "no", "0"):
            return value.lower() in ("true", "yes", "1")
        if kind is int and re.fullmatch(r"-?\d+", value):
            return int(value)
    elif kind is bool and type(value) is int and value in (0, 1):  # the CSV's 0/1
        return bool(value)
    return value


def _iso_dates(loss) -> None:
    """Dates written as 30-06-2024 or 2024/06/30 become 2024-06-30. Day-first unless a date in the same
    claim proves month-first (a middle number above 12). YYYY-MM-DD needs no guessing."""
    if not isinstance(loss, dict):
        return
    dated = {k: v.strip() for k in ("date", "reported_on") if isinstance(v := loss.get(k), str)}
    numeric = {k: m for k, v in dated.items() if (m := NUMERIC_DATE.fullmatch(v))}
    month_first = any(int(m.group(2)) > 12 for m in numeric.values())
    for key, text in dated.items():
        if m := YEAR_FIRST_DATE.fullmatch(text):
            year, month, day = (int(x) for x in m.groups())
        elif m := numeric.get(key):
            first, second, year = (int(x) for x in m.groups())
            day, month = (second, first) if month_first else (first, second)
        else:
            continue
        try:
            loss[key] = datetime.date(year, month, day).isoformat()
        except ValueError:
            pass  # not a real date: left as it is, so validation names the field


def _claim_from_fields(fields: dict):
    """A request built from flat name/value pairs, or None if there is no claim_id among them."""
    claim: dict = {}
    for name, value in fields.items():
        key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_").removeprefix("claim_history_")
        path = FLAT_NAMES.get(key)
        if path is None:
            continue  # e.g. "Case ID": not part of the request
        *parents, leaf = path.split(".")
        node = claim
        for parent in parents:
            node = node.setdefault(parent, {})
        node.setdefault(leaf, _typed(value, REQUIRED_FIELDS[path]))
    if "claim_id" not in claim:
        return None
    _iso_dates(claim.get("loss"))
    return claim


def _fields_in_text(text: str) -> dict:
    """{name: value} from lines such as "- loss_cause: THEFT" or "claim_id = AC-3001"."""
    fields = {}
    for line in text[:TEXT_SCAN_LIMIT].splitlines():
        if m := FIELD_LINE.fullmatch(line):
            fields.setdefault(m.group(1), m.group(2))
    return fields


def _as_claim(value, from_text: bool):
    """A found JSON value as a request: unwrapped, and converted when it uses flat field names."""
    value = _unwrap(value)
    if isinstance(value, dict) and "loss" not in value and "claim_history" not in value:
        flat = _claim_from_fields(value)
        if flat is not None:
            return flat
    if from_text and isinstance(value, dict):  # written by a language model: forgive its date format
        _iso_dates(value.get("loss"))
    return value


def _unwrap(value):
    """{"claim": {...}} -> {...}: callers sometimes wrap the claim in one outer object."""
    if isinstance(value, dict) and "claim_id" not in value:
        inner = [v for v in value.values() if isinstance(v, dict) and "claim_id" in v]
        if len(inner) == 1:
            return inner[0]
    return value


def claim_from_message(message: dict) -> tuple[object, list[str]]:
    """The claim in an A2A message (None if there is none) and the kinds of its parts, for the log.
    A data part wins over text parts. Raises KeyError/TypeError/AttributeError for a malformed message."""
    parts = message["parts"]
    if not isinstance(parts, list) or not parts:
        raise TypeError("the message has no parts")
    kinds = [part.get("kind") or part.get("type") for part in parts]  # "type": A2A before version 0.3
    for part, kind in zip(parts, kinds):
        if kind == "data":
            data = part.get("data")
            if isinstance(data, str):
                return _as_claim(_json_object_in(data), from_text=True), kinds
            return _as_claim(data, from_text=False), kinds
    texts = [part["text"] for part, kind in zip(parts, kinds) if kind == "text" and isinstance(part.get("text"), str)]
    for text in texts:  # JSON first, then "name: value" lines
        found = _json_object_in(text)
        if found is not None:
            return _as_claim(found, from_text=True), kinds
    for text in texts:
        found = _claim_from_fields(_fields_in_text(text))
        if found is not None:
            return found, kinds
    return None, kinds


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
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        **security_schemes(),
        "skills": [
            {
                "id": "assess_fraud_risk",
                "name": "Assess fraud risk",
                "description": "Scores five history-based fraud indicators for one motor claim and returns "
                "status, risk_score, confidence, risk_flags and a neutral explanation (reasoning). " + HOW_TO_SEND,
                "tags": ["fraud", "claims", "insurance"],
                "examples": [json.dumps(EXAMPLE_CLAIM)],
                "inputModes": ["application/json", "text/plain"],
                "outputModes": ["application/json", "text/plain"],
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
        claim, kinds = claim_from_message(message)
    except (KeyError, TypeError, AttributeError):
        return rpc_error(rpc.get("id"), -32602, "Invalid params: expected params.message with a list of parts")

    if claim is None:  # e.g. a question in plain words: say what to send, so a calling agent can retry
        seen = ", ".join(k if k in ("text", "data", "file") else "other" for k in kinds)  # never the content
        log.warning("a2a: no claim found in the message (parts: %s)", seen)
        result = failed("No claim was found in the message. " + HOW_TO_SEND)
    else:
        result = await run_in_threadpool(_assess, claim, arrived)
    message_id = message.get("messageId")
    reply = {
        "kind": "message",
        "role": "agent",
        # Derived from the request's messageId, so a repeat call gets an identical reply.
        "messageId": str(uuid.uuid5(uuid.NAMESPACE_URL, message_id if isinstance(message_id, str) else "")),
        # The same result twice: as data for programs, as JSON text for a calling agent's language model.
        "parts": [{"kind": "data", "data": result}, {"kind": "text", "text": json.dumps(result, ensure_ascii=False)}],
    }
    if isinstance(message.get("contextId"), str):
        reply["contextId"] = message["contextId"]  # keeps the reply in the caller's conversation
    return {"jsonrpc": "2.0", "id": rpc.get("id"), "result": reply}


@app.post("/v1/assess")
async def assess(request: Request, arrived: float = Depends(authenticate)) -> dict:
    try:
        payload = await request.json()
    except (ValueError, RecursionError):  # invalid or too deeply nested JSON
        payload = None  # the agent answers FAILED: "The request body must be a JSON object."
    return await run_in_threadpool(_assess, payload, arrived)
