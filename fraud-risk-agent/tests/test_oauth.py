"""OAuth 2.0 tests (spec s4): JWT access tokens checked against the provider's JWKS, plus the startup check.

The "provider" is an RSA key made here. Its public half is served by a real HTTP endpoint on 127.0.0.1,
so the agent's own download, cache and error handling run as in production. CI also runs the full flow
against Keycloak (oauth job).
"""
import base64
import hashlib
import hmac
import http.server
import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from conftest import build_request
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

import server
from fra_agent import FraudRiskAgent

ISSUER = "https://idp.example/realms/claims"
AUDIENCE = "fraud-risk-agent"
PROVIDER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwk(private_key, kid: str, **extra) -> dict:
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    return {**public, "kid": kid, "use": "sig", "alg": "RS256", **extra}


def make_token(key=PROVIDER_KEY, kid="k1", alg="RS256", **claims) -> str:
    now = int(time.time())
    body = {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 300, "scope": "profile fraud.assess",
            "azp": "pega-claim-assist", **claims}
    return jwt.encode({k: v for k, v in body.items() if v is not None}, key, algorithm=alg, headers={"kid": kid})


class Provider:
    """The provider's JWKS endpoint: a real HTTP server on 127.0.0.1 that counts its fetches.
    Tests change what it answers: other keys, an HTTP status, a raw body, or a slow trickle."""

    def __init__(self):
        self.keys, self.fetches = [jwk(PROVIDER_KEY, "k1")], 0
        self.status, self.body, self.headers, self.trickle = 200, None, {}, 0.0
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                provider.fetches += 1
                body = provider.body if provider.body is not None else json.dumps({"keys": provider.keys}).encode()
                self.send_response(provider.status)
                for name, value in provider.headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if provider.trickle:  # a byte every `trickle` seconds for 2 s: no single read times out
                    for i in range(int(2 / provider.trickle)):
                        self.wfile.write(body[i:i + 1])
                        self.wfile.flush()
                        time.sleep(provider.trickle)
                    body = body[int(2 / provider.trickle):]
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.handle_error = lambda *args: None  # a client that gave up mid-trickle is expected here
        self.url = f"http://127.0.0.1:{self.http.server_port}/realms/claims/protocol/openid-connect/certs"
        threading.Thread(target=self.http.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def stop(self):
        self.http.shutdown()
        self.http.server_close()


def wait_for_fetches():
    """Let any fetch still running finish, so it cannot hold up the next test (one fetch at a time)."""
    server._jwks_fetcher.submit(lambda: None).result(timeout=30)


@pytest.fixture
def provider(monkeypatch):
    stub = Provider()
    monkeypatch.setenv("FRA_OAUTH_ISSUER", ISSUER)
    monkeypatch.setenv("FRA_OAUTH_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("FRA_OAUTH_JWKS_URL", stub.url)
    monkeypatch.setenv("FRA_OAUTH_ALLOW_HTTP", "1")  # the test endpoint is plain http on loopback
    monkeypatch.setenv("FRA_API_TOKEN", "test-token")  # must be ignored once OAuth is configured
    for name in ("NO_PROXY", "no_proxy"):  # never send the loopback fetch through a proxy
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    monkeypatch.setattr(server, "agent", FraudRiskAgent())
    server.provider_keys.cache_clear()
    yield stub
    wait_for_fetches()
    stub.stop()
    server.provider_keys.cache_clear()


@pytest.fixture
def client(provider):
    return TestClient(server.app)


def keys_of(provider) -> server.ProviderKeys:
    return server.provider_keys(provider.url)


def allow_next_fetch(provider):
    """As if JWKS_RETRY_SECONDS had passed since the last fetch."""
    keys_of(provider).attempted_at -= server.JWKS_RETRY_SECONDS


def call(client, token, path="/v1/assess"):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return client.post(path, json=build_request(claim_history__potential_duplicate=True), headers=headers)


def test_valid_token_is_accepted_on_both_endpoints(client):
    reply = call(client, make_token())
    assert (reply.status_code, reply.json()["risk_score"]) == (200, 70)
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "message/send",
           "params": {"message": {"messageId": "m", "parts": [{"kind": "data", "data": build_request()}]}}}
    reply = client.post("/a2a", json=rpc, headers={"Authorization": f"Bearer {make_token()}"})
    assert reply.json()["result"]["parts"][0]["data"]["status"] == "COMPLETED"


def b64(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def unsigned_token() -> str:
    claims = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300, "scope": "fraud.assess"}
    return jwt.encode(claims, None, algorithm="none", headers={"kid": "k1"})


def hs256_with_public_key() -> str:
    """Algorithm confusion: HMAC 'signed' with the provider's public key, which anyone can download."""
    pem = PROVIDER_KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    claims = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300, "scope": "fraud.assess"}
    signing_input = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "k1"}).encode()) + b"." + b64(json.dumps(claims).encode())
    return (signing_input + b"." + b64(hmac.new(pem, signing_input, hashlib.sha256).digest())).decode()


def deeply_layered_header() -> str:
    """About 2.7 KB: older PyJWT raised RecursionError here, which must still be a plain 401."""
    header = b'{"alg":"RS256","kid":"k1","x":' + b"[" * 1000 + b"]" * 1000 + b"}"
    return (b64(header) + b"." + b64(b'{"iss":"x"}') + b".c2ln").decode()


REJECTED = {
    "no token": lambda: None,
    "static token": lambda: "test-token",
    "not a JWT": lambda: "abc.def",
    "wrong audience": lambda: make_token(aud="some-other-api"),
    "wrong issuer": lambda: make_token(iss="https://evil.example/realms/claims"),
    "expired": lambda: make_token(exp=int(time.time()) - 120),
    "expired beyond the 30 s leeway": lambda: make_token(exp=int(time.time()) - 35),
    "not yet valid": lambda: make_token(nbf=int(time.time()) + 120),
    "issued in the future": lambda: make_token(iat=int(time.time()) + 120),
    "no expiry": lambda: make_token(exp=None),
    "no audience": lambda: make_token(aud=None),
    "signed by another key": lambda: make_token(key=OTHER_KEY),
    "key id the provider never issued": lambda: make_token(key=OTHER_KEY, kid="k9"),
    "alg none": unsigned_token,
    "HS256 with the public key": hs256_with_public_key,
    "header 1000 levels deep": deeply_layered_header,
    "non-ASCII": lambda: "töken",
}


@pytest.mark.parametrize("path", ["/a2a", "/v1/assess"])
@pytest.mark.parametrize("case", REJECTED)
def test_bad_tokens_are_rejected_with_401(client, case, path):
    token = REJECTED[case]()
    headers = {"Authorization": f"Bearer {token}".encode("latin-1")} if token else {}
    reply = client.post(path, json=build_request(), headers=headers)
    assert reply.status_code == 401
    # RFC 6750 s3.1: no error code when the request carried no credentials.
    assert reply.headers["WWW-Authenticate"] == ('Bearer error="invalid_token"' if token else "Bearer")


@pytest.mark.parametrize("claims", [
    {"scope": "openid profile"},
    {"scope": "fraud.assess.all"},  # whole words only
    {"scope": None},
    {"scope": None, "scp": "fraud"},
    {"scope": None, "roles": ["fraud.read"]},
    {"scope": None, "scp": [{"fraud.assess": True}]},  # odd shapes are ignored, not crashed on
])
def test_token_without_the_scope_gets_403(client, claims):
    reply = call(client, make_token(**claims))
    assert reply.status_code == 403
    assert reply.headers["WWW-Authenticate"] == 'Bearer error="insufficient_scope"'


@pytest.mark.parametrize("claims", [
    {"scope": "fraud.assess"},                          # Keycloak
    {"scope": None, "scp": ["fraud.assess", "x"]},       # Okta
    {"scope": None, "roles": ["fraud.assess"]},          # Entra ID application permission (app role)
    {"aud": ["account", AUDIENCE]},                      # several audiences, ours among them
])
def test_provider_specific_token_shapes_are_accepted(client, claims):
    assert call(client, make_token(**claims)).status_code == 200


def test_clock_skew_allowance_is_30_seconds(client):
    now = int(time.time())
    assert call(client, make_token(exp=now - 25)).status_code == 200
    assert call(client, make_token(iat=now + 25, nbf=now + 25)).status_code == 200
    assert call(client, make_token(exp=now - 35)).status_code == 401
    assert call(client, make_token(iat=now + 35)).status_code == 401


def test_ec_keys_work_too(client, provider):
    key = ec.generate_private_key(ec.SECP256R1())
    provider.keys = [{**json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key())), "kid": "ec1", "use": "sig"}]
    assert call(client, make_token(key=key, kid="ec1", alg="ES256")).status_code == 200


def test_keys_published_without_alg_take_the_token_algorithm(client, provider):
    provider.keys = [{k: v for k, v in jwk(PROVIDER_KEY, "k1").items() if k != "alg"}]  # "alg" is optional
    assert call(client, make_token(alg="PS256")).status_code == 200
    # ...but only an algorithm that fits the key type: ES256 on this RSA key is refused.
    claims = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300, "scope": "fraud.assess"}
    es256 = b64(b'{"alg":"ES256","kid":"k1"}') + b"." + b64(json.dumps(claims).encode()) + b"." + b64(b"x" * 64)
    assert call(client, es256.decode()).status_code == 401


def test_custom_scope_name(client, monkeypatch):
    monkeypatch.setenv("FRA_OAUTH_SCOPE", "claims.fraud")
    assert call(client, make_token()).status_code == 403
    assert call(client, make_token(scope="claims.fraud")).status_code == 200


# ---------------------------------------------------------------- the provider's keys

def test_keys_are_cached_and_rotation_is_picked_up(client, provider):
    for _ in range(3):
        assert call(client, make_token()).status_code == 200
    assert provider.fetches == 1  # cached

    provider.keys = [jwk(PROVIDER_KEY, "k1"), jwk(OTHER_KEY, "k2")]  # the provider rotates in a new key
    allow_next_fetch(provider)
    assert call(client, make_token(key=OTHER_KEY, kid="k2")).status_code == 200
    assert provider.fetches == 2


def test_forged_key_ids_cannot_flood_the_provider(client, provider, caplog):
    with caplog.at_level(logging.INFO, logger="fra.server"):
        for i in range(20):
            assert call(client, make_token(key=OTHER_KEY, kid=f"forged-{i}")).status_code == 401
    assert provider.fetches == 1  # unknown key ids refetch at most once per JWKS_RETRY_SECONDS
    assert "token rejected: UnknownSigningKey" in caplog.text

    allow_next_fetch(provider)  # the window reopens
    assert call(client, make_token(key=OTHER_KEY, kid="forged-x")).status_code == 401
    assert provider.fetches == 2


def test_old_keys_are_refreshed_in_the_background(client, provider):
    assert call(client, make_token()).status_code == 200
    keys = keys_of(provider)
    keys.fetched_at -= server.JWKS_FRESH_SECONDS - 1
    allow_next_fetch(provider)
    assert call(client, make_token()).status_code == 200
    assert provider.fetches == 1  # still fresh

    keys.fetched_at -= 2  # now older than JWKS_FRESH_SECONDS
    assert call(client, make_token()).status_code == 200  # answered with the cached key at once...
    wait_for_fetches()
    assert provider.fetches == 2  # ...while a background fetch refreshed the keys
    assert time.monotonic() - keys.fetched_at < 5


def test_provider_outage_keeps_the_last_keys_for_an_hour(client, provider, caplog):
    assert call(client, make_token()).status_code == 200
    keys = keys_of(provider)
    provider.status = 500
    keys.fetched_at -= server.JWKS_FRESH_SECONDS + 1
    allow_next_fetch(provider)
    with caplog.at_level(logging.ERROR, logger="fra.server"):
        for _ in range(5):
            assert call(client, make_token()).status_code == 200  # valid tokens keep working
        wait_for_fetches()
    assert provider.fetches == 2  # one attempt, not one per request
    assert "cannot fetch signing keys from FRA_OAUTH_JWKS_URL: HTTP 500; still using keys fetched" in caplog.text

    keys.fetched_at -= server.JWKS_STALE_SECONDS  # an outage longer than an hour
    allow_next_fetch(provider)
    for _ in range(5):
        reply = call(client, make_token())
        assert (reply.status_code, reply.headers["Retry-After"]) == (503, str(server.JWKS_RETRY_SECONDS))
    assert provider.fetches == 3


def unused_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/certs"


@pytest.mark.parametrize("problem, reason", [
    ({"status": 404}, "HTTP 404"),
    ({"status": 302, "headers": {"Location": "https://elsewhere.example/certs"}}, "HTTP 302"),  # never followed
    ({"body": b"<html>Sign in</html>"}, "the response is not a JSON Web Key Set"),
    ({"keys": [jwk(PROVIDER_KEY, "k1", use="enc")]}, "the key set has no usable signing keys"),
    ({"keys": [{"kty": "RSA", "kid": "k1", "n": "!!", "e": "AQAB"}]}, "the key set has no usable signing keys"),
    ({"url": "unused port"}, "Connection refused"),
])
def test_key_endpoint_problems_fail_closed_with_503_and_a_logged_reason(client, provider, monkeypatch, caplog,
                                                                        problem, reason):
    for name, value in problem.items():
        if name == "url":
            monkeypatch.setenv("FRA_OAUTH_JWKS_URL", unused_port_url())
        else:
            setattr(provider, name, value)
    token = make_token()
    with caplog.at_level(logging.INFO, logger="fra.server"):
        reply = call(client, token)
    assert reply.status_code == 503 and reply.headers["Retry-After"] == str(server.JWKS_RETRY_SECONDS)
    assert "WWW-Authenticate" not in reply.headers  # not the caller's fault
    assert "cannot fetch signing keys from FRA_OAUTH_JWKS_URL: " in caplog.text and reason in caplog.text
    assert token not in caplog.text

    # A bad answer is never cached: once the provider is fixed, the next allowed fetch works.
    monkeypatch.setenv("FRA_OAUTH_JWKS_URL", provider.url)
    provider.status, provider.body, provider.headers, provider.keys = 200, None, {}, [jwk(PROVIDER_KEY, "k1")]
    allow_next_fetch(provider)
    assert call(client, token).status_code == 200


def test_a_stalled_key_endpoint_holds_up_no_request_for_long(client, provider, monkeypatch, caplog):
    monkeypatch.setattr(server, "JWKS_TIMEOUT", 0.5)  # also each socket read's timeout
    provider.trickle = 0.2  # a byte every 0.2 s: no single read times out, the whole download takes 2 s
    with caplog.at_level(logging.ERROR, logger="fra.server"):
        started = time.monotonic()
        first, second = call(client, make_token()), call(client, make_token())
        elapsed = time.monotonic() - started
    assert (first.status_code, second.status_code) == (503, 503)
    assert elapsed < 1.8  # each waited 0.5 s, not for the whole download
    assert provider.fetches == 1  # the second request shared the fetch in progress
    assert caplog.text.count("did not answer within 0.5 s") == 1  # logged once per stalled fetch

    wait_for_fetches()  # the slow download completes in its own thread...
    assert call(client, make_token()).status_code == 200  # ...and its keys are used, with no new fetch
    assert provider.fetches == 1


def test_requests_waiting_for_a_slow_fetch_hold_no_worker_threads(provider, monkeypatch):
    monkeypatch.setattr(server, "JWKS_TIMEOUT", 1.0)
    with TestClient(server.app) as shared:  # one event loop, so one pool of worker threads for all calls
        assert call(shared, make_token()).status_code == 200  # Pega's key is cached
        provider.trickle = 0.2  # the next fetch takes 2 s
        allow_next_fetch(provider)
        forged = make_token(key=OTHER_KEY, kid="forged")
        with ThreadPoolExecutor(60) as pool:  # more forged calls than the 40 worker threads
            waiting = [pool.submit(call, shared, forged) for _ in range(60)]
            time.sleep(0.3)  # all of them are now waiting for the slow fetch
            started = time.monotonic()
            assert call(shared, make_token()).status_code == 200
            assert time.monotonic() - started < 0.5  # Pega's call did not queue behind them
            assert {f.result().status_code for f in waiting} <= {401, 503}


def test_a_download_that_never_finishes_is_abandoned(client, provider, monkeypatch, caplog):
    monkeypatch.setattr(server, "JWKS_FETCH_SECONDS", 1)
    provider.trickle = 0.2  # bytes keep coming, so no single read times out, but it takes 2 s
    with caplog.at_level(logging.ERROR, logger="fra.server"):
        assert call(client, make_token()).status_code == 503
        wait_for_fetches()
    assert "the key endpoint took longer than 1 s" in caplog.text
    provider.trickle = 0  # the next allowed fetch starts afresh
    allow_next_fetch(provider)
    assert call(client, make_token()).status_code == 200
    assert provider.fetches == 2


def test_unpublished_key_id_is_401_even_when_its_refetch_fails(client, provider):
    assert call(client, make_token()).status_code == 200
    provider.status = 500  # the refetch the unknown kid triggers fails...
    allow_next_fetch(provider)
    assert call(client, make_token(key=OTHER_KEY, kid="k9")).status_code == 401  # ...but the keys are still usable
    assert call(client, make_token()).status_code == 200


def test_tokens_never_reach_the_log(client, caplog):
    tokens = [make_token(), make_token(aud="other"), make_token(scope="openid"), "abc.def.ghi"]
    with caplog.at_level(logging.DEBUG):
        for token in tokens:
            call(client, token)
    assert caplog.text and not any(t in caplog.text for t in tokens)
    assert "InvalidAudienceError" in caplog.text  # but the reason is


def test_agent_card_advertises_oauth(client, monkeypatch):
    card = client.get("/.well-known/agent.json").json()
    assert card["security"] == [{"oauth2": ["fraud.assess"]}]
    assert card["securitySchemes"]["oauth2"] == {
        "type": "openIdConnect", "openIdConnectUrl": ISSUER + "/.well-known/openid-configuration"}

    monkeypatch.setenv("FRA_OAUTH_TOKEN_URL", ISSUER + "/protocol/openid-connect/token")
    card = client.get("/.well-known/agent.json").json()
    assert card["securitySchemes"]["oauth2"] == {"type": "oauth2", "flows": {"clientCredentials": {
        "tokenUrl": ISSUER + "/protocol/openid-connect/token", "scopes": {"fraud.assess": "Assess the fraud risk of a claim"}}}}


def test_stray_whitespace_in_settings_is_ignored(provider, monkeypatch):
    monkeypatch.setenv("FRA_OAUTH_ISSUER", ISSUER + "\n")  # e.g. a Kubernetes secret made from a file
    monkeypatch.setenv("FRA_OAUTH_AUDIENCE", " " + AUDIENCE + " ")
    with TestClient(server.app) as started:
        assert call(started, make_token()).status_code == 200


# ---------------------------------------------------------------- startup check

OAUTH_ENV = {"FRA_OAUTH_ISSUER": ISSUER, "FRA_OAUTH_AUDIENCE": AUDIENCE, "FRA_OAUTH_JWKS_URL": "https://idp.example/certs"}


def start(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with TestClient(server.app) as started:
        return started.get("/health").status_code


def test_server_refuses_to_start_without_any_authentication(monkeypatch):
    with pytest.raises(RuntimeError, match="no authentication configured"):
        start(monkeypatch)


def test_server_refuses_to_start_with_half_configured_oauth(monkeypatch):
    with pytest.raises(RuntimeError, match="half configured: also set FRA_OAUTH_AUDIENCE, FRA_OAUTH_JWKS_URL"):
        start(monkeypatch, FRA_OAUTH_ISSUER=ISSUER, FRA_OAUTH_AUDIENCE="  ", FRA_API_TOKEN="test-token")


def test_blank_settings_count_as_unset(monkeypatch):
    with pytest.raises(RuntimeError, match="no authentication configured"):
        start(monkeypatch, FRA_OAUTH_ISSUER="   ", FRA_API_TOKEN=" \n")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://idp.example/certs", "https:///certs"])
def test_server_refuses_a_key_url_that_is_not_https(monkeypatch, url):
    with pytest.raises(RuntimeError, match="must be an https URL"):
        start(monkeypatch, **{**OAUTH_ENV, "FRA_OAUTH_JWKS_URL": url})


def test_plain_http_key_url_needs_an_explicit_opt_in(monkeypatch, caplog):
    env = {**OAUTH_ENV, "FRA_OAUTH_JWKS_URL": "http://keycloak:8080/realms/claims/protocol/openid-connect/certs"}
    with pytest.raises(RuntimeError, match="uses plain http"):
        start(monkeypatch, **env)
    with caplog.at_level(logging.WARNING, logger="fra.server"):
        assert start(monkeypatch, **env, FRA_OAUTH_ALLOW_HTTP="1") == 200
    assert "for local testing only" in caplog.text


@pytest.mark.parametrize("env", [{"FRA_API_TOKEN": "test-token"}, OAUTH_ENV])
def test_server_starts_with_either_kind_of_authentication(monkeypatch, env, caplog):
    with caplog.at_level(logging.INFO, logger="fra.server"):
        assert start(monkeypatch, **env) == 200
    static = [r for r in caplog.records if "static token" in r.getMessage()]
    # Static-token mode is announced as a warning, so it is seen even at FRA_LOG_LEVEL=WARNING.
    assert [r.levelname for r in static] == (["WARNING"] if "FRA_API_TOKEN" in env else [])


def test_settings_removed_after_startup_fail_closed(monkeypatch):
    monkeypatch.setenv("FRA_OAUTH_ISSUER", ISSUER)  # half configured at request time
    reply = TestClient(server.app).post("/v1/assess", json=build_request(), headers={"Authorization": "Bearer x"})
    assert reply.status_code == 503
