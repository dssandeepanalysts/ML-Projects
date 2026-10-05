"""OAuth 2.0 tests (spec s4): JWT access tokens checked against the provider's JWKS, plus the startup check.

The "provider" is an RSA key made here; its public half is served by a stubbed JWKS fetch, so no
network or identity provider is needed. CI also runs the real flow against Keycloak (oauth job).
"""
import base64
import hashlib
import hmac
import json
import logging
import time

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


def jwk(private_key, kid: str) -> dict:
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    return {**public, "kid": kid, "use": "sig", "alg": "RS256"}


def make_token(key=PROVIDER_KEY, kid="k1", alg="RS256", **claims) -> str:
    now = int(time.time())
    body = {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 300, "scope": "profile fraud.assess",
            "azp": "pega-claim-assist", **claims}
    return jwt.encode({k: v for k, v in body.items() if v is not None}, key, algorithm=alg, headers={"kid": kid})


class Provider:
    """Stands in for the provider's JWKS endpoint and counts how often it is fetched."""

    def __init__(self):
        self.keys, self.fetches, self.error = [jwk(PROVIDER_KEY, "k1")], 0, None

    def fetch_data(self, client):
        self.fetches += 1
        if self.error:
            raise self.error
        data = {"keys": list(self.keys)}
        client.jwk_set_cache.put(data)
        return data


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("FRA_OAUTH_ISSUER", ISSUER)
    monkeypatch.setenv("FRA_OAUTH_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("FRA_OAUTH_JWKS_URL", "https://idp.example/realms/claims/protocol/openid-connect/certs")
    monkeypatch.setenv("FRA_API_TOKEN", "test-token")  # must be ignored once OAuth is configured
    monkeypatch.setattr(server, "agent", FraudRiskAgent())
    stub = Provider()
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", lambda client: stub.fetch_data(client))
    server._jwks_client.cache_clear()
    server._last_refresh.clear()
    yield stub
    server._jwks_client.cache_clear()
    server._last_refresh.clear()


@pytest.fixture
def client(provider):
    return TestClient(server.app)


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


def unsigned_token() -> str:
    claims = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300, "scope": "fraud.assess"}
    return jwt.encode(claims, None, algorithm="none")


def hs256_with_public_key() -> str:
    """Algorithm confusion: HMAC 'signed' with the provider's public key, which anyone can download."""
    pem = PROVIDER_KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    claims = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300, "scope": "fraud.assess"}

    def b64(raw: bytes) -> bytes:
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    signing_input = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "k1"}).encode()) + b"." + b64(json.dumps(claims).encode())
    return (signing_input + b"." + b64(hmac.new(pem, signing_input, hashlib.sha256).digest())).decode()


REJECTED = {
    "no token": lambda: None,
    "static token": lambda: "test-token",
    "not a JWT": lambda: "abc.def",
    "wrong audience": lambda: make_token(aud="some-other-api"),
    "wrong issuer": lambda: make_token(iss="https://evil.example/realms/claims"),
    "expired": lambda: make_token(exp=int(time.time()) - 120),
    "not yet valid": lambda: make_token(nbf=int(time.time()) + 120),
    "no expiry": lambda: make_token(exp=None),
    "no audience": lambda: make_token(aud=None),
    "signed by another key": lambda: make_token(key=OTHER_KEY),
    "key id the provider never issued": lambda: make_token(key=OTHER_KEY, kid="k9"),
    "alg none": unsigned_token,
    "HS256 with the public key": hs256_with_public_key,
    "non-ASCII": lambda: "töken",
}


@pytest.mark.parametrize("path", ["/a2a", "/v1/assess"])
@pytest.mark.parametrize("case", REJECTED)
def test_bad_tokens_are_rejected_with_401(client, case, path):
    token = REJECTED[case]()
    headers = {"Authorization": f"Bearer {token}".encode("latin-1")} if token else {}
    reply = client.post(path, json=build_request(), headers=headers)
    assert reply.status_code == 401
    assert reply.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'


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
    {"exp": int(time.time()) - 10},                      # 10 s of clock skew is tolerated
])
def test_provider_specific_token_shapes_are_accepted(client, claims):
    assert call(client, make_token(**claims)).status_code == 200


def test_ec_keys_work_too(client, provider):
    key = ec.generate_private_key(ec.SECP256R1())
    provider.keys = [{**json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key())), "kid": "ec1", "use": "sig"}]
    assert call(client, make_token(key=key, kid="ec1", alg="ES256")).status_code == 200


def test_custom_scope_name(client, monkeypatch):
    monkeypatch.setenv("FRA_OAUTH_SCOPE", "claims.fraud")
    assert call(client, make_token()).status_code == 403
    assert call(client, make_token(scope="claims.fraud")).status_code == 200


def test_keys_are_cached_and_rotation_is_picked_up(client, provider):
    for _ in range(3):
        assert call(client, make_token()).status_code == 200
    assert provider.fetches == 1  # cached

    provider.keys = [jwk(PROVIDER_KEY, "k1"), jwk(OTHER_KEY, "k2")]  # the provider rotates in a new key
    assert call(client, make_token(key=OTHER_KEY, kid="k2")).status_code == 200
    assert provider.fetches == 2


def test_forged_key_ids_cannot_flood_the_provider(client, provider):
    for i in range(20):
        assert call(client, make_token(key=OTHER_KEY, kid=f"forged-{i}")).status_code == 401
    assert provider.fetches == 2  # the first load, then a single refresh within JWKS_REFRESH_SECONDS


@pytest.mark.parametrize("error", [jwt.PyJWKClientConnectionError("down"), ValueError("not JSON"), OSError("reset")])
def test_unreachable_key_endpoint_fails_closed_with_503(client, provider, error, caplog):
    provider.error = error
    token = make_token()
    with caplog.at_level(logging.INFO, logger="fra.server"):
        reply = call(client, token)
    assert reply.status_code == 503 and "error" in reply.headers["WWW-Authenticate"]
    assert token not in caplog.text


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


# ---------------------------------------------------------------- startup check

def start(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with TestClient(server.app) as started:
        return started.get("/health").status_code


def test_server_refuses_to_start_without_any_authentication(monkeypatch):
    with pytest.raises(RuntimeError, match="no authentication configured"):
        start(monkeypatch)


def test_server_refuses_to_start_with_half_configured_oauth(monkeypatch):
    with pytest.raises(RuntimeError, match="also set FRA_OAUTH_AUDIENCE, FRA_OAUTH_JWKS_URL"):
        start(monkeypatch, FRA_OAUTH_ISSUER=ISSUER, FRA_API_TOKEN="test-token")


def test_server_refuses_a_non_http_key_url(monkeypatch):
    with pytest.raises(RuntimeError, match="must be an http"):
        start(monkeypatch, FRA_OAUTH_ISSUER=ISSUER, FRA_OAUTH_AUDIENCE=AUDIENCE, FRA_OAUTH_JWKS_URL="file:///etc/passwd")


@pytest.mark.parametrize("env", [
    {"FRA_API_TOKEN": "test-token"},
    {"FRA_OAUTH_ISSUER": ISSUER, "FRA_OAUTH_AUDIENCE": AUDIENCE, "FRA_OAUTH_JWKS_URL": "https://idp.example/certs"},
])
def test_server_starts_with_either_kind_of_authentication(monkeypatch, env):
    assert start(monkeypatch, **env) == 200


def test_settings_removed_after_startup_fail_closed(monkeypatch):
    monkeypatch.setenv("FRA_OAUTH_ISSUER", ISSUER)  # half configured at request time
    reply = TestClient(server.app).post("/v1/assess", json=build_request(), headers={"Authorization": "Bearer x"})
    assert reply.status_code == 503
