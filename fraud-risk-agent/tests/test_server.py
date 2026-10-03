"""Phase 5 tests: A2A Agent Card, bearer auth, JSON-RPC message/send and the REST binding."""
import pytest
from conftest import build_request
from fastapi.testclient import TestClient

import server
from fra_agent import FraudRiskAgent

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FRA_API_TOKEN", "test-token")
    monkeypatch.setattr(server, "agent", FraudRiskAgent())  # template only: no Ollama in CI
    return TestClient(server.app)


def rpc(claim, method="message/send"):
    message = {"role": "user", "messageId": "msg-1", "parts": [{"kind": "data", "data": claim}]}
    return {"jsonrpc": "2.0", "id": 7, "method": method, "params": {"message": message}}


def test_agent_card_is_public(client):
    card = client.get("/.well-known/agent.json").json()
    assert [s["id"] for s in card["skills"]] == ["assess_fraud_risk"]
    assert card["version"]


@pytest.mark.parametrize("path", ["/a2a", "/v1/assess"])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_unauthenticated_calls_are_rejected(client, path, headers):
    assert client.post(path, json=build_request(), headers=headers).status_code == 401


def test_a2a_message_send_returns_result_in_a_data_part(client):
    claim = build_request(claim_history__potential_duplicate=True)
    first = client.post("/a2a", json=rpc(claim), headers=AUTH).json()
    data = first["result"]["parts"][0]["data"]
    assert (first["id"], data["status"], data["risk_score"], data["risk_flags"]) == (7, "COMPLETED", 70, ["DUPLICATE_PATTERN"])
    # AT-8 over the wire: the repeat call gets a byte-identical reply.
    assert client.post("/a2a", json=rpc(claim), headers=AUTH).json() == first


def test_a2a_errors(client):
    assert client.post("/a2a", json=rpc(build_request(), method="tasks/get"), headers=AUTH).json()["error"]["code"] == -32601
    no_data = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"parts": []}}}
    assert client.post("/a2a", json=no_data, headers=AUTH).json()["error"]["code"] == -32602


def test_rest_binding(client):
    assert client.post("/v1/assess", json=build_request(vehicle_policy_mismatch=True), headers=AUTH).json()["risk_score"] == 45
    bad = client.post("/v1/assess", content=b"not json", headers=AUTH).json()
    assert bad["status"] == "FAILED"
