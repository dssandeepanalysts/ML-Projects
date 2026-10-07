"""Phase 5 tests: A2A Agent Card, bearer auth, JSON-RPC message/send and the REST binding."""
import asyncio
import json
import logging

import pytest
from conftest import build_request
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

import server
from fra_agent import FraudRiskAgent, validate_request

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FRA_API_TOKEN", "test-token")
    monkeypatch.setattr(server, "agent", FraudRiskAgent())  # template only: no Ollama in CI
    return TestClient(server.app)


def rpc(claim, method="message/send"):
    message = {"role": "user", "messageId": "msg-1", "parts": [{"kind": "data", "data": claim}]}
    return {"jsonrpc": "2.0", "id": 7, "method": method, "params": {"message": message}}


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok", "version": "1.0.0", "llm_wording": "off"}


def test_agent_card_is_public_and_meets_spec_3_2(client):
    card = client.get("/.well-known/agent.json").json()
    assert [s["id"] for s in card["skills"]] == ["assess_fraud_risk"]
    skill = card["skills"][0]
    assert skill["inputModes"] == skill["outputModes"] == ["application/json", "text/plain"]
    assert validate_request(json.loads(skill["examples"][0])) is None  # a complete, valid example request
    assert "correlation_id" in skill["description"]  # tells a calling agent which fields to send
    assert card["url"].endswith("/a2a")  # endpoint URL
    assert card["securitySchemes"]["bearer"] == {"type": "http", "scheme": "bearer"}  # auth scheme
    assert card["security"] == [{"bearer": []}]
    assert card["version"]  # stored by Pega as AgentCardVersion


@pytest.mark.parametrize("path", ["/a2a", "/v1/assess"])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer t\u00f6ken".encode("latin-1")}])
def test_unauthenticated_calls_are_rejected(client, path, headers):
    assert client.post(path, json=build_request(), headers=headers).status_code == 401


def test_a2a_message_send_returns_result_in_a_data_part(client):
    claim = build_request(claim_history__potential_duplicate=True)
    first = client.post("/a2a", json=rpc(claim), headers=AUTH).json()
    data = first["result"]["parts"][0]["data"]
    assert (first["id"], data["status"], data["risk_score"], data["risk_flags"]) == (7, "COMPLETED", 70, ["DUPLICATE_PATTERN"])
    # AT-8 over the wire: the repeat call gets a byte-identical reply.
    assert client.post("/a2a", json=rpc(claim), headers=AUTH).json() == first


CLAIM = build_request(claim_history__potential_duplicate=True)


def rpc_with(parts, **message_fields):
    message = {"role": "user", "messageId": "m-2", "parts": parts, **message_fields}
    return {"jsonrpc": "2.0", "id": 9, "method": "message/send", "params": {"message": message}}


@pytest.mark.parametrize("parts", [
    [{"kind": "text", "text": json.dumps(CLAIM)}],  # a Pega AI agent: its language model sends JSON text
    [{"kind": "text", "text": "Please assess this claim:\n```json\n" + json.dumps(CLAIM, indent=2) + "\n```"}],
    [{"type": "data", "data": CLAIM}],  # A2A before version 0.3
    [{"kind": "data", "data": json.dumps(CLAIM)}],  # a data part holding JSON as a string
    [{"kind": "data", "data": {"claim": CLAIM}}],  # wrapped in one outer object
    [{"kind": "text", "text": "Assess the fraud risk."}, {"kind": "data", "data": CLAIM}],
], ids=["json-text", "json-in-prose", "a2a-0.2-type", "data-as-string", "wrapped", "text-and-data"])
def test_a2a_finds_the_claim_in_the_shapes_callers_send(client, parts):
    data = client.post("/a2a", json=rpc_with(parts), headers=AUTH).json()["result"]["parts"][0]["data"]
    assert (data["status"], data["risk_score"], data["risk_flags"]) == ("COMPLETED", 70, ["DUPLICATE_PATTERN"])


def test_a2a_message_without_a_claim_is_told_what_to_send(client, caplog):
    parts = [{"kind": "text", "text": "Is claim AC-1002 risky?"}, {"kind": "evil\nkind", "x": 1}]
    with caplog.at_level(logging.INFO):
        reply = client.post("/a2a", json=rpc_with(parts), headers=AUTH).json()
    data = reply["result"]["parts"][0]["data"]  # an answer the calling agent can act on, not a protocol error
    assert data["status"] == "FAILED" and "risk_score" not in data
    assert "correlation_id" in data["reasoning"] and "loss {cause" in data["reasoning"]
    assert "no claim found in the message (parts: text, other)" in caplog.text
    assert "AC-1002" not in caplog.text and "evil" not in caplog.text  # the caller's text is never logged


def test_a2a_reply_carries_the_result_as_text_too_and_keeps_the_context(client):
    reply = client.post("/a2a", json=rpc_with([{"kind": "data", "data": CLAIM}], contextId="ctx-7"), headers=AUTH)
    result = reply.json()["result"]
    assert json.loads(result["parts"][1]["text"]) == result["parts"][0]["data"]  # for a calling agent's model
    assert result["contextId"] == "ctx-7"


def test_a2a_errors(client):
    assert client.post("/a2a", json=rpc(build_request(), method="tasks/get"), headers=AUTH).json()["error"]["code"] == -32601
    no_data = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"parts": []}}}
    assert client.post("/a2a", json=no_data, headers=AUTH).json()["error"]["code"] == -32602


def test_rest_binding(client):
    assert client.post("/v1/assess", json=build_request(vehicle_policy_mismatch=True), headers=AUTH).json()["risk_score"] == 45
    bad = client.post("/v1/assess", content=b"not json", headers=AUTH).json()
    assert bad["status"] == "FAILED"


DEEP_JSON = b"[" * 200_000 + b"]" * 200_000


def test_deeply_nested_body_gets_a_parseable_answer(client):
    rest = client.post("/v1/assess", content=DEEP_JSON, headers=AUTH)
    assert (rest.status_code, rest.json()["status"]) == (200, "FAILED")
    rpc_reply = client.post("/a2a", content=DEEP_JSON, headers=AUTH)
    assert rpc_reply.json()["error"]["code"] == -32700


def test_unexpected_server_error_still_returns_json(monkeypatch):  # spec s11, internal-error row
    class Broken:
        def assess(self, payload):
            raise RuntimeError("bug")

    monkeypatch.setenv("FRA_API_TOKEN", "test-token")
    monkeypatch.setattr(server, "agent", Broken())
    client = TestClient(server.app, raise_server_exceptions=False)
    rest = client.post("/v1/assess", json=build_request(), headers=AUTH)
    assert (rest.status_code, rest.json()["status"]) == (500, "FAILED")
    assert client.post("/a2a", json=rpc(build_request()), headers=AUTH).json()["error"]["code"] == -32603


def test_health_never_waits_for_the_assessment_thread_pool():
    assert asyncio.iscoroutinefunction(server.health)  # runs on the event loop, not in a pool thread


def test_request_that_waited_too_long_gets_the_template(client, monkeypatch):
    # Pega allows 15 s: a request that queued past the budget must not also wait for the LLM.
    reply = ("One indicator was found on this collision claim: a potential duplicate of this claim "
             "was already flagged against the same policy and vehicle.")
    calls = []
    monkeypatch.setattr(server, "agent", FraudRiskAgent(llm=RunnableLambda(lambda _: calls.append(1) or reply)))
    claim = build_request(claim_history__potential_duplicate=True)

    monkeypatch.setattr(server, "QUEUE_BUDGET", 0.0)
    late = client.post("/v1/assess", json=claim, headers=AUTH).json()
    assert calls == [] and late["risk_score"] == 70 and late["reasoning"] != reply

    monkeypatch.setattr(server, "QUEUE_BUDGET", 60.0)
    on_time = client.post("/v1/assess", json=claim, headers=AUTH).json()
    assert calls == [1] and on_time["reasoning"] == reply
