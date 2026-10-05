import copy

import pytest

# AT-1 clean claim (spec s5.1 example with full-length hashes).
CLEAN_REQUEST = {
    "claim_id": "AC-1001",
    "correlation_id": "3f6c1b0a-7e2d-4c91-9a08-1d5e6f2b8c41",
    "policy_ref_hash": "9a41c3e8f7b2" + "0" * 52,
    "vin_hash": "77c0d9a1e4f6" + "1" * 52,
    "vehicle_policy_mismatch": False,
    "loss": {
        "cause": "COLLISION",
        "date": "2026-09-20",
        "reported_on": "2026-09-21",
        "description": "Rear-ended at a signal while waiting for the light to change.",
    },
    "claim_history": {
        "claims_last_24_months": 0,
        "days_since_policy_start": 262,
        "late_reported": False,
        "potential_duplicate": False,
    },
    "policy_data_available": True,
    "history_data_available": True,
}


def build_request(**changes):
    """Clean request with overrides; nested keys use '__', e.g. claim_history__late_reported=True."""
    request = copy.deepcopy(CLEAN_REQUEST)
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = request
        for key in parents:
            node = node[key]
        node[leaf] = value
    return request


@pytest.fixture
def make_request():
    return build_request


@pytest.fixture(autouse=True)
def no_auth_settings_from_the_shell(monkeypatch):
    """Each test sets the auth settings it needs; a developer's own FRA_OAUTH_* must not leak in."""
    for name in ("FRA_API_TOKEN", "FRA_OAUTH_ISSUER", "FRA_OAUTH_AUDIENCE", "FRA_OAUTH_JWKS_URL",
                 "FRA_OAUTH_SCOPE", "FRA_OAUTH_TOKEN_URL"):
        monkeypatch.delenv(name, raising=False)
