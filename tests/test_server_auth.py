from __future__ import annotations

from unittest.mock import patch

import apps.server.app as server_app


def make_client():
    """Return a test client with in-memory user store."""
    server_app.db = None  # ensure in-memory store is used
    server_app.app.config["TESTING"] = False
    return server_app.app.test_client()


def auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def test_requires_auth():
    client = make_client()
    resp = client.get("/api/v1/users/me")
    assert resp.status_code == 401


@patch("apps.server.app.auth.verify_id_token")
def test_profile_update_and_uniqueness(mock_verify):
    client = make_client()
    mock_verify.return_value = {"uid": "alice"}

    # Initially no profile
    resp = client.get("/api/v1/users/me", headers=auth_header())
    assert resp.status_code == 200
    assert resp.get_json() == {}

    # Update display name and publicId
    resp = client.put(
        "/api/v1/users/me",
        json={"displayName": "Alice", "publicId": "alice"},
        headers=auth_header(),
    )
    assert resp.status_code == 200
    assert resp.get_json()["displayName"] == "Alice"

    # Second user cannot take same publicId
    mock_verify.return_value = {"uid": "bob"}
    resp = client.put(
        "/api/v1/users/me",
        json={"publicId": "alice"},
        headers=auth_header(),
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "publicId taken"
