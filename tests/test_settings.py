import pytest
from fastapi import HTTPException

from app.services import auth_service


@pytest.mark.parametrize(
    ("field", "value", "limit"),
    [
        ("name", "N" * 101, 100),
        ("department", "D" * 101, 100),
        ("organization", "O" * 151, 150),
    ],
)
def test_profile_update_rejects_overlong_supported_fields(monkeypatch, field, value, limit):
    monkeypatch.setattr(auth_service, "get_collection", lambda _name: None)
    with pytest.raises(HTTPException) as error:
        auth_service.update_user_profile("user-1", {field: value})
    assert error.value.status_code == 400
    assert str(limit) in error.value.detail


def test_profile_update_is_scoped_to_authenticated_user_id(monkeypatch):
    users = {
        "signed-in": {
            "_id": "signed-in", "name": "Signed In", "email": "signed-in@example.com",
            "role": "user", "organization": "", "department": "",
        },
        "other-user": {
            "_id": "other-user", "name": "Other User", "email": "other@example.com",
            "role": "user", "organization": "", "department": "",
        },
    }
    monkeypatch.setattr(auth_service, "_in_memory_users", users)
    monkeypatch.setattr(auth_service, "get_collection", lambda _name: None)

    updated = auth_service.update_user_profile(
        "signed-in", {"user_id": "other-user", "name": "Updated Signed In"}
    )

    assert updated["name"] == "Updated Signed In"
    assert users["other-user"]["name"] == "Other User"


def test_profile_update_short_name_returns_validation_error(monkeypatch):
    monkeypatch.setattr(auth_service, "get_collection", lambda _name: None)
    with pytest.raises(HTTPException, match="Name must be at least 2 characters"):
        auth_service.update_user_profile("user-1", {"name": "A"})


def test_profile_endpoint_uses_signed_in_user_even_if_payload_names_another_user(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routes import auth as auth_routes

    captured = {}

    def record_update(user_id, updates):
        captured["user_id"] = user_id
        captured["updates"] = updates
        return {"id": user_id, "name": updates.get("name", "Signed In")}

    monkeypatch.setattr(auth_routes, "get_current_user", lambda _request: {"id": "signed-in-user"})
    monkeypatch.setattr(auth_routes, "update_user_profile", record_update)
    response = TestClient(app).post(
        "/api/auth/profile", json={"user_id": "other-user", "name": "Updated Name"}
    )

    assert response.status_code == 200
    assert captured["user_id"] == "signed-in-user"
    assert captured["updates"]["user_id"] == "other-user"


def test_profile_changes_appear_on_refreshed_settings_page_without_database(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    from app.main import app
    from app.routes import auth as auth_routes

    users = {
        "settings-user": {
            "_id": "settings-user", "name": "Original Name", "email": "profile@example.com",
            "role": "operator", "organization": "Old Facility", "department": "Operations",
            "created_at": "2026-01-01T00:00:00",
        }
    }
    monkeypatch.setattr(auth_service, "_in_memory_users", users)
    monkeypatch.setattr(auth_service, "get_collection", lambda _name: None)
    monkeypatch.setattr(auth_routes, "get_current_user", lambda _request: auth_service.sanitize_user(users["settings-user"]))
    monkeypatch.setattr(main, "get_current_user_optional", lambda _request: auth_service.sanitize_user(users["settings-user"]))
    client = TestClient(app)

    saved = client.post("/api/auth/profile", json={
        "name": "Updated Name", "department": "Inventory", "organization": "North Facility"
    })
    assert saved.status_code == 200
    assert client.get("/api/auth/me").json()["name"] == "Updated Name"

    refreshed = client.get("/settings")
    assert refreshed.status_code == 200
    assert 'value="Updated Name"' in refreshed.text
    assert 'value="Inventory"' in refreshed.text
    assert 'value="North Facility"' in refreshed.text
