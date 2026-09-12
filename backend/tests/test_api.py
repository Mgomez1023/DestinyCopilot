from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_status_without_configuration() -> None:
    with TestClient(app) as client:
        response = client.get("/api/auth/status")
    assert response.status_code == 200
    assert response.json()["authenticated"] is False


def test_exact_https_oauth_callback_path_is_registered() -> None:
    with TestClient(app) as client:
        response = client.get("/api/auth/callback")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired OAuth callback state."


def test_debug_guardian_tool_catalog_is_available_in_development() -> None:
    with TestClient(app) as client:
        response = client.get("/api/debug/guardian-tools")
    assert response.status_code == 200
    assert len(response.json()["tools"]) == 8


def test_debug_destiny_knowledge_catalog_is_available_in_development() -> None:
    with TestClient(app) as client:
        response = client.get("/api/debug/destiny-knowledge")
    assert response.status_code == 200
    assert {tool["name"] for tool in response.json()["tools"]} == {
        "search_destiny_entities",
        "get_item_details",
        "get_activity_details",
        "get_quest_details",
        "find_item_source",
    }
