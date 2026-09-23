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


def test_streaming_chat_rejects_unauthenticated_request_before_sse_starts() -> None:
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"message": "What should I do?"})

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == "Connect your Bungie account first."


def test_exact_https_oauth_callback_path_is_registered() -> None:
    with TestClient(app) as client:
        response = client.get("/api/auth/callback")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired OAuth callback state."


def test_debug_guardian_tool_catalog_is_available_in_development() -> None:
    with TestClient(app) as client:
        response = client.get("/api/debug/guardian-tools")
    assert response.status_code == 200
    tools = response.json()["tools"]
    assert len(tools) == 11
    assert "get_content_progression" in {value["name"] for value in tools}


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
        "search_destiny_guides",
        "get_destiny_guide",
        "get_live_destiny_status",
        "get_weekly_rotation",
        "get_vendor_status",
        "get_current_activity_status",
        "search_live_destiny",
    }


def test_debug_recent_chat_trace_endpoint_is_available_in_development() -> None:
    with TestClient(app) as client:
        response = client.get("/api/debug/chat-traces")
    assert response.status_code == 200
    assert response.json() == {"traces": [], "count": 0}
