import asyncio
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.destiny_knowledge import ManifestKnowledgeProvider
from app.main import app, create_app


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


def test_chat_rejects_actual_unauthenticated_request_with_connect_guidance() -> None:
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Check my Titan build."})

    assert response.status_code == 401
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
    assert len(tools) == 12
    names = {value["name"] for value in tools}
    assert {"get_content_progression", "get_title_progress"} <= names


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


def _pending_warmup(calls: list[str]) -> Any:
    async def warm_index(self: Any) -> dict[str, Any]:
        calls.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            calls.append("cancelled")
            raise
        return {}

    return warm_index


def test_startup_warmup_runs_in_background_and_never_blocks_health(monkeypatch: Any) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ManifestKnowledgeProvider, "warm_index", _pending_warmup(calls))
    application = create_app(Settings(bungie_api_key="test-key", manifest_index_warmup=True))

    with TestClient(application) as client:
        response = client.get("/api/health")
        for _ in range(100):
            if calls:
                break
            time.sleep(0.01)

    # Health answered while the warmup was still pending; shutdown cancelled it cleanly.
    assert response.status_code == 200
    assert calls == ["started", "cancelled"]


@pytest.mark.parametrize(
    "settings",
    [
        {"bungie_api_key": "", "manifest_index_warmup": True},
        {"bungie_api_key": "test-key", "manifest_index_warmup": False},
    ],
)
def test_startup_warmup_is_skipped_without_api_key_or_when_disabled(
    monkeypatch: Any, settings: dict[str, Any]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ManifestKnowledgeProvider, "warm_index", _pending_warmup(calls))
    application = create_app(Settings(**settings))

    with TestClient(application) as client:
        assert client.get("/api/health").status_code == 200
        provider = application.state.services.knowledge.providers[0]
        assert provider.warmup_state == {"state": "disabled"}

    assert calls == []


def test_test_environment_never_warms_the_manifest_index() -> None:
    assert Settings().manifest_index_warmup is False
