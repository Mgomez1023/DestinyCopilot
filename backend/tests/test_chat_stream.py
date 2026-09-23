import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat_stream import (
    CompletedStreamEvent,
    ErrorStreamEvent,
    MessageDeltaStreamEvent,
    SourcesStreamEvent,
    StatusStreamEvent,
    StreamStatusReporter,
    initial_guardian_status,
    message_delta_events,
    serialize_sse,
    status_for_tool,
)
from app.models import ChatResponse, ChatSource
from app.routes import chat as chat_routes


def parse_sse(value: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in value.strip().split("\n\n"):
        lines = frame.splitlines()
        name = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


@pytest.mark.parametrize(
    ("event", "name", "field"),
    [
        (StatusStreamEvent(stage="guardian", label="Checking your Guardian…"), "status", "stage"),
        (MessageDeltaStreamEvent(delta="Hello"), "message_delta", "delta"),
        (
            SourcesStreamEvent(sources=[ChatSource(title="Bungie", url="https://www.bungie.net/")]),
            "sources",
            "sources",
        ),
        (
            CompletedStreamEvent(response_mode="recommendation", source="openai"),
            "completed",
            "response_mode",
        ),
        (ErrorStreamEvent(message="Please try again."), "error", "message"),
    ],
)
def test_stream_event_serialization_and_sse_framing(event: Any, name: str, field: str) -> None:
    encoded = serialize_sse(event)

    assert encoded.startswith(f"event: {name}\ndata: ")
    assert encoded.endswith("\n\n")
    parsed = parse_sse(encoded)
    assert parsed[0][0] == name
    assert field in parsed[0][1]


def test_message_deltas_are_bounded_and_lossless() -> None:
    message = "A grounded answer with citations. " * 40
    events = list(message_delta_events(message, chunk_size=64))

    assert "".join(event.delta for event in events) == message
    assert all(len(event.delta) <= 64 for event in events)


def test_status_reporter_deduplicates_categories_and_hides_internal_tool_names() -> None:
    events: list[Any] = []

    async def emit(event: Any) -> None:
        events.append(event)

    reporter = StreamStatusReporter(emit)
    stage, label = status_for_tool("get_active_quests", "guardian")
    asyncio.run(reporter.emit(stage, label))
    asyncio.run(reporter.emit(stage, "Checking your Guardian…"))

    assert len(events) == 1
    encoded = serialize_sse(events[0])
    assert "Reviewing your quests" in encoded
    assert "get_active_quests" not in encoded
    assert "arguments" not in encoded


def test_build_statuses_are_specific_safe_and_deduplicatable() -> None:
    assert initial_guardian_status("Build around Sunshot.") == (
        "loadout",
        "Checking your loadout…",
    )
    assert status_for_tool("analyze_current_build", "guardian") == (
        "loadout",
        "Checking your loadout…",
    )
    assert status_for_tool("find_build_alternatives", "guardian") == (
        "inventory",
        "Searching your inventory…",
    )


class FakeStreamingRecommendations:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.delivery: dict[str, Any] | None = None

    async def chat(self, *_args: Any, stream_status: StreamStatusReporter, **_kwargs: Any) -> Any:
        stream_status.trace_id = "safe-trace"
        stream_status.response_mode = "comparison"
        await stream_status.emit("web")
        await stream_status.emit("web")
        await stream_status.emit("comparison")
        if self.fail:
            raise RuntimeError("secret tool argument character_id=123")
        return ChatResponse(
            message="Use the grounded option on your Titan.",
            source="openai",
            sources=[
                ChatSource(
                    title="Destiny update",
                    url="https://www.bungie.net/7/en/News/article/example",
                )
            ],
        )

    def mark_stream_delivery(self, trace_id: str | None, **values: Any) -> None:
        self.delivery = {"trace_id": trace_id, **values}


def streaming_client(monkeypatch: pytest.MonkeyPatch, recommendations: Any) -> TestClient:
    async def fake_credentials(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
        return "account", "access-token"

    async def fake_load_guardian(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace()

    monkeypatch.setattr(chat_routes, "guardian_credentials", fake_credentials)
    monkeypatch.setattr(chat_routes, "load_guardian", fake_load_guardian)
    test_app = FastAPI()
    test_app.state.services = SimpleNamespace(
        recommendations=recommendations,
        guardian_refresh=None,
    )
    test_app.include_router(chat_routes.router)
    return TestClient(test_app)


def test_stream_endpoint_orders_status_answer_sources_and_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recommendations = FakeStreamingRecommendations()
    with streaming_client(monkeypatch, recommendations) as client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "Given my Titan, Final Shape or Edge?"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names == [
        "status",
        "status",
        "status",
        "status",
        "message_delta",
        "sources",
        "completed",
    ]
    assert [data["stage"] for name, data in events if name == "status"] == [
        "guardian",
        "web",
        "comparison",
        "final",
    ]
    assert "character_id" not in response.text
    assert "access-token" not in response.text
    assert recommendations.delivery == {
        "trace_id": "safe-trace",
        "final_response_streamed": True,
        "stream_completed": True,
        "emitted_status_categories": ["guardian", "web", "comparison", "final"],
    }


def test_stream_endpoint_returns_safe_error_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    recommendations = FakeStreamingRecommendations(fail=True)
    with streaming_client(monkeypatch, recommendations) as client:
        response = client.post("/api/chat/stream", json={"message": "Help me."})

    events = parse_sse(response.text)
    assert events[-1] == (
        "error",
        {"message": "The Copilot could not complete that request. Please try again."},
    )
    assert "secret tool argument" not in response.text
    assert "traceback" not in response.text.casefold()
    assert recommendations.delivery == {
        "trace_id": "safe-trace",
        "final_response_streamed": False,
        "stream_completed": False,
        "emitted_status_categories": ["guardian", "web", "comparison"],
    }
