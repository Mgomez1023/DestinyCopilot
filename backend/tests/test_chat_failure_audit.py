"""Playtest 01 chat-reliability audit.

Reproduces, with deterministic fixtures, which backend failures reach the user as the
single generic stream error, and proves the failure log can tell them apart without
carrying private data. These tests characterise the collapse; they do not claim any of
these paths caused the production failures (that needs the Railway log line).
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai import RecommendationService
from app.bungie.client import BungieAPIError
from app.chat_stream import describe_chat_failure, safe_stream_error_message
from app.config import Settings
from app.models import ChatRequest
from app.routes import chat as chat_routes
from tests.test_chat_stream import parse_sse
from tests.test_guardian_tools import (
    FakeOpenAI,
    FakeResponses,
    ScriptedStreamingResponses,
    ScriptedWebResponses,
    assistant_message,
    guardian_context,
    web_call,
)

GENERIC = "The Copilot could not complete that request. Please try again."


class CrashingKnowledge:
    """Manifest-like provider whose lookup fails the way an unavailable database does."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "find_item_source",
                "description": "Find an item source.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    @staticmethod
    def handles(name: str) -> bool:
        return name == "find_item_source"

    @staticmethod
    def category_for_tool(name: str) -> str:
        del name
        return "manifest"

    @staticmethod
    def has_category(category: str) -> bool:
        return category == "manifest"

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del name, arguments
        raise self.error


def _tool_call(name: str = "find_item_source") -> Any:
    return SimpleNamespace(
        type="function_call", name=name, arguments="{}", call_id="call-1", id="fc-1"
    )


def _service(client: Any, knowledge: Any = None) -> RecommendationService:
    service = RecommendationService(Settings(openai_api_key="test-key"), knowledge)
    service.client = client
    return service


def _incomplete() -> RecommendationService:
    return _service(
        FakeOpenAI(
            [SimpleNamespace(type="reasoning")],
            request_tool=False,
            final_status="incomplete",
            incomplete_reason="max_output_tokens",
        )
    )


def _no_text() -> RecommendationService:
    return _service(FakeOpenAI([SimpleNamespace(type="reasoning")], request_tool=False))


def _unsupported_web_action() -> RecommendationService:
    # A hosted web-search item the stateless replay allow-list does not know, followed by a
    # function call so the item must be replayed into the next request.
    unknown = web_call(action=SimpleNamespace(type="open"))
    return _service(
        SimpleNamespace(
            responses=ScriptedWebResponses(
                [[unknown, _tool_call("get_character_summary")]], ["", "unused"]
            )
        )
    )


def _unsupported_annotation() -> RecommendationService:
    message = assistant_message("Checking.")
    message.content[0].annotations = [SimpleNamespace(type="future_citation")]
    return _service(
        SimpleNamespace(
            responses=ScriptedWebResponses(
                [[message, _tool_call("get_character_summary")]], ["", "unused"]
            )
        )
    )


def _manifest_unavailable() -> RecommendationService:
    return _service(
        SimpleNamespace(
            responses=ScriptedWebResponses([[_tool_call()]], ["", "unused"]),
        ),
        CrashingKnowledge(BungieAPIError("The local Destiny manifest database is unavailable.")),
    )


def _manifest_bug() -> RecommendationService:
    return _service(
        SimpleNamespace(responses=ScriptedWebResponses([[_tool_call()]], ["", "unused"])),
        CrashingKnowledge(KeyError("displayProperties")),
    )


CASES = {
    "incomplete_max_output_tokens": (_incomplete, RuntimeError, "ai:_require_completed_response"),
    "empty_final_text": (_no_text, RuntimeError, "ai:chat"),
    "unsupported_web_action": (
        _unsupported_web_action,
        RuntimeError,
        "ai:_normalize_replay_web_search_call",
    ),
    "unsupported_annotation": (
        _unsupported_annotation,
        RuntimeError,
        "ai:_normalize_replay_annotation",
    ),
    "manifest_unavailable": (_manifest_unavailable, BungieAPIError, "ai:_execute_tool_call"),
    "manifest_unexpected_shape": (_manifest_bug, KeyError, "ai:_execute_tool_call"),
}


def _run(factory: Any) -> BaseException:
    service = factory()
    with pytest.raises(Exception) as caught:
        asyncio.run(service.chat(ChatRequest(message="Will The Coil drop it?"), guardian_context()))
    return caught.value


@pytest.mark.parametrize("case", list(CASES))
def test_distinct_backend_failures_collapse_to_one_generic_stream_message(case: str) -> None:
    factory, expected_type, _ = CASES[case]
    error = _run(factory)

    assert isinstance(error, expected_type)
    # Every one of these reaches the user as the identical frontend text.
    assert safe_stream_error_message(error) == GENERIC


@pytest.mark.parametrize("case", list(CASES))
def test_failure_diagnostics_distinguish_each_cause(case: str) -> None:
    factory, expected_type, origin_prefix = CASES[case]
    details = describe_chat_failure(_run(factory))

    assert details["error_type"] == expected_type.__name__
    assert details["origin"].startswith(origin_prefix)


def test_tool_error_outside_the_caught_set_aborts_the_whole_turn() -> None:
    # _execute_tool_call only converts ValueError-family errors into a tool result, so a
    # Manifest outage (BungieAPIError is a RuntimeError) is not reported back to the model.
    error = _run(_manifest_unavailable)

    assert isinstance(error, BungieAPIError)


def test_diagnostics_omit_messages_and_private_values() -> None:
    secret = "access-token-123 character_id=9988 membership 4611686018"
    try:
        raise RuntimeError(secret)
    except RuntimeError as exc:
        details = describe_chat_failure(exc)
    try:
        raise BungieAPIError(secret)
    except BungieAPIError as exc:
        bungie_details = describe_chat_failure(exc)

    for value in (details, bungie_details):
        assert "access-token" not in str(value)
        assert "9988" not in str(value)
    assert "reason" not in bungie_details


def test_ai_runtime_error_reason_is_the_fixed_internal_message() -> None:
    details = describe_chat_failure(_run(_unsupported_web_action))

    assert details["reason"].startswith("Unsupported Responses web-search action type")


def _stream_client(monkeypatch: pytest.MonkeyPatch, service: RecommendationService) -> TestClient:
    async def credentials(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
        return "account", "access-token"

    async def load(*_args: Any, **_kwargs: Any) -> Any:
        return guardian_context()

    monkeypatch.setattr(chat_routes, "guardian_credentials", credentials)
    monkeypatch.setattr(chat_routes, "load_guardian", load)
    app = FastAPI()
    app.state.services = SimpleNamespace(recommendations=service, guardian_refresh=None)
    app.include_router(chat_routes.router)
    return TestClient(app)


def test_stream_failure_log_is_diagnosable_while_user_error_stays_generic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    unknown = web_call(action=SimpleNamespace(type="open"))
    streamed = ScriptedStreamingResponses(
        [FakeResponses.response([unknown, _tool_call("get_character_summary")])]
    )
    service = _service(SimpleNamespace(responses=streamed))
    with (
        caplog.at_level(logging.WARNING, logger="app.routes.chat"),
        _stream_client(monkeypatch, service) as client,
    ):
        response = client.post(
            "/api/chat/stream", json={"message": "I want to get mint retrograde"}
        )

    events = parse_sse(response.text)
    assert events[-1] == ("error", {"message": GENERIC})
    record = next(r.getMessage() for r in caplog.records if "Streaming chat failed" in r.message)
    assert "error_type" in record and "RuntimeError" in record
    assert "_normalize_replay_web_search_call" in record
    assert f"trace_id={service.latest_trace()['trace_id']}" in record
    assert "access-token" not in record
    assert "mint retrograde" not in record


def test_non_stream_route_logs_and_returns_502(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def load(*_args: Any, **_kwargs: Any) -> Any:
        return guardian_context()

    monkeypatch.setattr(chat_routes, "load_guardian", load)
    app = FastAPI()
    app.state.services = SimpleNamespace(recommendations=_incomplete(), guardian_refresh=None)
    app.include_router(chat_routes.router)
    with caplog.at_level(logging.WARNING, logger="app.routes.chat"), TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "What now?"})

    assert response.status_code == 502
    assert any("_require_completed_response" in r.getMessage() for r in caplog.records)
