import json
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Literal

from pydantic import BaseModel, Field

from app.models import ChatResponse, ChatSource
from app.response_quality import ResponseMode

StreamStatusStage = Literal[
    "guardian",
    "loadout",
    "inventory",
    "manifest",
    "guide",
    "live",
    "web",
    "comparison",
    "build",
    "final",
]


class StatusStreamEvent(BaseModel):
    event: Literal["status"] = "status"
    stage: StreamStatusStage
    label: str = Field(min_length=1, max_length=80)


class MessageDeltaStreamEvent(BaseModel):
    event: Literal["message_delta"] = "message_delta"
    delta: str = Field(min_length=1, max_length=512)


class SourcesStreamEvent(BaseModel):
    event: Literal["sources"] = "sources"
    sources: list[ChatSource] = Field(max_length=8)


class CompletedStreamEvent(BaseModel):
    event: Literal["completed"] = "completed"
    response_mode: ResponseMode
    source: Literal["openai", "local"]


class ErrorStreamEvent(BaseModel):
    event: Literal["error"] = "error"
    message: str = Field(min_length=1, max_length=240)


ChatStreamEvent = (
    StatusStreamEvent
    | MessageDeltaStreamEvent
    | SourcesStreamEvent
    | CompletedStreamEvent
    | ErrorStreamEvent
)
StreamEventEmitter = Callable[[ChatStreamEvent], Awaitable[None]]

STATUS_LABELS: dict[StreamStatusStage, str] = {
    "loadout": "Checking your loadout…",
    "inventory": "Searching your inventory…",
    "guardian": "Checking your Guardian…",
    "manifest": "Checking Destiny data…",
    "guide": "Looking up a guide…",
    "live": "Checking current Destiny activity…",
    "web": "Searching current Destiny info…",
    "comparison": "Comparing the available options…",
    "build": "Preparing your build…",
    "final": "Putting together your recommendation…",
}
WEB_SEARCH_STREAM_EVENTS = {
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.web_search_call.completed",
}

_CAMPAIGN_REQUEST = re.compile(
    r"\b(?:campaign|final shape|edge of fate|lightfall|witch queen|beyond light)\b",
    re.IGNORECASE,
)
_QUEST_REQUEST = re.compile(r"\bquests?\b", re.IGNORECASE)
_LOADOUT_REQUEST = re.compile(
    r"\b(?:equipped|loadout|weapons?|armor|subclass|build|boss dps|add clear|"
    r"what should i replace|do i own a better|prep(?:are)? me for)\b",
    re.IGNORECASE,
)


class StreamStatusReporter:
    """Emit sparse, user-safe execution categories without internal payloads."""

    def __init__(self, emit: StreamEventEmitter) -> None:
        self._emit = emit
        self._emitted: set[StreamStatusStage] = set()
        self.emitted_categories: list[StreamStatusStage] = []
        self.trace_id: str | None = None
        self.response_mode: ResponseMode | None = None

    async def emit(self, stage: StreamStatusStage, label: str | None = None) -> bool:
        if stage in self._emitted:
            return False
        self._emitted.add(stage)
        self.emitted_categories.append(stage)
        await self._emit(StatusStreamEvent(stage=stage, label=label or STATUS_LABELS[stage]))
        return True


def initial_guardian_status(message: str) -> tuple[StreamStatusStage, str]:
    if _CAMPAIGN_REQUEST.search(message):
        return "guardian", "Checking campaign progress…"
    if _QUEST_REQUEST.search(message):
        return "guardian", "Reviewing your quests…"
    if _LOADOUT_REQUEST.search(message):
        return "loadout", STATUS_LABELS["loadout"]
    return "guardian", STATUS_LABELS["guardian"]


def status_for_tool(tool_name: str, category: str) -> tuple[StreamStatusStage, str]:
    if category == "manifest":
        return "manifest", STATUS_LABELS["manifest"]
    if category == "guide":
        return "guide", STATUS_LABELS["guide"]
    if category == "live":
        return "live", STATUS_LABELS["live"]
    if tool_name in {"analyze_current_build", "get_build_details"}:
        return "loadout", STATUS_LABELS["loadout"]
    if tool_name in {"find_build_alternatives", "search_inventory"}:
        return "inventory", STATUS_LABELS["inventory"]
    guardian_labels = {
        "get_equipped_loadout": "Checking your loadout…",
        "get_active_quests": "Reviewing your quests…",
        "get_progression": "Checking your progress…",
        "get_content_progression": "Checking campaign progress…",
    }
    return "guardian", guardian_labels.get(tool_name, STATUS_LABELS["guardian"])


def is_web_search_stream_event(event_type: object) -> bool:
    return isinstance(event_type, str) and event_type in WEB_SEARCH_STREAM_EVENTS


def message_delta_events(message: str, chunk_size: int = 160) -> Iterable[MessageDeltaStreamEvent]:
    if chunk_size < 1 or chunk_size > 512:
        raise ValueError("Stream chunk size must be between 1 and 512 characters.")
    for offset in range(0, len(message), chunk_size):
        yield MessageDeltaStreamEvent(delta=message[offset : offset + chunk_size])


def response_events(response: ChatResponse, response_mode: ResponseMode) -> list[ChatStreamEvent]:
    events: list[ChatStreamEvent] = list(message_delta_events(response.message))
    if response.sources:
        events.append(SourcesStreamEvent(sources=response.sources))
    events.append(CompletedStreamEvent(response_mode=response_mode, source=response.source))
    return events


def serialize_sse(event: ChatStreamEvent) -> str:
    payload = event.model_dump(mode="json", exclude={"event"})
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.event}\ndata: {encoded}\n\n"


def safe_stream_error_message(error: Exception) -> str:
    detail = getattr(error, "detail", None)
    if isinstance(detail, str) and detail:
        return detail[:240]
    if isinstance(error, RuntimeError) and str(error) == (
        "The recommendation service is temporarily unavailable."
    ):
        return str(error)
    return "The Copilot could not complete that request. Please try again."
