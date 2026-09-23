import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai import RecommendationService
from app.chat_stream import StreamStatusReporter, response_events
from app.config import Settings
from app.guardian_tools import (
    CharacterNotFoundError,
    GuardianToolService,
    InventorySearchRequest,
)
from app.models import (
    AvailableActivitySummary,
    CharacterSummary,
    ChatRequest,
    ChatTurn,
    DataAvailability,
    GuardianContext,
    InventorySummary,
    ItemStatSummary,
    ItemSummary,
    MilestoneSummary,
    ObjectiveSummary,
    ProgressionSummary,
    QuestSummary,
    RecentActivitySummary,
    SocketedPlugSummary,
)
from app.routes import chat as chat_routes
from app.session_planning import build_session_planning_context


def item(
    name: str,
    item_type: str,
    *,
    subtype: str | None = None,
    location: str = "equipped",
    tier: str = "Legendary",
    character_id: str | None = "char-1",
    equipped: bool = True,
    plugs: list[SocketedPlugSummary] | None = None,
    stats: list[ItemStatSummary] | None = None,
) -> ItemSummary:
    return ItemSummary(
        item_hash=abs(hash((name, location))) % 2**32,
        instance_id=f"{name}-{location}",
        name=name,
        item_type=item_type,
        item_subtype=subtype,
        tier=tier,
        bucket_name="Energy Weapons" if item_type == "Weapon" else "Helmet",
        power=2000 if item_type in {"Weapon", "Armor"} else None,
        location=location,
        character_id=character_id,
        is_equipped=equipped,
        socketed_plugs=[plug.name for plug in plugs or []],
        socketed_plug_details=plugs or [],
        stats=stats or [],
    )


def guardian_context() -> GuardianContext:
    now = datetime(2026, 9, 11, 12, tzinfo=UTC)
    objective = ObjectiveSummary(
        objective_hash=50,
        name="Defeat targets",
        progress=8,
        completion_value=10,
        progress_percent=80,
        activity_name="Nightfall",
        destination_name="Cosmodrome",
    )
    subclass_plugs = [
        SocketedPlugSummary(
            item_hash=1,
            name="Roaring Flames",
            item_type="Mod",
            category_identifier="shared.aspects",
        ),
        SocketedPlugSummary(
            item_hash=2,
            name="Ember of Torches",
            item_type="Mod",
            category_identifier="shared.fragments",
        ),
        SocketedPlugSummary(
            item_hash=3,
            name="Thermite Grenade",
            item_type="Mod",
            category_identifier="shared.grenades",
        ),
    ]
    armor_mod = SocketedPlugSummary(
        item_hash=4,
        name="Heavy Handed",
        item_type="Mod",
        category_identifier="enhancements.mods",
    )
    subclass = item("Sunbreaker", "Subclass", plugs=subclass_plugs)
    weapon = item("Funnelweb", "Weapon", subtype="Submachine Gun", tier="Exotic")
    armor = item(
        "Synthoceps",
        "Armor",
        tier="Exotic",
        plugs=[armor_mod],
        stats=[ItemStatSummary(name="Resilience", value=25)],
    )
    active_quest = QuestSummary(
        quest_hash=100,
        name="Into the Light",
        description="Complete the objective.",
        character_id="char-1",
        tracked=True,
        objectives=[objective],
    )
    redeemed_quest = QuestSummary(
        quest_hash=101,
        name="Finished Quest",
        character_id="char-1",
        completed=True,
        redeemed=True,
    )
    activity = AvailableActivitySummary(
        activity_hash=200,
        name="Nightfall",
        character_id="char-1",
        activity_type="Strike",
        destination="Cosmodrome",
        difficulty="Hard",
        recommended_power=2000,
        can_lead=True,
        can_join=True,
        objectives=[objective],
    )
    hidden_activity = AvailableActivitySummary(
        activity_hash=201,
        name="Hidden Test",
        character_id="char-1",
        is_visible=False,
    )
    progression = ProgressionSummary(
        progression_hash=300,
        name="Vanguard Rank",
        scope="character",
        character_id="char-1",
        level=12,
        progress_to_next_level=50,
        next_level_at=100,
    )
    character = CharacterSummary(
        character_id="char-1",
        class_name="Titan",
        race_name="Exo",
        gender_name="Male",
        power=2000,
        last_played=now,
        subclass=subclass,
        equipped_gear=[subclass, weapon, armor],
        quests=[active_quest, redeemed_quest],
        milestones=[
            MilestoneSummary(
                milestone_hash=400,
                name="Weekly Vanguard",
                character_id="char-1",
                activity_names=["Nightfall"],
                objectives=[objective],
            )
        ],
        progressions=[progression],
        available_activities=[activity, hidden_activity],
        recent_activities=[
            RecentActivitySummary(
                activity_hash=200,
                name="Older Strike",
                character_id="char-1",
                period=now - timedelta(days=1),
            ),
            RecentActivitySummary(
                activity_hash=202,
                name="Newest Strike",
                character_id="char-1",
                period=now,
                completed=True,
                duration_seconds=900,
            ),
        ],
    )
    vault_smg = item(
        "The Hero's Burden",
        "Weapon",
        subtype="Submachine Gun",
        location="vault",
        character_id=None,
        equipped=False,
    )
    vault_armor = item(
        "Vault Helm",
        "Armor",
        location="vault",
        character_id=None,
        equipped=False,
    )
    return GuardianContext(
        bungie_display_name="Guardian#1234",
        membership_id="membership-secret-shape",
        membership_type=3,
        platform_name="Steam",
        characters=[character],
        inventory=InventorySummary(
            total_items=2,
            vault_items=2,
            unique_item_hashes=2,
            items=[vault_smg, vault_armor],
            returned_items=2,
        ),
        data_availability=DataAvailability(fetched_at=now),
    )


def test_character_summary_is_compact() -> None:
    result = GuardianToolService(guardian_context()).get_character_summary()
    assert result["characters"][0]["class"] == "Titan"
    assert result["characters"][0]["subclass"] == "Sunbreaker"
    assert result["characters"][0]["equipped_weapons"] == ["Funnelweb"]
    assert "inventory" not in result
    assert "power" not in result["characters"][0]


def test_active_quests_excludes_completed_and_redeemed() -> None:
    result = GuardianToolService(guardian_context()).get_active_quests()
    assert result["count"] == 1
    assert result["quests"][0]["objectives"][0]["progress_percent"] == 80


def test_available_activities_filters_hidden_and_reports_launchability() -> None:
    result = GuardianToolService(guardian_context()).get_available_activities()
    assert result["total_matching"] == 1
    assert result["activities"][0]["activity_type"] == "Strike"
    assert result["activities"][0]["launchable"] is True
    assert "recommended_power" not in result["activities"][0]
    assert result["activities"][0]["power_eligibility"] == "unknown_not_compared"
    assert result["availability_scope"] == "guardian_character_activities"
    assert result["current_rotation_authoritative"] is False
    assert "does not establish" in result["limitations"][0]


def test_recent_activities_are_sorted_and_limited() -> None:
    result = GuardianToolService(guardian_context()).get_recent_activities(limit=1)
    assert result["activities"][0]["name"] == "Newest Strike"
    assert result["total_matching"] == 2


def test_inventory_search_applies_name_type_subtype_and_location_filters() -> None:
    tools = GuardianToolService(guardian_context())
    result = tools.search_inventory(
        InventorySearchRequest(
            query="hero",
            item_type="weapon",
            subtype="machine",
            bucket="energy",
            limit=25,
        )
    )
    assert result["total_matching"] == 1
    assert result["items"][0]["location"] == "vault"
    equipped = tools.search_inventory(InventorySearchRequest(equipped_only=True, limit=25))
    assert {value["name"] for value in equipped["items"]} == {
        "Sunbreaker",
        "Funnelweb",
        "Synthoceps",
    }


def test_build_details_extracts_subclass_exotics_stats_and_mods() -> None:
    result = GuardianToolService(guardian_context()).get_build_details("char-1")
    assert result["subclass"]["aspects"] == ["Roaring Flames"]
    assert result["subclass"]["fragments"] == ["Ember of Torches"]
    assert result["armor_stats"]["Resilience"] == 25
    assert result["mods"] == ["Heavy Handed"]
    assert {value["name"] for value in result["equipped_exotics"]} == {
        "Funnelweb",
        "Synthoceps",
    }


def test_missing_character_and_empty_results() -> None:
    tools = GuardianToolService(guardian_context())
    with pytest.raises(CharacterNotFoundError):
        tools.get_build_details("missing")
    result = tools.search_inventory(InventorySearchRequest(query="does-not-exist"))
    assert result == {"items": [], "total_matching": 0, "limit": 25, "truncated": False}


class FakeResponses:
    def __init__(
        self,
        final_output: list[Any] | None = None,
        *,
        final_status: str = "completed",
        incomplete_reason: str | None = None,
        request_tool: bool = True,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.final_output = final_output or [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Your Titan is ready.")],
            )
        ]
        self.final_status = final_status
        self.incomplete_reason = incomplete_reason
        self.request_tool = request_tool

    @staticmethod
    def response(
        output: list[Any],
        *,
        status: str = "completed",
        incomplete_reason: str | None = None,
    ) -> Any:
        for index, item in enumerate(output):
            if getattr(item, "type", None) != "message":
                continue
            if not hasattr(item, "id"):
                item.id = f"message-{index}"
            if not hasattr(item, "role"):
                item.role = "assistant"
            if not hasattr(item, "status"):
                item.status = "completed"
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text" and not hasattr(
                    content, "annotations"
                ):
                    content.annotations = []
        return SimpleNamespace(
            output=output,
            output_text="",
            status=status,
            incomplete_details=(
                SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
            ),
            error=None,
            usage=SimpleNamespace(
                input_tokens=125,
                output_tokens=60,
                output_tokens_details=SimpleNamespace(reasoning_tokens=20),
            ),
            model="gpt-5-mini",
            max_output_tokens=2000,
            reasoning=SimpleNamespace(effort="low"),
            truncation="disabled",
        )

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.request_tool and len(self.requests) == 1:
            call = SimpleNamespace(
                type="function_call",
                name="get_character_summary",
                arguments='{"character_id":null}',
                call_id="call-1",
                parsed_arguments={"character_id": None},
            )
            return self.response([call])
        # Reproduce the reported integration failure: the convenience accessor
        # is empty even though the normal message output contains assistant text.
        return self.response(
            self.final_output,
            status=self.final_status,
            incomplete_reason=self.incomplete_reason,
        )


class FakeOpenAI:
    def __init__(
        self,
        final_output: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.responses = FakeResponses(final_output, **kwargs)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_ai_refreshes_declared_guardian_tool_before_execution() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI()
    refreshed_for: list[str] = []

    async def refresh_guardian(tool_name: str) -> GuardianContext:
        refreshed_for.append(tool_name)
        return guardian_context()

    response = await service.chat(
        ChatRequest(message="What am I currently using?"),
        guardian_context(),
        refresh_guardian=refresh_guardian,
    )

    assert response.message == "Your Titan is ready."
    assert refreshed_for == ["get_character_summary"]


@pytest.mark.asyncio
async def test_content_progression_tool_preserves_stateless_function_call_loop() -> None:
    class ContentProgressionResponses(FakeResponses):
        async def create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return self.response(
                    [
                        SimpleNamespace(
                            type="function_call",
                            name="get_content_progression",
                            arguments=(
                                '{"character_id":"char-1","content_name":"The Final Shape"}'
                            ),
                            call_id="content-1",
                        )
                    ]
                )
            return self.response(self.final_output)

    context = guardian_context()
    context.characters[0].quests.append(
        QuestSummary(
            quest_hash=910,
            name="The Final Shape",
            step_name="Temptation",
            character_id="char-1",
        )
    )
    responses = ContentProgressionResponses()
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=responses)

    response = await service.chat(
        ChatRequest(message="How far am I into The Final Shape on my Titan?"), context
    )

    assert response.message == "Your Titan is ready."
    output = next(
        value
        for value in responses.requests[1]["input"]
        if isinstance(value, dict) and value.get("type") == "function_call_output"
    )
    result = json.loads(output["output"])
    assert result["content"]["status"] == "in_progress"
    assert result["content"]["current_step"] == "Temptation"
    assert output["call_id"] == "content-1"
    assert "previous_response_id" not in responses.requests[1]


def test_api_chat_returns_message_after_guardian_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI()
    service.client = fake

    async def fake_load_guardian(*_args: Any, **_kwargs: Any) -> GuardianContext:
        return guardian_context()

    monkeypatch.setattr(chat_routes, "load_guardian", fake_load_guardian)
    test_app = FastAPI()
    test_app.state.services = SimpleNamespace(recommendations=service)
    test_app.include_router(chat_routes.router)
    with TestClient(test_app) as client:
        response = client.post("/api/chat", json={"message": "How is my character?"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "Your Titan is ready.",
        "source": "openai",
        "sources": [],
    }
    assert len(fake.responses.requests) == 2
    first_messages = [
        value
        for value in fake.responses.requests[0]["input"]
        if isinstance(value, dict) and value.get("role")
    ]
    first_input = json.dumps(first_messages)
    assert "guardian_context" not in first_input
    tool_outputs = [
        value
        for value in fake.responses.requests[1]["input"]
        if isinstance(value, dict) and value.get("type") == "function_call_output"
    ]
    assert len(tool_outputs) == 1
    assert tool_outputs[0]["call_id"] == "call-1"
    assert json.loads(tool_outputs[0]["output"])["characters"][0]["class"] == "Titan"
    second_input = fake.responses.requests[1]["input"]
    assert [
        value.get("type") or f"{value.get('role')}_message"
        if isinstance(value, dict)
        else value.type
        for value in second_input
    ] == ["user_message", "function_call", "function_call_output"]
    replayed_call = second_input[1]
    assert replayed_call == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "get_character_summary",
        "arguments": '{"character_id":null}',
    }
    assert "parsed_arguments" not in json.dumps(second_input)
    second_request = fake.responses.requests[1]
    assert "After you have gathered enough account information" in second_request["instructions"]
    assert second_request["reasoning"] == {"effort": "low"}
    assert second_request["max_output_tokens"] == 2000
    assert second_request["truncation"] == "disabled"
    assert "previous_response_id" not in second_request


def test_completed_response_containing_assistant_text() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(request_tool=False)

    response = asyncio.run(
        service.chat(ChatRequest(message="Give me a recommendation."), guardian_context())
    )

    assert response.message == "Your Titan is ready."


def test_incomplete_response_due_to_max_output_tokens_is_rejected() -> None:
    reasoning_only = [SimpleNamespace(type="reasoning")]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(
        reasoning_only,
        request_tool=False,
        final_status="incomplete",
        incomplete_reason="max_output_tokens",
    )

    with pytest.raises(RuntimeError, match="max_output_tokens"):
        asyncio.run(service.chat(ChatRequest(message="What should I do?"), guardian_context()))


def test_completed_reasoning_only_response_is_rejected() -> None:
    reasoning_only = [SimpleNamespace(type="reasoning")]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(reasoning_only, request_tool=False)

    with pytest.raises(RuntimeError, match="without assistant text"):
        asyncio.run(service.chat(ChatRequest(message="What should I do?"), guardian_context()))


def test_ai_extracts_message_text_when_it_is_not_first_output_item() -> None:
    final_output = [
        SimpleNamespace(type="reasoning"),
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(type="output_text", text="Your Titan "),
                SimpleNamespace(type="output_text", text="is ready."),
            ],
        ),
    ]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI(final_output)
    service.client = fake

    response = asyncio.run(
        service.chat(ChatRequest(message="How is my character?"), guardian_context())
    )

    assert response.message == "Your Titan is ready."


def test_ai_blocks_unsupported_power_level_conclusion() -> None:
    unsafe_output = [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text="Your Power 191 is too low for activities recommending 300+.",
                )
            ],
        )
    ]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(unsafe_output, request_tool=False)

    response = asyncio.run(
        service.chat(ChatRequest(message="Am I high enough Power?"), guardian_context())
    )

    assert "too low" not in response.message
    assert "eligibility is unknown" in response.message


class ScriptedWebResponses:
    def __init__(self, outputs: list[list[Any]], texts: list[str]) -> None:
        self.outputs = outputs
        self.texts = texts
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        index = len(self.requests)
        self.requests.append(kwargs)
        return SimpleNamespace(
            output=self.outputs[index],
            output_text=self.texts[index],
            status="completed",
            incomplete_details=None,
            error=None,
            usage=None,
        )


class FakeResponseStream:
    def __init__(self, events: list[Any], response: Any) -> None:
        self.events = events
        self.response = response

    async def __aenter__(self) -> "FakeResponseStream":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def __aiter__(self):
        async def iterate():
            for event in self.events:
                yield event

        return iterate()

    async def get_final_response(self) -> Any:
        return self.response


class ScriptedStreamingResponses:
    def __init__(self, responses: list[Any], events: list[list[Any]] | None = None) -> None:
        self.responses = responses
        self.events = events or [[] for _ in responses]
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeResponseStream:
        index = len(self.requests)
        self.requests.append(kwargs)
        return FakeResponseStream(self.events[index], self.responses[index])


def web_call(urls: list[str] | None = None, *, action: Any | None = None) -> Any:
    resolved_action = action or SimpleNamespace(
        type="search",
        sources=[SimpleNamespace(type="url", url=url) for url in (urls or [])],
        parsed_query={"sdk_only": True},
    )
    return SimpleNamespace(
        type="web_search_call",
        id="web-call",
        status="completed",
        action=resolved_action,
        sdk_metadata={"drop": True},
    )


def cited_message(*citations: tuple[str, str]) -> Any:
    return SimpleNamespace(
        type="message",
        id="message-with-citations",
        role="assistant",
        status="completed",
        content=[
            SimpleNamespace(
                type="output_text",
                text="Sourced Destiny answer.",
                annotations=[
                    SimpleNamespace(
                        type="url_citation",
                        url=url,
                        title=title,
                        start_index=0,
                        end_index=8,
                    )
                    for url, title in citations
                ],
            )
        ],
    )


def assistant_message(text: str) -> Any:
    return SimpleNamespace(
        type="message",
        id="assistant-message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
        parsed_content={"sdk_only": True},
    )


def reasoning_item() -> Any:
    return SimpleNamespace(
        type="reasoning",
        id="reasoning-1",
        summary=[SimpleNamespace(type="summary_text", text="")],
        content=[],
        encrypted_content="encrypted-reasoning",
        status="completed",
        parsed_summary={"sdk_only": True},
    )


def test_replay_normalizer_uses_strict_supported_wire_shapes() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    output = [
        reasoning_item(),
        assistant_message("I will check your Titan."),
        web_call(["https://www.bungie.net/7/en/News/article/guide"]),
        SimpleNamespace(
            type="function_call",
            call_id="call-1",
            name="get_character_summary",
            arguments='{"character_id":null}',
            parsed_arguments={"character_id": None},
            status="completed",
        ),
    ]

    normalized = service._normalize_replay_output_items(output)

    assert [item["type"] for item in normalized] == [
        "reasoning",
        "message",
        "web_search_call",
        "function_call",
    ]
    assert normalized[0] == {
        "type": "reasoning",
        "id": "reasoning-1",
        "summary": [{"type": "summary_text", "text": ""}],
        "encrypted_content": "encrypted-reasoning",
        "content": [],
        "status": "completed",
    }
    assert normalized[1]["content"] == [
        {"type": "output_text", "text": "I will check your Titan.", "annotations": []}
    ]
    assert normalized[2]["action"] == {
        "type": "search",
        "sources": [{"type": "url", "url": "https://www.bungie.net/7/en/News/article/guide"}],
    }
    assert normalized[3] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "get_character_summary",
        "arguments": '{"character_id":null}',
    }
    serialized = json.dumps(normalized)
    assert "parsed_arguments" not in serialized
    assert "parsed_content" not in serialized
    assert "parsed_query" not in serialized
    assert "parsed_summary" not in serialized
    assert "sdk_metadata" not in serialized


def test_replay_normalizer_rejects_unknown_required_output_type() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))

    with pytest.raises(RuntimeError, match="Unsupported Responses output item type"):
        service._normalize_replay_output_items(
            [SimpleNamespace(type="future_required_output", sdk_only=True)]
        )


def test_streaming_guardian_fact_preserves_multi_round_function_call_loop() -> None:
    first = FakeResponses.response(
        [
            SimpleNamespace(
                type="function_call",
                name="get_equipped_loadout",
                arguments='{"character_id":"char-1"}',
                call_id="loadout-1",
                parsed_arguments={"character_id": "char-1"},
            )
        ]
    )
    final = FakeResponses.response([assistant_message("Your Titan has Sunshot equipped.")])
    streamed = ScriptedStreamingResponses(
        [first, final],
        events=[
            [SimpleNamespace(type="response.function_call_arguments.done")],
            [SimpleNamespace(type="response.output_text.delta", delta="ignored")],
        ],
    )
    statuses: list[Any] = []

    async def emit(event: Any) -> None:
        statuses.append(event)

    reporter = StreamStatusReporter(emit)
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=streamed)

    response = asyncio.run(
        service.chat(
            ChatRequest(message="What weapons do I have equipped?"),
            guardian_context(),
            stream_status=reporter,
        )
    )

    assert response.message == "Your Titan has Sunshot equipped."
    assert [event.stage for event in statuses] == ["guardian", "final"]
    assert len(streamed.requests) == 2
    assert "previous_response_id" not in streamed.requests[1]
    tool_output = next(
        value
        for value in streamed.requests[1]["input"]
        if isinstance(value, dict) and value.get("type") == "function_call_output"
    )
    assert tool_output["call_id"] == "loadout-1"
    replayed_call = next(
        value
        for value in streamed.requests[1]["input"]
        if isinstance(value, dict) and value.get("type") == "function_call"
    )
    assert "parsed_arguments" not in replayed_call
    assert service.latest_trace()["continuation_items_normalized"] is True


def test_streaming_multiple_tool_rounds_preserve_normalized_correlation_order() -> None:
    first = FakeResponses.response(
        [
            SimpleNamespace(
                type="function_call",
                name="get_character_summary",
                arguments='{"character_id":"char-1"}',
                call_id="summary-1",
                parsed_arguments={"character_id": "char-1"},
            )
        ]
    )
    second = FakeResponses.response(
        [
            reasoning_item(),
            SimpleNamespace(
                type="function_call",
                name="get_active_quests",
                arguments='{"character_id":"char-1"}',
                call_id="quests-2",
                parsed_arguments={"character_id": "char-1"},
            ),
        ]
    )
    final = FakeResponses.response([assistant_message("Your Titan is ready for its next quest.")])
    streamed = ScriptedStreamingResponses([first, second, final])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=streamed)

    response = asyncio.run(
        service.chat(
            ChatRequest(message="What quests are active on my Titan?"),
            guardian_context(),
            stream_status=StreamStatusReporter(lambda _event: asyncio.sleep(0)),
        )
    )

    assert response.message == "Your Titan is ready for its next quest."
    assert len(streamed.requests) == 3
    continuation = streamed.requests[2]["input"]
    assert [value.get("type") or f"{value.get('role')}_message" for value in continuation] == [
        "user_message",
        "function_call",
        "function_call_output",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert [
        value["call_id"] for value in continuation if value.get("type") == "function_call_output"
    ] == ["summary-1", "quests-2"]
    assert "parsed_arguments" not in json.dumps(continuation)


def test_streaming_guardian_web_comparison_preserves_sources_and_status_order() -> None:
    source_url = "https://www.bungie.net/7/en/News/article/campaigns"
    first = FakeResponses.response(
        [
            web_call([source_url]),
            SimpleNamespace(
                type="function_call",
                name="get_character_summary",
                arguments='{"character_id":"char-1"}',
                call_id="guardian-1",
            ),
        ]
    )
    final = FakeResponses.response([cited_message((source_url, "Campaign details"))])
    streamed = ScriptedStreamingResponses(
        [first, final],
        events=[
            [SimpleNamespace(type="response.web_search_call.searching")],
            [SimpleNamespace(type="response.output_text.done")],
        ],
    )
    statuses: list[Any] = []

    async def emit(event: Any) -> None:
        statuses.append(event)

    async def run() -> Any:
        reporter = StreamStatusReporter(emit)
        await reporter.emit("guardian")
        service = RecommendationService(Settings(openai_api_key="test-key"))
        service.client = SimpleNamespace(responses=streamed)
        response = await service.chat(
            ChatRequest(message="Given my Titan, Final Shape or Edge?"),
            guardian_context(),
            stream_status=reporter,
        )
        service.mark_stream_delivery(
            reporter.trace_id,
            final_response_streamed=True,
            stream_completed=True,
            emitted_status_categories=list(reporter.emitted_categories),
        )
        return response, service.latest_trace()

    response, trace = asyncio.run(run())

    assert [event.stage for event in statuses] == [
        "guardian",
        "web",
        "comparison",
        "final",
    ]
    assert response.sources[0].url == source_url
    assert trace["streaming"] is True
    assert trace["final_response_streamed"] is True
    assert trace["stream_completed"] is True
    assert trace["emitted_status_categories"] == [
        "guardian",
        "web",
        "comparison",
        "final",
    ]
    serialized_trace = json.dumps(trace)
    assert "Sourced Destiny answer" not in serialized_trace
    assert '"character_id"' not in serialized_trace
    assert '"arguments"' not in serialized_trace


def test_streaming_content_progression_comparison_uses_safe_campaign_status() -> None:
    first = FakeResponses.response(
        [
            SimpleNamespace(
                type="function_call",
                name="get_content_progression",
                arguments='{"character_id":"char-1","content_name":"The Final Shape"}',
                call_id="progress-1",
            )
        ]
    )
    final = FakeResponses.response(
        [assistant_message("Bungie's data doesn't confirm whether your Titan finished it.")]
    )
    streamed = ScriptedStreamingResponses([first, final])
    statuses: list[Any] = []

    async def emit(event: Any) -> None:
        statuses.append(event)

    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=streamed)
    response = asyncio.run(
        service.chat(
            ChatRequest(message="Given my Titan, Final Shape or Edge?"),
            guardian_context(),
            stream_status=StreamStatusReporter(emit),
        )
    )

    assert response.message.startswith("Bungie's data doesn't confirm")
    assert statuses[0].stage == "guardian"
    assert statuses[0].label == "Checking campaign progress…"
    assert [event.stage for event in statuses[1:]] == ["comparison", "final"]


def test_streaming_correction_never_emits_rejected_draft() -> None:
    bad = "Based on the available data, run Nightfall next."
    corrected = "Run Nightfall next because it fits your current goal and offers useful rewards."
    streamed = ScriptedStreamingResponses(
        [
            FakeResponses.response([assistant_message(bad)]),
            FakeResponses.response([assistant_message(corrected)]),
        ]
    )
    emitted: list[Any] = []

    async def emit(event: Any) -> None:
        emitted.append(event)

    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=streamed)
    response = asyncio.run(
        service.chat(
            ChatRequest(message="What should I do next?"),
            guardian_context(),
            stream_status=StreamStatusReporter(emit),
        )
    )
    client_events = response_events(response, "recommendation")
    streamed_text = "".join(
        event.delta for event in client_events if event.event == "message_delta"
    )

    assert streamed_text == corrected
    assert bad not in streamed_text
    assert len(streamed.requests) == 2
    assert streamed.requests[1]["tool_choice"] == "none"
    replayed_message = next(
        value
        for value in streamed.requests[1]["input"]
        if isinstance(value, dict) and value.get("type") == "message"
    )
    assert replayed_message["role"] == "assistant"
    assert "parsed_content" not in replayed_message
    assert service.latest_trace()["planning_correction"]["succeeded"] is True


def test_parallel_streaming_guardian_tools_emit_one_guardian_status() -> None:
    calls = [
        SimpleNamespace(
            type="function_call",
            name=name,
            arguments='{"character_id":null}',
            call_id=f"call-{index}",
            parsed_arguments={"character_id": None},
        )
        for index, name in enumerate(
            ("get_character_summary", "get_active_quests", "get_progression"), start=1
        )
    ]
    streamed = ScriptedStreamingResponses(
        [
            FakeResponses.response([reasoning_item(), *calls]),
            FakeResponses.response(
                [
                    assistant_message(
                        "Run Nightfall because it fits your goal and offers a useful reward."
                    )
                ]
            ),
        ]
    )
    statuses: list[Any] = []

    async def emit(event: Any) -> None:
        statuses.append(event)

    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=streamed)
    asyncio.run(
        service.chat(
            ChatRequest(message="What should I do next?"),
            guardian_context(),
            stream_status=StreamStatusReporter(emit),
        )
    )

    assert [event.stage for event in statuses].count("guardian") == 1
    assert [event.stage for event in statuses] == ["guardian", "comparison", "final"]
    continuation = streamed.requests[1]["input"]
    assert [value.get("type") or f"{value.get('role')}_message" for value in continuation] == [
        "user_message",
        "reasoning",
        "function_call",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "function_call_output",
    ]
    assert [
        value["call_id"] for value in continuation if value.get("type") == "function_call_output"
    ] == ["call-1", "call-2", "call-3"]
    assert "parsed_arguments" not in json.dumps(continuation)


def test_guardian_only_question_does_not_require_web_research() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI()
    service.client = fake

    response = asyncio.run(
        service.chat(ChatRequest(message="What subclass is my Titan using?"), guardian_context())
    )

    assert response.sources == []
    assert {tool["type"] for tool in fake.responses.requests[0]["tools"]} == {"function"}
    assert service.latest_trace()["web_research"] == {
        "occurred": False,
        "source_count": 0,
    }


@pytest.mark.parametrize(
    "question",
    [
        "Give me a broad guide to Destiny 2's campaigns and DLC story order.",
        "What changed in the latest Destiny 2 patch?",
    ],
)
def test_broad_and_patch_sensitive_questions_can_use_web_research(question: str) -> None:
    output = [
        web_call(["https://www.bungie.net/7/en/News/article/example-patch"]),
        cited_message(
            ("https://www.bungie.net/7/en/News/article/example-patch", "Destiny 2 Update")
        ),
    ]
    scripted = ScriptedWebResponses([output], ["Sourced Destiny answer."])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(service.chat(ChatRequest(message=question), guardian_context()))

    assert response.sources[0].title == "Destiny 2 Update"
    assert response.sources[0].domain == "www.bungie.net"
    assert scripted.requests[0]["include"] == [
        "web_search_call.action.sources",
        "reasoning.encrypted_content",
    ]
    assert service.latest_trace()["web_research"] == {
        "occurred": True,
        "source_count": 1,
    }


def test_guardian_account_facts_are_reserved_for_guardian_tools() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI()
    service.client = fake

    asyncio.run(service.chat(ChatRequest(message="Do I own Gjallarhorn?"), guardian_context()))

    instructions = fake.responses.requests[0]["instructions"]
    assert "MUST come from Guardian tool" in instructions
    assert "must never be treated as evidence" in instructions
    assert "Never put private Guardian facts" in instructions
    assert service.latest_trace()["grounding"]["guardian_account"] is True
    assert "user" not in service.latest_trace()


def test_returned_web_sources_are_validated_deduplicated_and_bounded() -> None:
    valid_urls = [f"https://example{i}.com/guide" for i in range(10)]
    output = [
        web_call(valid_urls + ["javascript:alert(1)", "https://user:pass@example.com/private"]),
        cited_message((valid_urls[0], "Primary guide"), ("not-a-url", "Invalid")),
    ]
    scripted = ScriptedWebResponses([output], ["Sourced Destiny answer."])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(ChatRequest(message="Find a missing Destiny guide."), guardian_context())
    )

    assert len(response.sources) == 8
    assert response.sources[0].title == "Primary guide"
    assert len({source.url for source in response.sources}) == 8
    assert all(source.url.startswith("https://example") for source in response.sources)
    assert service.latest_trace()["web_research"]["source_count"] == 8


def test_equivalent_source_urls_are_deduplicated_with_the_useful_title() -> None:
    output = [
        web_call(
            [
                "https://www.example.com/guide/?utm_source=search#section",
                "http://example.com/guide",
                "https://different.example/article",
            ]
        ),
        cited_message(("https://example.com/guide", "Useful campaign guide")),
    ]
    scripted = ScriptedWebResponses([output], ["Sourced answer."])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(ChatRequest(message="Find a missing Destiny guide."), guardian_context())
    )

    assert len(response.sources) == 2
    assert response.sources[0].title == "Useful campaign guide"
    assert response.sources[0].domain == "example.com"


def test_normal_fact_research_keeps_sources_compact() -> None:
    urls = [f"https://example{index}.com/patch" for index in range(7)]
    output = [web_call(urls), cited_message((urls[0], "Patch notes"))]
    scripted = ScriptedWebResponses([output], ["The patch changed weapon tuning."])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(ChatRequest(message="What changed in the latest patch?"), guardian_context())
    )

    assert len(response.sources) == 4
    assert response.sources[0].title == "Patch notes"


@pytest.mark.parametrize(
    "call",
    [
        web_call(action=SimpleNamespace(type="search", sources=None)),
        web_call(action=SimpleNamespace(type="search")),
        SimpleNamespace(type="web_search_call", action=None),
    ],
)
def test_malformed_or_absent_web_source_metadata_fails_safely(call: Any) -> None:
    scripted = ScriptedWebResponses(
        [[call, cited_message(("javascript:alert(1)", "Unsafe"))]],
        ["Answer without usable sources."],
    )
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(ChatRequest(message="Research this Destiny topic."), guardian_context())
    )

    assert response.message == "Answer without usable sources."
    assert response.sources == []
    assert service.latest_trace()["web_research"]["occurred"] is True


def test_hosted_web_output_and_custom_function_call_continue_statelessly() -> None:
    first_output = [
        web_call(["https://www.bungie.net/7/en/News/article/guide"]),
        SimpleNamespace(
            type="function_call",
            name="get_character_summary",
            arguments='{"character_id":null}',
            call_id="call-with-web",
        ),
    ]
    final_output = [
        cited_message(("https://www.bungie.net/7/en/News/article/guide", "Bungie guide"))
    ]
    scripted = ScriptedWebResponses(
        [first_output, final_output],
        ["", "Grounded personalized answer."],
    )
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(ChatRequest(message="What should I do next?"), guardian_context())
    )

    assert response.message == "Grounded personalized answer."
    assert len(scripted.requests) == 2
    second_input = scripted.requests[1]["input"]
    assert [value.get("type") or f"{value.get('role')}_message" for value in second_input] == [
        "user_message",
        "web_search_call",
        "function_call",
        "function_call_output",
    ]
    assert all(isinstance(value, dict) for value in second_input)
    assert "sdk_metadata" not in json.dumps(second_input)
    assert response.sources[0].url == "https://www.bungie.net/7/en/News/article/guide"


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("What should I do next?", "general"),
        ("I have 30 minutes. What should I play?", "time_limited"),
        ("I want better gear.", "gear"),
        ("I want something chill.", "casual"),
    ],
)
def test_session_planning_intent_is_classified_without_storing_prompt(
    message: str, intent: str
) -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI(request_tool=False)
    service.client = fake

    asyncio.run(service.chat(ChatRequest(message=message), guardian_context()))

    trace = service.latest_trace()
    assert trace["session_planning"] is True
    assert trace["intent_category"] == intent
    assert trace["number_of_guardian_tools_used"] == 0
    assert trace["knowledge_categories_used"] == []
    assert trace["web_research_occurred"] is False
    assert "user" not in trace
    instructions = fake.responses.requests[0]["instructions"]
    assert "handle this turn as session planning" in instructions
    assert f"safe intent category is {intent}" in instructions
    assert "Application-generated session planning context" in instructions
    assert '"evidence_semantics"' in instructions
    assert "membership-secret-shape" not in instructions
    assert "char-1" not in instructions


def test_session_planning_uses_retained_preferences_for_shorthand_followup() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI(request_tool=False)
    service.client = fake
    request = ChatRequest(
        message="Actually something chill.",
        history=[
            ChatTurn(
                role="user",
                content="I have an hour and want story on my Titan.",
            ),
            ChatTurn(role="assistant", content="I can help with that."),
        ],
    )

    asyncio.run(service.chat(request, guardian_context()))

    trace = service.latest_trace()
    assert trace["session_planning"] is True
    assert trace["intent_category"] == "casual"
    assert trace["planning_context"]["requested_character_scope"] == "Titan"
    assert trace["planning_context"]["current_preference_updates"] == [
        "primary_goal",
        "intensity",
    ]
    instructions = fake.responses.requests[0]["instructions"]
    assert '"character":"Titan"' in instructions
    assert '"time_minutes":60' in instructions
    assert '"primary_goal":"casual_chill"' in instructions
    assert '"intensity":"chill"' in instructions


def test_current_turn_preferences_override_history_in_planning_instructions() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI(request_tool=False)
    service.client = fake

    asyncio.run(
        service.chat(
            ChatRequest(
                message="Give me a 20-minute gear run on Hunter.",
                history=[
                    ChatTurn(
                        role="user",
                        content="I have an hour and want story on my Titan.",
                    )
                ],
            ),
            guardian_context(),
        )
    )

    trace = service.latest_trace()
    assert trace["intent_category"] == "gear"
    assert trace["planning_context"]["requested_character_scope"] == "Hunter"
    instructions = fake.responses.requests[0]["instructions"]
    assert '"character":"Hunter"' in instructions
    assert '"time_minutes":20' in instructions
    assert '"primary_goal":"gear_rewards"' in instructions
    assert '"character":"Titan"' not in instructions
    assert '"time_minutes":60' not in instructions


def test_cleared_goal_is_not_reintroduced_by_current_turn_words() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI(request_tool=False)
    service.client = fake

    asyncio.run(
        service.chat(
            ChatRequest(
                message="Forget the story preference.",
                history=[
                    ChatTurn(
                        role="user",
                        content="I have an hour and want story on my Titan.",
                    )
                ],
            ),
            guardian_context(),
        )
    )

    trace = service.latest_trace()
    assert trace["intent_category"] == "time_limited"
    instructions = fake.responses.requests[0]["instructions"]
    assert '"primary_goal":"unspecified"' in instructions
    assert '"time_minutes":60' in instructions


def test_planning_correction_blocks_reasking_known_preferences() -> None:
    repeated_questions = [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text="Which character? How much time do you have? Story or loot?",
                )
            ],
        )
    ]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(repeated_questions, request_tool=False)

    response = asyncio.run(
        service.chat(
            ChatRequest(
                message="What should I do?",
                history=[
                    ChatTurn(
                        role="user",
                        content="I have 45 minutes and want story on my Titan.",
                    )
                ],
            ),
            guardian_context(),
        )
    )

    assert "Which character" not in response.message
    assert "How much time" not in response.message
    assert "Story or loot" not in response.message
    assert "Titan" in response.message
    trace = service.latest_trace()
    assert trace["planning_correction"]["attempted"] is True
    assert {
        "reasked_known_character",
        "reasked_known_time",
        "reasked_known_goal",
        "too_many_followups",
    } <= set(trace["planning_correction"]["violation_codes"])


def test_broad_session_planning_can_gather_multiple_relevant_guardian_slices() -> None:
    calls = [
        SimpleNamespace(
            type="function_call",
            name=name,
            arguments=arguments,
            call_id=f"planning-{index}",
        )
        for index, (name, arguments) in enumerate(
            [
                ("get_character_summary", '{"character_id":null}'),
                ("get_active_quests", '{"character_id":null}'),
                ("get_progression", '{"character_id":null}'),
                ("get_available_activities", '{"character_id":null}'),
            ]
        )
    ]
    scripted = ScriptedWebResponses(
        [calls, [SimpleNamespace(type="message", content=[])]],
        ["", "Play the campaign next; it is the strongest meaningful progression option."],
    )
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(ChatRequest(message="What should I do next?"), guardian_context())
    )

    assert response.message.startswith("Play the campaign next")
    trace = service.latest_trace()
    assert trace["session_planning"] is True
    assert trace["number_of_guardian_tools_used"] == 4
    assert trace["knowledge_categories_used"] == ["guardian"]
    assert trace["tools"] == [
        "get_character_summary",
        "get_active_quests",
        "get_progression",
        "get_available_activities",
    ]


def test_session_planning_instructions_prevent_shallow_recommendations() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI(request_tool=False)
    service.client = fake

    asyncio.run(
        service.chat(
            ChatRequest(message="Should I keep doing this quest or start the campaign?"),
            guardian_context(),
        )
    )

    instructions = fake.responses.requests[0]["instructions"]
    assert "high completion percentage is not enough" in instructions
    assert "active Edge of Fate quest does not by itself" in instructions
    assert "meaningful campaign, unlock, quest, or content progression" in instructions
    assert "gather enough information about BOTH" in instructions
    assert "research it instead of inventing importance" in instructions
    assert 'Avoid unsupported phrases such as "likely grants"' in instructions
    assert "ask at most ONE targeted follow-up" in instructions
    assert "Interpret a question about" in instructions
    assert '"completing The Final Shape" as the main campaign' in instructions
    assert "typical useful answer is about 80-180 words" in instructions
    assert "compare both sides briefly" in instructions
    assert "Do not dump full quest lists" in instructions
    assert "Never expose character IDs" in instructions
    assert "only that no matching active quest was returned" in instructions
    assert "does NOT establish completed, not started" in instructions
    assert "active quest only as evidence" in instructions
    assert "Keep a recommendation scoped to the character the user named" in instructions
    assert "first inspect relevant Guardian progress for the" in instructions
    assert "calling get_content_progression for BOTH lines" in instructions
    assert "significance, unlocks, prerequisites, and" in instructions
    assert "BOTH content lines" in instructions


def test_loot_recommendation_allows_useful_moderate_detail() -> None:
    answer = (
        "Run matchmade Nightfalls on your Titan first. They give you a focused route toward useful "
        "drops without requiring a raid team, and your Vanguard objective can progress alongside "
        "them. Check each reward before dismantling it, prioritizing weapon rolls and armor that "
        "improves the stats your build needs. As a backup, use another visible matchmade activity "
        "that overlaps with an active quest, but don't choose it solely because it is tracked. "
        "Are you chasing a weapon or an armor upgrade?"
    )
    scripted = ScriptedWebResponses([[assistant_message(answer)]], [answer])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(
            ChatRequest(
                message=(
                    "I want better loot specifically. I want to increase my power level on my "
                    "Titan."
                )
            ),
            guardian_context(),
        )
    )

    assert response.message == answer
    assert len(response.message.split()) > 60
    assert response.message.count("?") == 1
    assert len(scripted.requests) == 1
    assert service.latest_trace()["response_mode"] == "recommendation"


def test_constraint_followup_adapts_without_restarting_or_recommending_raids() -> None:
    answer = (
        "Then skip raids. Run matchmade Nightfalls on your Titan, with solo campaign missions as "
        "the backup when you want zero matchmaking. Both keep the route aligned with better gear "
        "without requiring a premade group. Are you chasing a weapon or armor slot?"
    )
    scripted = ScriptedWebResponses([[assistant_message(answer)]], [answer])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(
            ChatRequest(
                message=(
                    "I don't want to do raids. Solo activities or matchmaking activities "
                    "preferably."
                ),
                history=[
                    ChatTurn(
                        role="user",
                        content=(
                            "I want better loot specifically. I want to increase my power level "
                            "on my Titan."
                        ),
                    ),
                    ChatTurn(role="assistant", content="Start with endgame loot routes."),
                ],
            ),
            guardian_context(),
        )
    )

    assert response.message.startswith("Then skip raids")
    assert "Goal:" not in response.message
    assert "Can you run raids" not in response.message
    assert response.message.count("?") == 1
    instructions = scripted.requests[0]["instructions"]
    assert '"character":"Titan"' in instructions
    assert '"primary_goal":"gear_rewards"' in instructions
    assert '"fireteam":"either"' in instructions
    assert '"exclusions":["raids"]' in instructions
    assert "conversational follow-up" in instructions


def test_equipped_weapon_question_is_direct_guardian_only_and_id_free() -> None:
    answer = "On char-1: Funnelweb, Sunshot, and Nox Perennial V."
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI([assistant_message(answer)], request_tool=False)
    service.client = fake

    response = asyncio.run(
        service.chat(ChatRequest(message="What weapons do I have equipped?"), guardian_context())
    )

    assert response.message == "On Titan: Funnelweb, Sunshot, and Nox Perennial V."
    assert "char-1" not in response.message
    assert all(tool["type"] == "function" for tool in fake.responses.requests[0]["tools"])
    assert service.latest_trace()["web_research_occurred"] is False
    assert service.latest_trace()["response_mode"] == "direct_fact"
    assert (
        "typical useful answer is about 20-80 words" in fake.responses.requests[0]["instructions"]
    )


def test_unrequested_timed_itinerary_uses_existing_single_correction_retry() -> None:
    bad = "Spend 5 minutes prepping, then play for 20 minutes, then clean up for 5 minutes."
    corrected = (
        "Run one matchmade Nightfall on your Titan; it suits a short session without requiring an "
        "invented minute-by-minute schedule."
    )
    scripted = ScriptedWebResponses(
        [[assistant_message(bad)], [assistant_message(corrected)]], [bad, corrected]
    )
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(
            ChatRequest(message="I have 30 minutes, what should I do on my Titan?"),
            guardian_context(),
        )
    )

    assert response.message == corrected
    assert len(scripted.requests) == 2
    trace = service.latest_trace()
    assert trace["planning_correction"]["violation_codes"] == ["unrequested_timed_itinerary"]
    assert trace["planning_correction"]["succeeded"] is True


def test_comparison_correction_replaces_internal_evidence_vocabulary() -> None:
    bad = "The Final Shape completion_state is UNKNOWN, while Edge evidence_state is KNOWN_TRUE."
    corrected = (
        "Continue Edge on your Titan for now. Bungie's data doesn't confirm whether that Titan "
        "finished Final Shape, so this is a conditional choice rather than a completion claim."
    )
    scripted = ScriptedWebResponses(
        [[assistant_message(bad)], [assistant_message(corrected)]], [bad, corrected]
    )
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(
            ChatRequest(message="Given my Titan, Final Shape or Edge?"), guardian_context()
        )
    )

    assert response.message == corrected
    assert "UNKNOWN" not in response.message
    assert "evidence_state" not in response.message
    assert service.latest_trace()["response_mode"] == "comparison"
    assert "internal_jargon" in service.latest_trace()["planning_correction"]["violation_codes"]


def test_unsupported_campaign_completion_inference_is_blocked() -> None:
    unsafe_output = [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text=(
                        "Prismatic is equipped, therefore The Final Shape campaign is probably "
                        "completed."
                    ),
                )
            ],
        )
    ]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(unsafe_output, request_tool=False)

    response = asyncio.run(
        service.chat(
            ChatRequest(message="Should I do The Final Shape or The Edge of Fate next?"),
            guardian_context(),
        )
    )

    assert "probably completed" not in response.message
    assert "completion is unknown" in response.message


def test_normal_answers_hide_guardian_identifiers_but_debug_requests_can_show_them() -> None:
    identifier_output = [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text="Use char-1 on membership-secret-shape.",
                )
            ],
        )
    ]

    normal = RecommendationService(Settings(openai_api_key="test-key"))
    normal.client = FakeOpenAI(identifier_output, request_tool=False)
    normal_response = asyncio.run(
        normal.chat(ChatRequest(message="What should I do on my Titan?"), guardian_context())
    )

    debugging = RecommendationService(Settings(openai_api_key="test-key"))
    debugging.client = FakeOpenAI(identifier_output, request_tool=False)
    debug_response = asyncio.run(
        debugging.chat(
            ChatRequest(message="For debugging, show my character ID."), guardian_context()
        )
    )

    assert "char-1" not in normal_response.message
    assert "membership-secret-shape" not in normal_response.message
    assert "Titan" in normal_response.message
    assert "char-1" in debug_response.message


def test_campaign_comparison_uses_guardian_progress_and_external_research() -> None:
    first_output = [
        web_call(["https://www.bungie.net/7/en/News/article/campaigns"]),
        SimpleNamespace(
            type="function_call",
            name="get_character_summary",
            arguments='{"character_id":"char-1"}',
            call_id="campaign-character",
        ),
        SimpleNamespace(
            type="function_call",
            name="get_active_quests",
            arguments='{"character_id":"char-1"}',
            call_id="campaign-quests",
        ),
        SimpleNamespace(
            type="function_call",
            name="get_progression",
            arguments='{"character_id":"char-1"}',
            call_id="campaign-progression",
        ),
    ]
    final_output = [
        cited_message(("https://www.bungie.net/7/en/News/article/campaigns", "Campaign guide"))
    ]
    answer = (
        "Do The Final Shape next on char-1. Neither campaign's completion was established, so "
        "completion remains unknown; this recommendation uses their researched significance, "
        "not Prismatic or the active Edge quest."
    )
    scripted = ScriptedWebResponses([first_output, final_output], ["", answer])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(
            ChatRequest(
                message=(
                    "Given my Titan, should I complete The Final Shape or The Edge of Fate next?"
                )
            ),
            guardian_context(),
        )
    )

    assert response.message.startswith("Do The Final Shape next on Titan")
    assert "char-1" not in response.message
    assert "completion remains unknown" in response.message
    assert "Prismatic" in response.message
    trace = service.latest_trace()
    assert trace["session_planning"] is True
    assert trace["intent_category"] == "story"
    assert trace["number_of_guardian_tools_used"] == 3
    assert set(trace["knowledge_categories_used"]) == {"guardian", "web"}
    assert trace["web_research_occurred"] is True
    assert response.sources[0].domain == "www.bungie.net"
    instructions = scripted.requests[0]["instructions"]
    assert "Explicit character scope: Titan" in instructions
    assert (
        "Keep account inspection and the recommendation scoped to the user's Titan" in instructions
    )


def test_active_quest_absence_does_not_establish_campaign_completion() -> None:
    unsafe_output = [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text=("The Final Shape is not active on your Titan, so it must be completed."),
                )
            ],
        )
    ]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(unsafe_output, request_tool=False)

    response = asyncio.run(
        service.chat(
            ChatRequest(
                message=(
                    "Given my Titan, should I complete The Final Shape or The Edge of Fate next?"
                )
            ),
            guardian_context(),
        )
    )

    assert "must be completed" not in response.message
    assert "missing active quest data does not establish completion" in response.message
    assert "completion state remains unknown" in response.message
    assert len(service.client.responses.requests) == 2
    assert service.latest_trace()["planning_correction"] == {
        "attempted": True,
        "violation_codes": ["unknown_completion_as_known"],
        "succeeded": False,
    }


def test_active_edge_quest_cannot_force_choice_or_cross_character_scope() -> None:
    unsafe_output = [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text=(
                        "I recommend The Edge of Fate because it is active on your Titan. "
                        "The Final Shape is only active on your Hunter, so switch to Hunter as "
                        "the backup."
                    ),
                )
            ],
        )
    ]
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = FakeOpenAI(unsafe_output, request_tool=False)

    response = asyncio.run(
        service.chat(
            ChatRequest(
                message=(
                    "Given my Titan, should I complete The Final Shape or The Edge of Fate next?"
                )
            ),
            guardian_context(),
        )
    )

    assert response.message.startswith("I'd keep this comparison scoped to your Titan")
    assert "recommend The Edge of Fate" not in response.message
    assert "Hunter" not in response.message
    assert "completion state remains unknown" in response.message
    assert "The Final Shape and The Edge of Fate" in response.message
    assert len(service.client.responses.requests) == 2
    assert service.latest_trace()["planning_correction"]["attempted"] is True


def test_planning_context_encodes_campaign_absence_without_inferring_completion() -> None:
    context = guardian_context()
    context.characters[0].quests.append(
        QuestSummary(
            quest_hash=900,
            name="The Edge of Fate",
            character_id="char-1",
            objectives=[],
        )
    )
    hunter = context.characters[0].model_copy(deep=True)
    hunter.character_id = "char-2"
    hunter.class_name = "Hunter"
    hunter.quests = [
        QuestSummary(
            quest_hash=901,
            name="The Final Shape",
            character_id="char-2",
            objectives=[],
        )
    ]
    context.characters.append(hunter)

    planning = build_session_planning_context(
        context,
        "Given my Titan, should I complete The Final Shape or The Edge of Fate next?",
        "story",
        "Titan",
    )

    by_name = {value.content: value for value in planning.named_content}
    assert planning.requested_character_scope == "Titan"
    assert planning.character_class == "Titan"
    assert by_name["The Edge of Fate"].active_quest_status == "KNOWN_TRUE"
    assert by_name["The Edge of Fate"].status == "in_progress"
    assert by_name["The Edge of Fate"].evidence_state == "KNOWN_TRUE"
    assert by_name["The Edge of Fate"].completion_status == "UNKNOWN"
    assert by_name["The Final Shape"].active_quest_status == "NOT_RETURNED"
    assert by_name["The Final Shape"].status == "unknown"
    assert by_name["The Final Shape"].completion_status == "UNKNOWN"
    assert "does not establish completion" in by_name["The Final Shape"].limitations[1]
    assert all(value.external_significance_needed for value in planning.candidates[:2])
    serialized = planning.model_dump_json()
    assert "char-1" not in serialized
    assert "char-2" not in serialized
    assert "Hunter" not in serialized


def test_planning_context_classifies_near_complete_records_without_promoting_them() -> None:
    context = guardian_context()
    context.records.near_completion = [
        ObjectiveSummary(
            objective_hash=901,
            name="Long Record Cleanup",
            progress=97,
            completion_value=100,
            progress_percent=97,
        )
    ]

    planning = build_session_planning_context(
        context,
        "What should I do next?",
        "general",
        None,
    )

    record = next(value for value in planning.candidates if value.name == "Long Record Cleanup")
    assert record.classification == "record_or_collectible_cleanup"
    assert record.objectives[0].progress_percent == 97
    assert planning.candidates.index(record) > 0


@pytest.mark.parametrize(
    ("prompt", "bad_answer", "corrected_answer", "violation_code"),
    [
        (
            "Given my Titan, should I complete The Final Shape or The Edge of Fate next?",
            "The Final Shape is not active on Titan, so Edge is clearly next.",
            (
                "On your Titan, either campaign may be appropriate for story progression. "
                "Their completion states remain unknown, so compare their grounded unlocks first."
            ),
            "active_quest_only",
        ),
        (
            "What should I do next?",
            "Play Into the Light because it is already active.",
            "Play Nightfall because it fits your current goal and offers a useful reward.",
            "active_quest_only",
        ),
        (
            "Given my Titan, should I complete The Final Shape or The Edge of Fate next?",
            "Switch to your Hunter to finish The Final Shape.",
            (
                "Stay on your Titan and compare both campaigns' grounded story progression; "
                "their completion states remain unknown."
            ),
            "character_scope",
        ),
        (
            "What should I do next?",
            "Do the 97%-complete record next because it is almost complete.",
            "Start a campaign for meaningful story progression instead of routine cleanup.",
            "near_completion_only",
        ),
    ],
)
def test_bad_planning_answers_receive_one_bounded_correction(
    prompt: str,
    bad_answer: str,
    corrected_answer: str,
    violation_code: str,
) -> None:
    scripted = ScriptedWebResponses(
        [[assistant_message(bad_answer)], [assistant_message(corrected_answer)]],
        [bad_answer, corrected_answer],
    )
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(service.chat(ChatRequest(message=prompt), guardian_context()))

    assert response.message == corrected_answer
    assert len(scripted.requests) == 2
    assert scripted.requests[1]["tool_choice"] == "none"
    assert scripted.requests[1]["input"][-1]["role"] == "developer"
    trace = service.latest_trace()
    assert trace["planning_correction"]["attempted"] is True
    assert violation_code in trace["planning_correction"]["violation_codes"]
    assert trace["planning_correction"]["succeeded"] is True


def test_grounded_active_quest_recommendation_is_accepted_without_retry() -> None:
    answer = (
        "Play Into the Light next: it is active, advances your story, and unlocks a useful reward."
    )
    scripted = ScriptedWebResponses([[assistant_message(answer)]], [answer])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(
        service.chat(ChatRequest(message="What should I do next?"), guardian_context())
    )

    assert response.message == answer
    assert len(scripted.requests) == 1
    assert service.latest_trace()["planning_correction"]["attempted"] is False
