import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai import RecommendationService
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
    assert response.json() == {"message": "Your Titan is ready.", "source": "openai"}
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
