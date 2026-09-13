import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.ai import RecommendationService
from app.config import Settings
from app.live_destiny import (
    LIVE_DESTINY_TOOL_DEFINITIONS,
    BungiePublicLiveSource,
    LiveDestinyProvider,
    MemoryLiveCache,
)
from app.live_knowledge import unavailable_live_data, volatile_topic
from app.models import ChatRequest
from tests.test_guardian_tools import guardian_context


@pytest.mark.parametrize(
    ("query", "topic"),
    [
        ("What's the featured dungeon this week?", "featured_dungeon"),
        ("What's the featured raid this week?", "featured_raid"),
        ("Which dungeon is featured?", "featured_dungeon"),
        ("What is the Nightfall this week?", "weekly_nightfall"),
        ("What's the Nightfall?", "weekly_nightfall"),
        ("Where is Xur?", "xur_location"),
        ("What is in Xur's inventory?", "xur_inventory"),
        ("Show me the weekly rotation", "weekly_rotation"),
        ("What is the current vendor inventory?", "current_vendor_inventory"),
        ("Show the vendor inventory", "current_vendor_inventory"),
        ("What is the current loot rotation?", "current_loot_rotation"),
        ("What are the current modifiers?", "current_modifiers"),
        ("What should I farm this week?", "weekly_farming"),
        ("What is available today?", "daily_availability"),
        ("What is the current meta?", "current_meta"),
        ("What is the drop rate?", "drop_rate"),
        ("Is double loot active?", "double_loot"),
        ("What is the current Nightfall?", "weekly_nightfall"),
        ("What is the current exotic mission?", "current_exotic_mission"),
        ("What changed at reset?", "reset_changes"),
        ("What is worth doing this week?", "weekly_farming"),
        ("Is this activity worth doing now?", "current_destiny_state"),
    ],
)
def test_volatile_current_queries_require_live_provider(query: str, topic: str) -> None:
    assert volatile_topic(query) == topic
    result = unavailable_live_data(query)
    assert result["topic"] == topic
    assert result["freshness"] == "volatile"
    assert result["status"] == "unavailable_live_data"
    assert result["requires_live_provider"] is True
    assert result["live_data_available"] is False


@pytest.mark.parametrize(
    "query",
    [
        "What is Warlord's Ruin?",
        "Walk me through Warlord's Ruin.",
        "What activities are available to my Guardian?",
        "Who is Xur?",
        "How does a Nightfall work?",
    ],
)
def test_general_destiny_questions_are_not_misclassified_as_live(query: str) -> None:
    assert volatile_topic(query) is None


class FakeLiveSource:
    def __init__(
        self,
        *,
        source_name: str = "fake-live",
        authority: str = "official",
        rotations: dict[str, Any] | None = None,
        milestones: list[dict[str, Any]] | None = None,
        vendors: list[dict[str, Any]] | None = None,
        stale_after: str = "2026-09-15T17:00:00Z",
    ) -> None:
        self.source_name = source_name
        self.authority = authority
        self.calls = 0
        self.snapshot = {
            "source": {
                "source_id": source_name,
                "title": source_name,
                "provider": source_name,
                "source_url": f"https://example.test/{source_name}",
                "authority": authority,
                "retrieved_at": "2026-09-12T12:00:00Z",
            },
            "retrieved_at": "2026-09-12T12:00:00Z",
            "stale_after": stale_after,
            "rotations": rotations or {},
            "milestones": milestones or [],
            "vendors": vendors or [],
            "warnings": [],
        }

    async def retrieve(self, now: datetime) -> dict[str, Any]:
        self.calls += 1
        value = deepcopy(self.snapshot)
        value["retrieved_at"] = now.isoformat()
        value["source"]["retrieved_at"] = now.isoformat()
        return value


def rotation(name: str, entity_hash: int) -> dict[str, Any]:
    return {
        "explicit": True,
        "confidence": "high",
        "entries": [
            {
                "name": name,
                "hash": entity_hash,
                "effective_from": "2026-09-08T17:00:00Z",
                "effective_until": "2026-09-15T17:00:00Z",
            }
        ],
    }


def live_provider(
    *sources: FakeLiveSource,
    clock: list[datetime] | None = None,
    ttl: int = 300,
) -> LiveDestinyProvider:
    current = clock or [datetime(2026, 9, 12, 12, tzinfo=UTC)]
    return LiveDestinyProvider(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        Settings(live_cache_ttl_seconds=ttl),
        sources=list(sources),  # type: ignore[arg-type]
        cache=MemoryLiveCache(),
        now=lambda: current[0],
    )


def test_weekly_rotation_categories_are_normalized_and_current() -> None:
    source = FakeLiveSource(
        rotations={
            "dungeon": rotation("Warlord's Ruin", 1),
            "raid": rotation("Salvation's Edge", 2),
            "nightfall": rotation("Birthplace of the Vile", 3),
            "exotic_mission": rotation("Encore", 4),
        }
    )
    provider = live_provider(source)

    dungeon = asyncio.run(provider.get_weekly_rotation("dungeon"))
    general = asyncio.run(provider.get_weekly_rotation("general"))

    assert dungeon["rotations"]["dungeon"]["entries"][0]["name"] == "Warlord's Ruin"
    assert dungeon["supported_live_topics"] == ["featured_dungeon"]
    assert dungeon["effective_window"]["reset_cadence"] == "weekly"
    assert general["live_data_available"] is True
    assert "weekly_rotation" in general["supported_live_topics"]


def test_xur_status_has_current_inventory_provenance() -> None:
    source = FakeLiveSource(
        vendors=[
            {
                "vendor_hash": 2190858386,
                "name": "Xûr",
                "enabled": True,
                "locations": [{"destination_hash": 1, "destination": "Tower"}],
                "location_precision": "destination_only",
                "inventory": [{"item_hash": 2, "name": "Orpheus Rig"}],
                "next_refresh": "2026-09-15T17:00:00Z",
            }
        ]
    )

    result = asyncio.run(live_provider(source).get_vendor_status("Xur"))

    assert result["vendor"]["inventory"][0]["name"] == "Orpheus Rig"
    assert result["vendor"]["locations"][0]["destination"] == "Tower"
    assert result["effective_window"]["reset_cadence"] == "vendor"
    assert set(result["supported_live_topics"]) == {
        "current_vendor_inventory",
        "xur_inventory",
        "xur_status",
    }


def test_xur_unavailable_is_explicit() -> None:
    result = asyncio.run(live_provider(FakeLiveSource()).get_vendor_status("Xur"))

    assert result["status"] == "not_exposed_by_current_sources"
    assert result["vendor"] is None
    assert result["live_data_available"] is False


def test_live_cache_hits_then_expires_before_reset_boundary() -> None:
    clock = [datetime(2026, 9, 12, 12, tzinfo=UTC)]
    source = FakeLiveSource(stale_after="2026-09-12T12:02:00Z")
    provider = live_provider(source, clock=clock, ttl=3600)

    first = asyncio.run(provider.get_status())
    second = asyncio.run(provider.get_status())
    clock[0] += timedelta(minutes=3)
    third = asyncio.run(provider.get_status())

    assert first["cache"]["status"] == "miss"
    assert first["effective_window"]["stale_after"] == "2026-09-12T12:02:00Z"
    assert second["cache"]["status"] == "hit"
    assert third["cache"]["status"] == "miss"
    assert source.calls == 2
    assert first["reset_windows"]["daily"]["reset_cadence"] == "daily"
    assert first["reset_windows"]["weekly"]["effective_until"] == (
        "2026-09-15T17:00:00Z"
    )


def test_live_cache_failure_does_not_break_retrieval() -> None:
    class BrokenCache:
        async def get(self, key: str, now: datetime) -> dict[str, Any] | None:
            del key, now
            raise OSError("cache unavailable")

        async def set(
            self, key: str, value: dict[str, Any], stale_after: datetime
        ) -> None:
            del key, value, stale_after
            raise OSError("cache unavailable")

    source = FakeLiveSource(rotations={"dungeon": rotation("A", 1)})
    provider = LiveDestinyProvider(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        Settings(),
        sources=[source],  # type: ignore[list-item]
        cache=BrokenCache(),  # type: ignore[arg-type]
        now=lambda: datetime(2026, 9, 12, 12, tzinfo=UTC),
    )

    result = asyncio.run(provider.get_weekly_rotation("dungeon"))

    assert result["rotations"]["dungeon"]["known"] is True


def test_equally_authoritative_rotation_conflict_is_not_answerable() -> None:
    first = FakeLiveSource(source_name="first", rotations={"dungeon": rotation("A", 1)})
    second = FakeLiveSource(source_name="second", rotations={"dungeon": rotation("B", 2)})

    result = asyncio.run(live_provider(first, second).get_weekly_rotation("dungeon"))

    assert result["rotations"]["dungeon"]["known"] is False
    assert result["conflicts"][0]["status"] == "unresolved"
    assert result["live_data_available"] is False
    assert result["supported_live_topics"] == []


def test_higher_authority_rotation_resolves_conflict() -> None:
    official = FakeLiveSource(
        source_name="official", authority="official", rotations={"raid": rotation("A", 1)}
    )
    derived = FakeLiveSource(
        source_name="derived", authority="derived", rotations={"raid": rotation("B", 2)}
    )

    result = asyncio.run(live_provider(official, derived).get_weekly_rotation("raid"))

    assert result["rotations"]["raid"]["entries"][0]["name"] == "A"
    assert result["conflicts"][0]["status"] == "resolved"


def test_vendor_conflict_is_explicit_and_not_answerable() -> None:
    vendor_a = {
        "vendor_hash": 2190858386,
        "name": "Xûr",
        "enabled": True,
        "locations": [{"destination": "Tower"}],
        "inventory": [{"name": "Item A"}],
        "next_refresh": "2026-09-15T17:00:00Z",
    }
    vendor_b = {**vendor_a, "inventory": [{"name": "Item B"}]}
    first = FakeLiveSource(source_name="first", vendors=[vendor_a])
    second = FakeLiveSource(source_name="second", vendors=[vendor_b])

    result = asyncio.run(live_provider(first, second).get_vendor_status("Xur"))

    assert result["vendor"] is None
    assert result["conflicts"][0]["status"] == "unresolved"
    assert result["live_data_available"] is False


def test_public_milestone_presence_does_not_imply_featured_dungeon() -> None:
    source = FakeLiveSource(
        milestones=[
            {
                "hash": 1,
                "name": "Warlord's Ruin",
                "effective_from": "2026-09-08T17:00:00Z",
                "effective_until": "2026-09-15T17:00:00Z",
                "activities": [],
            }
        ]
    )
    provider = live_provider(source)

    activity = asyncio.run(provider.get_current_activity_status("Warlord's Ruin"))
    featured = asyncio.run(provider.get_weekly_rotation("dungeon"))

    assert activity["matches"][0]["listed_in_public_milestones"] is True
    assert activity["matches"][0]["featured"] is None
    assert featured["live_data_available"] is False
    assert "featured_dungeon" not in featured["supported_live_topics"]


def test_live_search_and_current_activity_status_are_bounded() -> None:
    source = FakeLiveSource(
        milestones=[
            {
                "hash": 10,
                "name": "Current Nightfall",
                "effective_from": "2026-09-08T17:00:00Z",
                "effective_until": "2026-09-15T17:00:00Z",
                "activities": [{"hash": 11, "name": "The Corrupted", "modifiers": []}],
            }
        ]
    )
    provider = live_provider(source)

    search = asyncio.run(provider.search_live_destiny("The Corrupted"))
    activity = asyncio.run(provider.get_current_activity_status("11"))

    assert search["results"][0]["kind"] == "public_milestone_activity"
    assert activity["matches"][0]["activities"][0]["name"] == "The Corrupted"
    assert activity["supported_live_topics"] == ["current_activity_status"]


def test_unknown_volatile_live_search_returns_explicit_unsupported_state() -> None:
    result = asyncio.run(
        live_provider(FakeLiveSource()).search_live_destiny(
            "What is the current loot rotation?"
        )
    )

    assert result["status"] == "unsupported_by_current_sources"
    assert result["live_data_available"] is False
    assert result["supported_live_topics"] == []


class FakeBungieClient:
    async def get_public_milestones(self) -> dict[str, Any]:
        return {
            "1": {
                "startDate": "2026-09-08T17:00:00Z",
                "endDate": "2026-09-15T17:00:00Z",
                "activities": [{"activityHash": 10, "modifierHashes": [20]}],
            }
        }

    async def get_public_vendors(self) -> dict[str, Any]:
        return {
            "vendors": {
                "data": {
                    str(2190858386): {
                        "enabled": True,
                        "nextRefreshDate": "2026-09-15T17:00:00Z",
                    }
                }
            },
            "sales": {
                "data": {str(2190858386): {"saleItems": {"0": {"itemHash": 30}}}}
            },
        }


class FakeResolver:
    definitions = {
        "DestinyMilestoneDefinition": {
            1: {"displayProperties": {"name": "Warlord's Ruin"}}
        },
        "DestinyActivityDefinition": {
            10: {"displayProperties": {"name": "Warlord's Ruin"}}
        },
        "DestinyActivityModifierDefinition": {
            20: {"displayProperties": {"name": "Extinguish"}}
        },
        "DestinyVendorDefinition": {
            2190858386: {
                "displayProperties": {"name": "Xûr"},
                "locations": [{"destinationHash": 40}],
            }
        },
        "DestinyInventoryItemDefinition": {
            30: {"displayProperties": {"name": "Orpheus Rig"}}
        },
        "DestinyDestinationDefinition": {
            40: {"displayProperties": {"name": "Tower"}}
        },
    }

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        table = self.definitions.get(entity_type, {})
        return {value: table[value] for value in entity_hashes if value in table}


def test_official_source_normalizes_xur_but_does_not_infer_featured() -> None:
    source = BungiePublicLiveSource(
        FakeBungieClient(),  # type: ignore[arg-type]
        FakeResolver(),  # type: ignore[arg-type]
    )

    result = asyncio.run(source.retrieve(datetime(2026, 9, 12, 12, tzinfo=UTC)))

    assert result["vendors"][0]["name"] == "Xûr"
    assert result["vendors"][0]["inventory"][0]["name"] == "Orpheus Rig"
    assert result["milestones"][0]["name"] == "Warlord's Ruin"
    assert result["rotations"] == {}


class ScriptedLiveResponses:
    def __init__(self, tool_result: dict[str, Any]) -> None:
        self.tool_result = tool_result
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="get_weekly_rotation",
                        arguments=json.dumps({"category": "dungeon"}),
                        call_id="live-call",
                    )
                ],
                output_text="",
                status="completed",
                incomplete_details=None,
                error=None,
                usage=None,
            )
        return SimpleNamespace(
            output=[SimpleNamespace(type="message", content=[])],
            output_text="Warlord's Ruin is featured this week.",
            status="completed",
            incomplete_details=None,
            error=None,
            usage=None,
        )


class FakeLiveKnowledge:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return LIVE_DESTINY_TOOL_DEFINITIONS

    @staticmethod
    def handles(name: str) -> bool:
        return name == "get_weekly_rotation"

    @staticmethod
    def category_for_tool(name: str) -> str:
        del name
        return "live"

    @staticmethod
    def has_category(category: str) -> bool:
        return category == "live"

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del name, arguments
        return self.result


def test_chat_requires_topic_specific_live_provenance() -> None:
    unsupported = {
        "source": "live_destiny",
        "freshness": "volatile",
        "live_data_available": True,
        "requires_live_provider": False,
        "supported_live_topics": ["live_status"],
    }
    service = RecommendationService(
        Settings(openai_api_key="test"),
        FakeLiveKnowledge(unsupported),  # type: ignore[arg-type]
    )
    service.client = SimpleNamespace(responses=ScriptedLiveResponses(unsupported))

    response = asyncio.run(
        service.chat(
            ChatRequest(message="What's the featured dungeon this week?"),
            guardian_context(),
        )
    )

    assert response.source == "local"
    assert "Warlord's Ruin" not in response.message
    assert service.latest_trace()["grounding"]["live_provider"] is False
    assert service.latest_trace()["required_live_topic"] == "featured_dungeon"


def test_chat_accepts_matching_current_live_provenance() -> None:
    supported = {
        "source": "live_destiny",
        "freshness": "volatile",
        "live_data_available": True,
        "requires_live_provider": False,
        "supported_live_topics": ["featured_dungeon"],
    }
    service = RecommendationService(
        Settings(openai_api_key="test"),
        FakeLiveKnowledge(supported),  # type: ignore[arg-type]
    )
    service.client = SimpleNamespace(responses=ScriptedLiveResponses(supported))

    response = asyncio.run(
        service.chat(
            ChatRequest(message="What's the featured dungeon this week?"),
            guardian_context(),
        )
    )

    assert response.source == "openai"
    assert service.latest_trace()["grounding"]["live_provider"] is True


class ScriptedCallsResponses:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            output = [
                SimpleNamespace(
                    type="function_call",
                    name=name,
                    arguments=json.dumps(arguments),
                    call_id=f"call-{index}",
                )
                for index, (name, arguments) in enumerate(self.calls)
            ]
            text = ""
        else:
            output = [SimpleNamespace(type="message", content=[])]
            text = "Grounded current recommendation."
        return SimpleNamespace(
            output=output,
            output_text=text,
            status="completed",
            incomplete_details=None,
            error=None,
            usage=None,
        )


class FakeComposedLiveKnowledge(FakeLiveKnowledge):
    @staticmethod
    def handles(name: str) -> bool:
        return name in {"search_live_destiny", "get_vendor_status"}


@pytest.mark.parametrize(
    ("message", "topic", "calls", "expected_guardian_tools"),
    [
        (
            "What should I farm this week?",
            "weekly_farming",
            [
                ("search_live_destiny", {"query": "What should I farm this week?"}),
                (
                    "search_inventory",
                    {
                        "query": None,
                        "character_id": None,
                        "item_type": None,
                        "subtype": None,
                        "bucket": None,
                        "equipped_only": False,
                        "limit": 25,
                    },
                ),
            ],
            {"search_inventory"},
        ),
        (
            "Does Xur have anything useful for my Titan?",
            "xur_inventory",
            [
                ("get_vendor_status", {"vendor": "Xur"}),
                ("get_build_details", {"character_id": "char-1"}),
            ],
            {"get_build_details"},
        ),
    ],
)
def test_live_and_account_tools_can_compose(
    message: str,
    topic: str,
    calls: list[tuple[str, dict[str, Any]]],
    expected_guardian_tools: set[str],
) -> None:
    result = {
        "source": "live_destiny",
        "freshness": "volatile",
        "live_data_available": True,
        "requires_live_provider": False,
        "supported_live_topics": [topic],
    }
    service = RecommendationService(
        Settings(openai_api_key="test"),
        FakeComposedLiveKnowledge(result),  # type: ignore[arg-type]
    )
    scripted = ScriptedCallsResponses(calls)
    service.client = SimpleNamespace(responses=scripted)

    response = asyncio.run(service.chat(ChatRequest(message=message), guardian_context()))

    assert response.source == "openai"
    trace = service.latest_trace()
    assert trace["grounding"]["live_provider"] is True
    assert trace["grounding"]["guardian_account"] is True
    assert expected_guardian_tools <= set(trace["tools"])
