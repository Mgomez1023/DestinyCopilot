import asyncio
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

from app.ai import RecommendationService
from app.config import Settings
from app.destiny_knowledge import (
    DESTINY_KNOWLEDGE_TOOL_DEFINITIONS,
    DestinyKnowledgeService,
    ManifestKnowledgeProvider,
)
from app.models import ChatRequest
from tests.test_guardian_tools import guardian_context


def display(name: str, description: str = "") -> dict[str, Any]:
    return {"displayProperties": {"name": name, "description": description}}


def manifest_tables() -> dict[str, dict[str, Any]]:
    wish_ender = {
        **display("Wish-Ender", "A bow that sees through walls."),
        "hash": 1,
        "itemType": 3,
        "itemTypeDisplayName": "Combat Bow",
        "inventory": {"tierTypeName": "Exotic"},
        "collectibleHash": 2,
        "equippable": True,
        "damageTypeHashes": [99],
        "itemCategoryHashes": [41],
        "traitHashes": [7],
        "sockets": {"socketEntries": [{"singleInitialItemHash": 3}]},
        "perks": [{"perkHash": 4}],
        "stats": {"stats": {"5": {"statHash": 5, "value": 92}}},
        "sourceData": {"sourceHashes": [6]},
    }
    package = {
        **display("Wish-Ender", "A quest package."),
        "hash": 8,
        "itemType": 25,
        "itemTypeDisplayName": "Package",
        "inventory": {"tierTypeName": "Exotic"},
    }
    plug = {**display("Queen's Wrath", "Intrinsic bow behavior."), "hash": 3}
    quest = {
        **display("Hunter's Remembrance", "Awaken three hidden tokens."),
        "hash": 20,
        "itemType": 15,
        "itemTypeDisplayName": "Quest",
        "setData": {
            "questLineName": "Hunter's Remembrance",
            "questLineDescription": "An Awoken keepsake.",
            "itemList": [{"itemHash": 21}, {"itemHash": 22}],
        },
    }
    step_one = {
        **display("The Shattered Throne", "Enter the dungeon."),
        "hash": 21,
        "itemType": 12,
        "itemTypeDisplayName": "Quest Step",
        "objectives": {"objectiveHashes": [15], "displayActivityHashes": [10]},
    }
    step_two = {
        **display("Three Tokens", "Awaken the tokens."),
        "hash": 22,
        "itemType": 12,
        "itemTypeDisplayName": "Quest Step",
        "objectives": {"objectiveHashes": [16], "displayActivityHashes": [10]},
    }
    no_source = {
        **display("Unknown Relic", "An item with no acquisition metadata."),
        "hash": 30,
        "itemType": 3,
        "itemTypeDisplayName": "Scout Rifle",
    }
    activity = {
        **display("The Shattered Throne", "Explore Eleusinia."),
        "hash": 10,
        "activityTypeHash": 11,
        "destinationHash": 12,
        "placeHash": 13,
        "tier": 4,
        "activityLightLevel": 1900,
        "matchmaking": {"isMatchmade": False, "minParty": 1, "maxParty": 3},
        "modifiers": [{"activityModifierHash": 14}],
        "challenges": [{"objectiveHash": 15}],
        "rewards": [{"rewardText": "Dungeon rewards", "rewardItems": [{"itemHash": 1}]}],
    }
    return {
        "DestinyInventoryItemDefinition": {
            "1": wish_ender,
            "3": plug,
            "8": package,
            "20": quest,
            "21": step_one,
            "22": step_two,
            "30": no_source,
        },
        "DestinyActivityDefinition": {"10": activity},
        "DestinyActivityTypeDefinition": {"11": {**display("Dungeon"), "hash": 11}},
        "DestinyDestinationDefinition": {"12": {**display("Dreaming City"), "hash": 12}},
        "DestinyPlaceDefinition": {"13": {**display("The Reef"), "hash": 13}},
        "DestinyObjectiveDefinition": {
            "15": {
                **display("Enter the dungeon", "Find the hidden entrance."),
                "hash": 15,
                "completionValue": 1,
            },
            "16": {**display("Awaken the tokens", "Defeat the hidden foes."), "hash": 16},
        },
        "DestinyCollectibleDefinition": {
            "2": {
                **display("Wish-Ender"),
                "hash": 2,
                "sourceString": "Complete the Wish-Ender quest.",
                "itemHash": 1,
            }
        },
        "DestinySandboxPerkDefinition": {
            "4": {**display("Broadhead", "Piercing arrowhead."), "hash": 4}
        },
        "DestinyStatDefinition": {"5": {**display("Impact"), "hash": 5}},
        "DestinyTraitDefinition": {"7": {**display("Exotic"), "hash": 7}},
        "DestinyActivityModifierDefinition": {
            "14": {**display("Extinguish", "Return to orbit on a wipe."), "hash": 14}
        },
        "DestinyRecordDefinition": {
            "40": {
                **display("Oathkeeper", "Claim the Exotic bow."),
                "hash": 40,
                "recordTypeName": "Triumph",
                "rewardItems": [{"itemHash": 1, "quantity": 1}],
            }
        },
        "DestinyDamageTypeDefinition": {"99": {**display("Solar", "Solar damage."), "hash": 99}},
        "DestinyItemCategoryDefinition": {
            "41": {**display("Weapon", "A weapon category."), "hash": 41}
        },
        "DestinyRewardSourceDefinition": {
            "6": {**display("Exotic Quest", "Granted by an Exotic quest."), "hash": 6}
        },
    }


class FakeResolver:
    def __init__(self) -> None:
        self.tables = manifest_tables()

    async def manifest_version(self) -> str:
        return "knowledge-tests-v1"

    async def load_table(self, entity_type: str) -> dict[str, Any]:
        return self.tables.get(entity_type, {})

    async def iter_definitions(self, entity_type: str):
        for value in self.tables.get(entity_type, {}).values():
            yield value

    def release_table(self, _entity_type: str) -> None:
        pass

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        table = self.tables.get(entity_type, {})
        return {
            entity_hash: table[str(entity_hash)]
            for entity_hash in entity_hashes
            if str(entity_hash) in table
        }


def knowledge_service(tmp_path: Any) -> DestinyKnowledgeService:
    settings = Settings(manifest_cache_dir=tmp_path)
    provider = ManifestKnowledgeProvider(FakeResolver(), settings)  # type: ignore[arg-type]
    return DestinyKnowledgeService([provider])


def test_exact_item_search_prefers_real_inventory_item(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(
        service.execute(
            "search_destiny_entities",
            {"query": "Wish-Ender", "entity_types": ["item"], "limit": 10},
        )
    )
    assert result["results"][0]["name"] == "Wish-Ender"
    assert result["results"][0]["hash"] == 1
    details = asyncio.run(service.execute("get_item_details", {"item_name_or_hash": "Wish-Ender"}))
    assert details["item"]["hash"] == 1


def test_fuzzy_entity_search(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(
        service.execute(
            "search_destiny_entities",
            {"query": "Shatered Throne", "entity_types": ["activity"], "limit": 5},
        )
    )
    assert result["results"][0]["name"] == "The Shattered Throne"


def test_unknown_entity_is_graceful(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(
        service.execute("get_item_details", {"item_name_or_hash": "Definitely Not Real"})
    )
    assert result["found"] is False
    assert result["limitations"]


def test_item_details_resolve_perks_stats_and_collectible(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(service.execute("get_item_details", {"item_name_or_hash": "1"}))
    assert result["item"]["subtype"] == "Combat Bow"
    assert result["item"]["intrinsic_perks"][0]["name"] == "Broadhead"
    assert result["item"]["base_definition_stats"][0] == {"name": "Impact", "value": 92}
    assert result["item"]["collectible"]["source_string"]
    assert result["item"]["damage_types"][0]["name"] == "Solar"
    assert result["item"]["item_categories"][0]["name"] == "Weapon"


def test_activity_details_resolve_location_and_possible_metadata(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(
        service.execute("get_activity_details", {"activity_name_or_hash": "The Shattered Throne"})
    )
    assert result["activity"]["activity_type"]["name"] == "Dungeon"
    assert result["activity"]["destination"]["name"] == "Dreaming City"
    assert result["activity"]["possible_modifiers"][0]["name"] == "Extinguish"


def test_quest_details_resolve_steps_objectives_and_activity(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(
        service.execute("get_quest_details", {"quest_name_or_hash": "Hunter's Remembrance"})
    )
    assert result["quest"]["name"] == "Hunter's Remembrance"
    assert len(result["quest"]["steps"]) == 2
    assert result["quest"]["steps"][0]["objectives"][0]["activity_hash"] == 10
    assert result["quest"]["steps"][0]["objectives"][0]["completion_value"] == 1
    assert result["quest"]["related_activities"][0]["name"] == "The Shattered Throne"
    assert result["quest"]["related_destinations"][0]["name"] == "Dreaming City"


def test_item_source_returns_manifest_evidence(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(service.execute("find_item_source", {"item_name_or_hash": "Wish-Ender"}))
    assert result["source_known"] is True
    assert {value["kind"] for value in result["evidence"]} >= {
        "collectible_source_string",
        "heuristic_reward_source",
        "record_reward",
    }


def test_item_source_reports_missing_source(tmp_path: Any) -> None:
    service = knowledge_service(tmp_path)
    result = asyncio.run(
        service.execute("find_item_source", {"item_name_or_hash": "Unknown Relic"})
    )
    assert result["source_known"] is False
    assert "no reliable acquisition source" in result["limitations"][0]


class FakeKnowledge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return DESTINY_KNOWLEDGE_TOOL_DEFINITIONS

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"found": True, "activity": {"name": "The Shattered Throne"}}


class ToolChoosingResponses:
    def __init__(self, combined: bool = False) -> None:
        self.combined = combined
        self.requests: list[dict[str, Any]] = []

    @staticmethod
    def response(output: list[Any], output_text: str = "") -> Any:
        return SimpleNamespace(
            output=output,
            output_text=output_text,
            status="completed",
            incomplete_details=None,
            error=None,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=40,
                output_tokens_details=SimpleNamespace(reasoning_tokens=10),
            ),
            model="gpt-5-mini",
            max_output_tokens=2000,
            reasoning=SimpleNamespace(effort="low"),
            truncation="disabled",
        )

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            calls = [
                SimpleNamespace(
                    type="function_call",
                    name="get_activity_details",
                    arguments='{"activity_name_or_hash":"The Shattered Throne"}',
                    call_id="knowledge-1",
                )
            ]
            if self.combined:
                calls.insert(
                    0,
                    SimpleNamespace(
                        type="function_call",
                        name="get_active_quests",
                        arguments='{"character_id":null}',
                        call_id="guardian-1",
                    ),
                )
            return self.response(calls)
        message = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="Grounded answer.")],
        )
        return self.response([message], "Grounded answer.")


class FakeOpenAI:
    def __init__(self, combined: bool = False) -> None:
        self.responses = ToolChoosingResponses(combined)

    async def close(self) -> None:
        pass


def test_ai_uses_destiny_tool_for_general_question() -> None:
    knowledge = FakeKnowledge()
    service = RecommendationService(
        Settings(openai_api_key="test-key"),
        knowledge,  # type: ignore[arg-type]
    )
    fake = FakeOpenAI()
    service.client = fake
    response = asyncio.run(
        service.chat(ChatRequest(message="What is The Shattered Throne?"), guardian_context())
    )
    assert response.message == "Grounded answer."
    assert knowledge.calls[0][0] == "get_activity_details"
    assert any(
        value["name"] == "get_activity_details" for value in fake.responses.requests[0]["tools"]
    )


def test_ai_combines_guardian_and_destiny_tools() -> None:
    knowledge = FakeKnowledge()
    service = RecommendationService(
        Settings(openai_api_key="test-key"),
        knowledge,  # type: ignore[arg-type]
    )
    fake = FakeOpenAI(combined=True)
    service.client = fake
    response = asyncio.run(
        service.chat(
            ChatRequest(message="How do I finish my Shattered Throne quest?"),
            guardian_context(),
        )
    )
    assert response.message == "Grounded answer."
    tool_outputs = [
        value
        for value in fake.responses.requests[1]["input"]
        if isinstance(value, dict) and value.get("type") == "function_call_output"
    ]
    assert [value["call_id"] for value in tool_outputs] == ["guardian-1", "knowledge-1"]
    assert json.loads(tool_outputs[0]["output"])["quests"][0]["name"] == "Into the Light"
    assert json.loads(tool_outputs[1]["output"])["activity"]["name"] == ("The Shattered Throne")


class CountingIndexResolver(FakeResolver):
    def __init__(self, *, fail_version: bool = False) -> None:
        super().__init__()
        self.scans: list[str] = []
        self.fail_version = fail_version

    async def manifest_version(self) -> str:
        if self.fail_version:
            raise RuntimeError("Manifest metadata unavailable")
        return await super().manifest_version()

    async def iter_definitions(self, entity_type: str):
        self.scans.append(entity_type)
        await asyncio.sleep(0)
        async for value in super().iter_definitions(entity_type):
            yield value


def _provider(tmp_path: Any, resolver: FakeResolver | None = None) -> ManifestKnowledgeProvider:
    settings = Settings(manifest_cache_dir=tmp_path)
    return ManifestKnowledgeProvider(resolver or FakeResolver(), settings)  # type: ignore[arg-type]


def test_fuzzy_scoring_runs_in_a_worker_thread_with_unchanged_results(tmp_path: Any) -> None:
    provider = _provider(tmp_path)
    scoring_threads: list[int] = []
    original = provider._score_entries

    def recording(*args: Any) -> Any:
        scoring_threads.append(threading.get_ident())
        return original(*args)

    provider._score_entries = recording  # type: ignore[method-assign]

    async def run() -> tuple[int, dict[str, Any]]:
        result = await provider.search_entities("Wish Ender", ["item"], 5)
        return threading.get_ident(), result

    loop_thread, result = asyncio.run(run())

    assert scoring_threads and all(value != loop_thread for value in scoring_threads)
    direct = original("Wish Ender", ["item"], 5)
    assert [value["hash"] for value in result["results"]] == [
        entry.entity_hash for entry, _ in direct
    ]
    assert result["results"][0]["name"] == "Wish-Ender"


def test_slow_fuzzy_scoring_does_not_block_the_event_loop(tmp_path: Any) -> None:
    provider = _provider(tmp_path)
    original = provider._score_entries

    def slow(*args: Any) -> Any:
        time.sleep(0.3)  # Stand-in for a broad query's CPU-bound scoring.
        return original(*args)

    provider._score_entries = slow  # type: ignore[method-assign]

    async def run() -> float:
        await provider.warm_index()
        gaps: list[float] = []
        stop = asyncio.Event()

        async def heartbeat() -> None:
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        await provider.search_entities("Wish Ender", None, 5)
        stop.set()
        await beat
        return max(gaps)

    assert asyncio.run(run()) < 0.15


def test_warmup_builds_the_index_once_even_with_a_concurrent_first_search(tmp_path: Any) -> None:
    resolver = CountingIndexResolver()
    provider = _provider(tmp_path, resolver)

    async def run() -> tuple[dict[str, Any], dict[str, Any]]:
        warmup = asyncio.create_task(provider.warm_index())
        await asyncio.sleep(0)
        search = await provider.search_entities("Wish Ender", ["item"], 5)
        return await warmup, search

    warm_state, search = asyncio.run(run())

    assert warm_state["state"] == "ready"
    assert warm_state["entries"] > 0
    assert search["results"][0]["name"] == "Wish-Ender"
    assert resolver.scans.count("DestinyInventoryItemDefinition") == 1
    status = asyncio.run(provider.index_status())
    assert status["warmup"]["state"] == "ready"


def test_warmup_reuses_the_versioned_on_disk_index(tmp_path: Any) -> None:
    asyncio.run(_provider(tmp_path).warm_index())
    resolver = CountingIndexResolver()
    provider = _provider(tmp_path, resolver)

    state = asyncio.run(provider.warm_index())

    assert state["state"] == "ready"
    assert resolver.scans == []


def test_failed_warmup_is_recorded_and_search_still_builds_lazily(tmp_path: Any) -> None:
    resolver = CountingIndexResolver(fail_version=True)
    provider = _provider(tmp_path, resolver)

    state = asyncio.run(provider.warm_index())

    assert state["state"] == "failed"
    assert state["error_type"] == "RuntimeError"
    assert provider._entries is None
    resolver.fail_version = False
    result = asyncio.run(provider.search_entities("Wish Ender", ["item"], 5))
    assert result["results"][0]["name"] == "Wish-Ender"
