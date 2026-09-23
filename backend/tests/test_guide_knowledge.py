import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from app.ai import RecommendationService
from app.config import Settings
from app.destiny_knowledge import DESTINY_KNOWLEDGE_TOOL_DEFINITIONS
from app.guide_knowledge import (
    GUIDE_KNOWLEDGE_TOOL_DEFINITIONS,
    GuideKnowledgeProvider,
    MemoryGuideCache,
    _normalized,
)
from app.knowledge_models import GuideRecord
from app.models import ChatRequest, ChatTurn
from tests.test_guardian_tools import guardian_context


class FakeManifest:
    entities = {
        "wish ender": ("Wish-Ender", "item", "DestinyInventoryItemDefinition", 814876684),
        "hunter s remembrance": (
            "Hunter's Remembrance",
            "item",
            "DestinyInventoryItemDefinition",
            20,
        ),
        "the shattered throne": (
            "The Shattered Throne",
            "activity",
            "DestinyActivityDefinition",
            10,
        ),
    }

    async def search_entities(
        self, query: str, entity_types: list[str] | None, limit: int
    ) -> dict[str, Any]:
        del entity_types, limit
        value = self.entities.get(_normalized(query))
        if value is None:
            return {"results": []}
        name, entity_type, definition_type, entity_hash = value
        return {
            "results": [
                {
                    "name": name,
                    "entity_type": entity_type,
                    "definition_type": definition_type,
                    "hash": entity_hash,
                }
            ]
        }


class FakeSource:
    def __init__(self, records: list[GuideRecord], version: str = "tests-v1") -> None:
        self._records = records
        self.version = version

    async def records(self) -> list[GuideRecord]:
        return self._records


def provider() -> GuideKnowledgeProvider:
    return GuideKnowledgeProvider(
        FakeManifest(),  # type: ignore[arg-type]
        Settings(),
        cache=MemoryGuideCache(),
        now=lambda: datetime(2026, 9, 12, tzinfo=UTC),
    )


def test_exact_guide_lookup_and_cache_behavior() -> None:
    guides = provider()
    first = asyncio.run(guides.get_guide("Wish-Ender", "exotic_acquisition"))
    second = asyncio.run(guides.get_guide("Wish-Ender", "exotic_acquisition"))

    assert first["found"] is True
    assert first["guide"]["subject"] == "Wish-Ender"
    assert first["resolved_canonical_entity"]["hash"] == 814876684
    assert first["cache"]["status"] == "miss"
    assert second["cache"]["status"] == "hit"


def test_fuzzy_misspelled_entity_lookup() -> None:
    result = asyncio.run(
        provider().search_guides(
            "How do I finish hunters rememberence?",
            None,
            "quest_walkthrough",
            5,
        )
    )

    assert result["results"][0]["subject"] == "Hunter's Remembrance"
    assert result["resolved_canonical_entity"]["name"] == "Hunter's Remembrance"
    assert result["resolved_canonical_entity"]["manifest_resolved"] is True


def test_exotic_acquisition_guide_is_practical_and_bounded() -> None:
    result = asyncio.run(provider().get_guide("How do I get wishender?", "exotic_acquisition"))

    assert len(result["guide"]["steps"]) == 4
    assert result["guide"]["steps"][0]["location"] == "Erebus, The Shattered Throne"
    assert result["guide"]["reward"]["name"] == "Wish-Ender"
    assert all("factual_claims" in value for value in result["sources"])
    assert "article_body" not in result

    search = asyncio.run(provider().search_guides("Wish-Ender", limit=1))
    assert search["results"][0]["conflicts"][0]["status"] == "resolved"


def test_quest_walkthrough_and_activity_walkthrough() -> None:
    quest = asyncio.run(provider().get_guide("Hunter's Remembrance", "quest_walkthrough"))
    activity = asyncio.run(
        provider().get_guide("Walk me through Shattered Throne", "activity_walkthrough")
    )

    assert quest["guide"]["steps"][2]["objective"] == "Third memento"
    assert [value["name"] for value in activity["guide"]["encounters"]] == [
        "Erebus",
        "The Descent",
        "Vorgeth, the Boundless Hunger",
        "Dul Incaru, the Eternal Return",
    ]


def test_encounter_mechanics_lookup() -> None:
    result = asyncio.run(
        provider().get_guide("How does the vorgeth fight work?", "encounter_mechanics")
    )

    assert result["guide"]["subject"] == "Vorgeth, the Boundless Hunger"
    encounter = result["guide"]["encounters"][0]
    assert encounter["mechanics"]
    assert encounter["failure_conditions"]


def test_missing_guide_is_explicit() -> None:
    result = asyncio.run(provider().get_guide("Definitely Not A Real Destiny Thing"))

    assert result["found"] is False
    assert result["warnings"] == ["No maintained detailed guide matched this query."]


def test_conflicting_information_is_represented() -> None:
    base_provider = provider()
    record = asyncio.run(base_provider.source.records())[0].model_copy(deep=True)
    record.subject = "Conflict Test"
    record.aliases = ["conflict test"]
    record.claim_evidence = [
        {
            "claim_key": "mechanic",
            "value": "Stand on the plate",
            "source_id": "source-a",
            "authority": "editorial",
            "updated_at": "2026-01-01T00:00:00Z",
        },
        {
            "claim_key": "mechanic",
            "value": "Avoid the plate",
            "source_id": "source-b",
            "authority": "editorial",
            "updated_at": "2026-02-01T00:00:00Z",
        },
    ]
    guides = GuideKnowledgeProvider(
        FakeManifest(),  # type: ignore[arg-type]
        Settings(),
        source=FakeSource([record]),  # type: ignore[arg-type]
        cache=MemoryGuideCache(),
        now=lambda: datetime(2026, 9, 12, tzinfo=UTC),
    )

    result = asyncio.run(guides.get_guide("Conflict Test"))

    assert result["conflicts"][0]["status"] == "unresolved"
    assert "do not present" in result["warnings"][-1]


def test_stale_semi_stable_information_warns() -> None:
    base_provider = provider()
    record = asyncio.run(base_provider.source.records())[0].model_copy(deep=True)
    for source in record.sources:
        source.updated_at = datetime(2020, 1, 1, tzinfo=UTC)
        source.published_at = datetime(2020, 1, 1, tzinfo=UTC)
    guides = GuideKnowledgeProvider(
        FakeManifest(),  # type: ignore[arg-type]
        Settings(),
        source=FakeSource([record]),  # type: ignore[arg-type]
        cache=MemoryGuideCache(),
        now=lambda: datetime(2026, 9, 12, tzinfo=UTC),
    )

    result = asyncio.run(guides.get_guide("Wish-Ender", "exotic_acquisition"))

    assert any(
        "current availability should be re-verified" in value for value in result["warnings"]
    )


def test_volatile_information_bypasses_guide_cache() -> None:
    result = asyncio.run(provider().get_guide("What's the featured dungeon this week?"))

    assert result["found"] is False
    assert result["freshness"] == "volatile"
    assert result["requires_live_provider"] is True
    assert result["cache"]["status"] == "bypass"


class FakeCombinedKnowledge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return DESTINY_KNOWLEDGE_TOOL_DEFINITIONS + GUIDE_KNOWLEDGE_TOOL_DEFINITIONS

    @staticmethod
    def handles(name: str) -> bool:
        return name in {
            value["name"]
            for value in DESTINY_KNOWLEDGE_TOOL_DEFINITIONS + GUIDE_KNOWLEDGE_TOOL_DEFINITIONS
        }

    @staticmethod
    def category_for_tool(name: str) -> str | None:
        return "guide" if "guide" in name else "manifest"

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"found": True, "source": self.category_for_tool(name)}


class ScriptedResponses:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls
        self.requests: list[dict[str, Any]] = []

    @staticmethod
    def _response(output: list[Any], text: str = "") -> Any:
        return SimpleNamespace(
            output=output,
            output_text=text,
            status="completed",
            incomplete_details=None,
            error=None,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=30,
                output_tokens_details=SimpleNamespace(reasoning_tokens=5),
            ),
            model="gpt-5-mini",
            max_output_tokens=2000,
            reasoning=SimpleNamespace(effort="low"),
            truncation="disabled",
        )

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if len(self.requests) == 1 and self.calls:
            return self._response(
                [
                    SimpleNamespace(
                        type="function_call",
                        name=name,
                        arguments=json.dumps(arguments),
                        call_id=f"call-{index}",
                    )
                    for index, (name, arguments) in enumerate(self.calls)
                ]
            )
        return self._response(
            [
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="Grounded answer.")],
                )
            ],
            "Grounded answer.",
        )


class FakeOpenAI:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.responses = ScriptedResponses(calls)

    async def close(self) -> None:
        pass


def test_general_acquisition_orchestration_uses_manifest_and_guide() -> None:
    knowledge = FakeCombinedKnowledge()
    service = RecommendationService(
        Settings(openai_api_key="test-key"),
        knowledge,  # type: ignore[arg-type]
    )
    service.client = FakeOpenAI(
        [
            (
                "search_destiny_entities",
                {"query": "Wish-Ender", "entity_types": ["item"], "limit": 5},
            ),
            (
                "get_destiny_guide",
                {"entity_or_query": "Wish-Ender", "guide_type": "exotic_acquisition"},
            ),
        ]
    )

    asyncio.run(service.chat(ChatRequest(message="How do I get Wish-Ender?"), guardian_context()))

    assert [value[0] for value in knowledge.calls] == [
        "search_destiny_entities",
        "get_destiny_guide",
    ]
    assert service.latest_trace()["grounding"] == {
        "guardian_account": False,
        "manifest": True,
        "guide_provider": True,
        "live_provider": False,
        "web_research": False,
    }
    assert service.latest_trace()["tool_trace"] == [
        {
            "name": "search_destiny_entities",
            "category": "manifest",
            "result_source": "manifest",
        },
        {
            "name": "get_destiny_guide",
            "category": "guide",
            "result_source": "guide",
        },
    ]


def test_personalized_orchestration_uses_all_grounding_categories() -> None:
    knowledge = FakeCombinedKnowledge()
    service = RecommendationService(
        Settings(openai_api_key="test-key"),
        knowledge,  # type: ignore[arg-type]
    )
    service.client = FakeOpenAI(
        [
            ("get_active_quests", {"character_id": None}),
            (
                "get_quest_details",
                {"quest_name_or_hash": "Hunter's Remembrance"},
            ),
            (
                "get_destiny_guide",
                {
                    "entity_or_query": "Hunter's Remembrance",
                    "guide_type": "quest_walkthrough",
                },
            ),
        ]
    )

    asyncio.run(
        service.chat(
            ChatRequest(message="How do I finish Hunter's Remembrance on my account?"),
            guardian_context(),
        )
    )

    assert service.latest_trace()["grounding"] == {
        "guardian_account": True,
        "manifest": True,
        "guide_provider": True,
        "live_provider": False,
        "web_research": False,
    }
    assert service.latest_trace()["tool_trace"] == [
        {"name": "get_active_quests", "category": "guardian"},
        {
            "name": "get_quest_details",
            "category": "manifest",
            "result_source": "manifest",
        },
        {
            "name": "get_destiny_guide",
            "category": "guide",
            "result_source": "guide",
        },
    ]


def test_volatile_query_cannot_be_satisfied_by_non_live_sources_or_model() -> None:
    attempted_tool_sequences = [
        [("get_available_activities", {"character_id": None})],
        [
            (
                "search_destiny_entities",
                {"query": "Warlord's Ruin", "entity_types": ["activity"], "limit": 5},
            )
        ],
        [
            (
                "get_destiny_guide",
                {"entity_or_query": "Warlord's Ruin", "guide_type": "general"},
            )
        ],
        [],
    ]
    for attempted_calls in attempted_tool_sequences:
        knowledge = FakeCombinedKnowledge()
        service = RecommendationService(
            Settings(openai_api_key="test-key"),
            knowledge,  # type: ignore[arg-type]
        )
        fake = FakeOpenAI(attempted_calls)
        service.client = fake
        context = guardian_context().model_copy(deep=True)
        context.characters[0].available_activities[0].name = "Warlord's Ruin"

        response = asyncio.run(
            service.chat(
                ChatRequest(message="What's the featured dungeon this week?"),
                context,
            )
        )

        assert response.source == "local"
        assert "don't have a live weekly-rotation source" in response.message
        assert "Warlord's Ruin" not in response.message
        assert fake.responses.requests == []
        assert knowledge.calls == []
        trace = service.latest_trace()
        assert trace["tools"] == []
        assert trace["tool_trace"] == []
        assert trace["answer"] == "unavailable_live_data"
        assert trace["live_data"]["requires_live_provider"] is True
        assert trace["live_data"]["disallowed_as_current_sources"] == [
            "guardian_account",
            "bungie_manifest",
            "guide_provider",
            "model_memory",
        ]


def test_general_warlords_ruin_question_is_not_blocked_as_live() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI([])
    service.client = fake

    response = asyncio.run(
        service.chat(ChatRequest(message="What is Warlord's Ruin?"), guardian_context())
    )

    assert response.source == "openai"
    assert fake.responses.requests


def test_guardian_activity_trace_marks_non_authoritative_rotation_scope() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))
    fake = FakeOpenAI([("get_available_activities", {"character_id": None})])
    service.client = fake

    asyncio.run(
        service.chat(
            ChatRequest(message="What activities are available to my Guardian?"),
            guardian_context(),
        )
    )

    assert service.latest_trace()["tool_trace"] == [
        {
            "name": "get_available_activities",
            "category": "guardian",
            "availability_scope": "guardian_character_activities",
            "current_rotation_authoritative": False,
        }
    ]


def test_model_cannot_answer_volatile_query_without_successful_live_tool_result() -> None:
    class FakeRegisteredLiveKnowledge(FakeCombinedKnowledge):
        @staticmethod
        def has_category(category: str) -> bool:
            return category == "live"

    knowledge = FakeRegisteredLiveKnowledge()
    service = RecommendationService(
        Settings(openai_api_key="test-key"),
        knowledge,  # type: ignore[arg-type]
    )
    fake = FakeOpenAI([])
    service.client = fake

    response = asyncio.run(
        service.chat(
            ChatRequest(message="What's the featured dungeon this week?"),
            guardian_context(),
        )
    )

    assert len(fake.responses.requests) == 1
    assert response.source == "local"
    assert "connected live source" in response.message
    assert service.latest_trace()["grounding"]["live_provider"] is False


def test_follow_up_keeps_conversation_context_for_referent() -> None:
    knowledge = FakeCombinedKnowledge()
    service = RecommendationService(
        Settings(openai_api_key="test-key"),
        knowledge,  # type: ignore[arg-type]
    )
    fake = FakeOpenAI(
        [
            (
                "get_destiny_guide",
                {
                    "entity_or_query": "Hunter's Remembrance next step",
                    "guide_type": "quest_walkthrough",
                },
            )
        ]
    )
    service.client = fake
    request = ChatRequest(
        message="What do I do after that?",
        history=[
            ChatTurn(role="user", content="How do I finish Hunter's Remembrance?"),
            ChatTurn(role="assistant", content="Next, purify the first memento in Erebus."),
        ],
    )

    asyncio.run(service.chat(request, guardian_context()))

    first_input = fake.responses.requests[0]["input"]
    messages = [value for value in first_input if isinstance(value, dict) and value.get("role")]
    assert [value["content"] for value in messages] == [
        "How do I finish Hunter's Remembrance?",
        "Next, purify the first memento in Erebus.",
        "What do I do after that?",
    ]
    assert [value[0] for value in knowledge.calls] == ["get_destiny_guide"]
