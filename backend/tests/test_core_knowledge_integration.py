"""Knowledge Engine v0.1 runtime integration: provider routing, the per-turn grounded-facts
block, trace observability, and the trace-only signals."""

import asyncio
import builtins
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.ai import CORE_FACTS_HEADER, SYSTEM_INSTRUCTIONS, RecommendationService
from app.build_analysis import forced_alternatives_relevance
from app.chat_stream import StreamStatusReporter, status_for_tool
from app.config import Settings
from app.core_knowledge import DestinyKnowledgeEngine
from app.core_knowledge_provider import CoreKnowledgeProvider
from app.destiny_knowledge import (
    DESTINY_KNOWLEDGE_TOOL_DEFINITIONS,
    DestinyKnowledgeService,
    UnknownKnowledgeToolError,
)
from app.main import core_knowledge_provider
from app.models import ChatRequest
from tests.test_build_analysis import (
    ANALYZE_CALL,
    ScriptedBuildResponses,
    ScriptedStreamingBuildResponses,
    _answer,
    _call,
    _find,
    build_context,
)
from tests.test_guardian_tools import guardian_context

CORPUS = Path(__file__).parents[1] / "app" / "data" / "core_knowledge.json"
ENGINE = DestinyKnowledgeEngine.from_file(CORPUS)


class FakeManifestProvider:
    """A Manifest-category provider with a fixed result and destination vocabulary."""

    source_name = "bungie_manifest"
    knowledge_category = "manifest"
    tool_names = frozenset({"get_activity_details"})

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {"found": False}

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return [
            value
            for value in DESTINY_KNOWLEDGE_TOOL_DEFINITIONS
            if value["name"] == "get_activity_details"
        ]

    def handles(self, name: str) -> bool:
        return name in self.tool_names

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.result

    async def status(self) -> dict[str, Any]:
        return {"source": self.source_name}

    @staticmethod
    def destination_names() -> frozenset[str]:
        return frozenset({"The Moon", "Nessus", "European Dead Zone"})


def knowledge(*extra: Any) -> DestinyKnowledgeService:
    return DestinyKnowledgeService([*extra, CoreKnowledgeProvider(ENGINE)])


def service(knowledge_service: Any = None, *, streaming: bool = False, outputs: Any = None):
    recommendations = RecommendationService(
        Settings(openai_api_key="test-key", enable_debug_tools=True),
        knowledge_service if knowledge_service is not None else knowledge(),
    )
    responses_type = ScriptedStreamingBuildResponses if streaming else ScriptedBuildResponses
    responses = responses_type(outputs or [[_answer("Answer.")]])
    recommendations.client = SimpleNamespace(responses=responses)
    return recommendations, responses


def chat(message: str, outputs: Any = None, knowledge_service: Any = None, context: Any = None):
    recommendations, responses = service(knowledge_service, outputs=outputs)
    answer = asyncio.run(
        recommendations.chat(ChatRequest(message=message), context or guardian_context())
    )
    return answer, responses, recommendations.latest_trace()


def concept_call(name: str, call_id: str = "core-1") -> Any:
    return _call("get_destiny_concept", {"name": name}, call_id)


def provider_result(name: str) -> dict[str, Any]:
    return asyncio.run(knowledge().execute("get_destiny_concept", {"name": name}))


# --- Provider and routing ---------------------------------------------------------------


def test_core_provider_routes_through_the_existing_knowledge_service() -> None:
    routed = knowledge(FakeManifestProvider())

    assert routed.category_for_tool("get_destiny_concept") == "core"
    assert routed.category_for_tool("get_activity_details") == "manifest"
    assert routed.has_category("core")
    assert [value["name"] for value in routed.definitions()] == [
        "get_activity_details",
        "get_destiny_concept",
    ]
    status = asyncio.run(routed.index_status())
    assert {"source": "core_knowledge", "category": "core"}.items() <= status["providers"][
        1
    ].items()


def test_startup_loads_the_production_corpus_once_and_can_be_disabled() -> None:
    provider = core_knowledge_provider(Settings(core_knowledge_file=CORPUS))
    disabled = core_knowledge_provider(Settings(core_knowledge_enabled=False))
    broken = core_knowledge_provider(Settings(core_knowledge_file=Path("missing.json")))

    assert provider is not None and provider.engine.entity_count == 48
    assert disabled is None
    assert broken is None


def test_concept_tool_returns_grounded_card_with_source_tags() -> None:
    result = provider_result("Altars of Sorrow")

    assert result["found"] is True and result["status"] == "entity"
    assert result["matched_via"] == "canonical"
    entity = result["entity"]
    assert entity["name"] == "Altars of Sorrow" and entity["kind"] == "activity"
    assert {"predicate": "located_on", "value": "The Moon", "sources": ["manifest_structure"]} in (
        entity["facts"]
    )
    assert entity["unknown"] == ["matchmaking", "fireteam_size"]
    assert entity["live_sensitive"] == ["current_availability", "rewards"]
    assert entity["not_same_as"] == ["Haunted Altars of Sorrow"]
    assert len(result["limitations"]) == 2
    assert result["source"] == "core_knowledge" and result["corpus_version"]


def test_concept_tool_shorthand_alias_and_status_match_the_engine() -> None:
    hand_cannon = provider_result("HC")
    trials = provider_result("trials")
    haunted = provider_result("Haunted Altars of Sorrow")

    assert (hand_cannon["entity"]["name"], hand_cannon["matched_via"]) == (
        "Hand Cannon",
        "shorthand",
    )
    assert (trials["entity"]["name"], trials["matched_via"]) == ("Trials of Osiris", "alias")
    assert haunted["entity"]["status"] == "seasonal"


def test_concept_tool_ambiguity_returns_every_candidate() -> None:
    result = provider_result("Altars")

    assert result["found"] is False and result["status"] == "ambiguous"
    assert [value["name"] for value in result["candidates"]] == [
        "Altars of Sorrow",
        "Haunted Altars of Sorrow",
        "Altars of Summoning",
        "Altar of Reflection",
    ]
    assert "entity" not in result


@pytest.mark.parametrize("name", ["Altars of Sorow", "shotty", "AR", "Prismatic", "hand canon"])
def test_concept_tool_never_fuzzy_matches(name: str) -> None:
    result = provider_result(name)

    assert (result["found"], result["status"]) == (False, "unknown")
    assert "does not fuzzy-match" in result["limitations"][0]


def test_concept_tool_validates_arguments_and_tool_names() -> None:
    with pytest.raises(ValidationError):
        provider_result("")
    with pytest.raises(UnknownKnowledgeToolError):
        asyncio.run(CoreKnowledgeProvider(ENGINE).execute("get_item_details", {"name": "HC"}))


def test_core_maps_to_the_existing_manifest_stream_stage() -> None:
    assert status_for_tool("get_destiny_concept", "core") == status_for_tool(
        "get_item_details", "manifest"
    )


def test_core_tool_results_are_never_guardian_evidence() -> None:
    answer, _, trace = chat(
        "What is a Hand Cannon?", [[concept_call("Hand Cannon")], [_answer("A weapon type.")]]
    )

    assert answer.message == "A weapon type."
    assert answer.sources == []
    assert trace["grounding"]["core_knowledge"] is True
    assert trace["grounding"]["guardian_account"] is False
    assert trace["number_of_guardian_tools_used"] == 0
    assert trace["knowledge_categories_used"] == ["core"]
    assert trace["tool_trace"] == [
        {
            "name": "get_destiny_concept",
            "category": "core",
            "result_source": "core_knowledge",
            "resolution": "entity",
            "matched_via": "canonical",
            "entity_kind": "weapon_archetype",
        }
    ]
    assert trace["core_knowledge"]["tool_used"] is True


# --- Per-turn grounded-facts block ------------------------------------------------------


def instructions_of(responses: Any, index: int = 0) -> str:
    return responses.requests[index]["instructions"]


def test_bundle_is_appended_once_after_routing_and_keeps_the_static_prefix() -> None:
    _, responses, trace = chat("Where is Altars of Sorrow?")

    instructions = instructions_of(responses)
    bundle_text = instructions.split(f"{CORE_FACTS_HEADER}\n", 1)[1]
    assert instructions.startswith(SYSTEM_INSTRUCTIONS)
    assert instructions.count(CORE_FACTS_HEADER) == 1
    assert instructions.endswith(f"\n\n{CORE_FACTS_HEADER}\n{bundle_text}")
    assert json.loads(bundle_text)["entities"][0]["name"] == "Altars of Sorrow"
    assert trace["core_knowledge"]["instruction_chars_added"] == (
        len(CORE_FACTS_HEADER) + 1 + len(bundle_text) + 2
    )


def test_unmatched_turn_adds_zero_characters() -> None:
    message = "Tell me about Shatterdive and the AR I use."
    _, with_core, trace = chat(message)
    without_service, without_core = service(knowledge_service=None, outputs=[[_answer("A.")]])
    without_service.knowledge = None
    asyncio.run(without_service.chat(ChatRequest(message=message), guardian_context()))

    assert instructions_of(with_core) == instructions_of(without_core)
    assert CORE_FACTS_HEADER not in instructions_of(with_core)
    assert trace["core_knowledge"]["bundle_attached"] is False
    assert trace["core_knowledge"]["instruction_chars_added"] == 0
    assert trace["grounding"]["core_knowledge"] is False


def test_bundle_is_built_once_and_identical_on_every_model_round(monkeypatch: Any) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    original = ENGINE.bundle

    def counting(message: str, candidate_names: tuple[str, ...] = ()) -> Any:
        calls.append((message, candidate_names))
        return original(message, candidate_names)

    monkeypatch.setattr(ENGINE, "bundle", counting)
    _, responses, _ = chat(
        "Is a raid PvE?",
        [[concept_call("Raid")], [concept_call("Dungeon", "core-2")], [_answer("Yes.")]],
    )

    assert len(calls) == 1
    assert len(responses.requests) == 3
    assert len({value["instructions"] for value in responses.requests}) == 1


def test_core_grounding_adds_no_model_round() -> None:
    _, responses, trace = chat("What is a Nightfall?")

    assert len(responses.requests) == 1
    assert trace["core_knowledge"]["bundle_attached"] is True
    assert trace["tool_budget"]["rounds_used"] == 1


def test_planning_candidates_feed_the_bundle_in_order_after_message_entities(
    monkeypatch: Any,
) -> None:
    seen: list[tuple[str, ...]] = []
    original = ENGINE.bundle

    def capture(message: str, candidate_names: tuple[str, ...] = ()) -> Any:
        seen.append(candidate_names)
        return original(message, candidate_names)

    monkeypatch.setattr(ENGINE, "bundle", capture)
    _, responses, trace = chat("What should I do next? Maybe a raid?")

    bundle = json.loads(instructions_of(responses).split(f"{CORE_FACTS_HEADER}\n", 1)[1])
    planning_json = instructions_of(responses).split("planning context follows")[1]
    candidate_names = [
        value["name"]
        for value in json.loads(planning_json.split(":\n", 1)[1].split("\n")[0])["candidates"]
    ]
    assert seen == [tuple(candidate_names)]
    assert "Nightfall" in candidate_names
    assert [value["name"] for value in bundle["entities"]] == ["Raid", "Nightfall"]
    assert [value["origin"] for value in trace["core_knowledge"]["resolutions"]] == [
        "message",
        "candidate",
    ]
    assert instructions_of(responses).index("planning context follows") < (
        instructions_of(responses).index(CORE_FACTS_HEADER)
    )


def test_planning_refresh_reattaches_the_same_block_without_rebuilding(monkeypatch: Any) -> None:
    calls: list[str] = []
    original = ENGINE.bundle

    def counting(message: str, candidate_names: tuple[str, ...] = ()) -> Any:
        calls.append(message)
        return original(message, candidate_names)

    monkeypatch.setattr(ENGINE, "bundle", counting)
    context = guardian_context()

    async def refresh(_tool: str) -> Any:
        return context

    recommendations, responses = service(
        outputs=[
            [_call("get_active_quests", {"character_id": None, "character_class": None}, "q-1")],
            [_answer("Run the Nightfall.")],
        ]
    )
    asyncio.run(
        recommendations.chat(
            ChatRequest(message="What should I do next?"), context, refresh_guardian=refresh
        )
    )

    assert len(calls) == 1
    assert all(instructions_of(responses, index).count(CORE_FACTS_HEADER) == 1 for index in (0, 1))
    assert instructions_of(responses, 0) == instructions_of(responses, 1)


def test_altars_reaches_the_model_as_ambiguity_not_a_choice() -> None:
    _, responses, trace = chat('What does "Altars" refer to?')

    bundle = json.loads(instructions_of(responses).split(f"{CORE_FACTS_HEADER}\n", 1)[1])
    assert "entities" not in bundle
    assert len(bundle["ambiguous"][0]["candidates"]) == 4
    assert trace["core_knowledge"]["ambiguity_groups"] == ["ambiguity.altars"]
    assert trace["core_knowledge"]["matched_entity_ids"] == []


def test_hc_resolves_and_unregistered_shorthand_injects_nothing() -> None:
    _, hc_responses, hc_trace = chat("What is an HC?")
    _, ar_responses, ar_trace = chat("What is an AR?")

    assert hc_trace["core_knowledge"]["resolutions"] == [
        {"id": "weapon_archetype.hand_cannon", "via": "shorthand", "origin": "message"}
    ]
    assert '"name":"Hand Cannon"' in instructions_of(hc_responses)
    assert CORE_FACTS_HEADER not in instructions_of(ar_responses)
    assert ar_trace["core_knowledge"]["resolutions"] == []


def test_live_sensitive_metadata_never_bypasses_the_live_gate() -> None:
    answer, responses, trace = chat("What is the Nightfall this week?")

    assert trace["required_live_topic"] == "weekly_nightfall"
    assert trace["answer"] == "unavailable_live_data"
    assert answer.source == "local"
    # The model ran (web research was possible) but found nothing, so the limit stands.
    assert trace["grounding"]["web_research"] is False
    assert trace["grounding"]["live_provider"] is False


def test_core_provider_works_on_the_streaming_path() -> None:
    recommendations, responses = service(
        streaming=True,
        outputs=[[concept_call("Altars")], [_answer("Which altar do you mean?")]],
    )
    emitted: list[Any] = []

    async def emit(event: Any) -> None:
        emitted.append(event)

    reporter = StreamStatusReporter(emit)
    answer = asyncio.run(
        recommendations.chat(
            ChatRequest(message="Where are the altars?"),
            guardian_context(),
            stream_status=reporter,
        )
    )
    trace = recommendations.latest_trace()

    assert answer.message == "Which altar do you mean?"
    assert "manifest" in reporter.emitted_categories
    assert reporter.emitted_categories[-1] == "final"
    assert all(value["instructions"].count(CORE_FACTS_HEADER) == 1 for value in responses.requests)
    assert trace["tool_trace"][0]["resolution"] == "ambiguous"
    assert trace["tool_trace"][0]["candidate_count"] == 4


def test_trace_reports_caps_omissions_and_sizes() -> None:
    message = (
        "Titan or Hunter for the raid, dungeon and Nightfall? I run an HC, a sniper, a sword "
        "and two exotics, Exotic armor too, before Trials of Osiris, Altars, Gambit, Iron "
        "Banner and Lost Sectors."
    )
    _, responses, trace = chat(message)
    core = trace["core_knowledge"]

    assert core["truncated"] is True
    assert core["omitted"]["entities"] > 0
    assert core["ambiguity_groups"] == ["ambiguity.altars"]
    assert core["matched_entity_ids"][:2] == ["class.titan", "class.hunter"]
    assert {value["via"] for value in core["resolutions"]} >= {"canonical", "ambiguity"}
    assert core["bundle_chars"] <= 2400
    assert core["bundle_token_estimate"] == -(-core["bundle_chars"] // 4)
    assert core["corpus_version"] == ENGINE.corpus_version
    assert core["manifest_conflicts"] == [] and core["destination_conflicts"] == []
    assert instructions_of(responses).count(CORE_FACTS_HEADER) == 1


# --- Trace-only signals -----------------------------------------------------------------


def test_destination_conflict_is_trace_only() -> None:
    routed = knowledge(FakeManifestProvider())
    wrong = "Altars of Sorrow is located on Nessus."
    right = "Altars of Sorrow is in Sorrow's Harbor on The Moon."

    wrong_answer, _, wrong_trace = chat("Where is Altars of Sorrow?", [[_answer(wrong)]], routed)
    right_answer, _, right_trace = chat("Where is Altars of Sorrow?", [[_answer(right)]], routed)

    assert wrong_answer.message == wrong
    assert right_answer.message == right
    assert wrong_trace["core_knowledge"]["destination_conflicts"] == [
        {
            "entity_id": "activity.altars_of_sorrow",
            "grounded_located_on": "The Moon",
            "answer_destinations": ["Nessus"],
        }
    ]
    assert right_trace["core_knowledge"]["destination_conflicts"] == []
    assert wrong_trace["planning_correction"] == right_trace["planning_correction"]


def test_exact_manifest_destination_contradicting_curation_is_traced() -> None:
    contradiction = {
        "found": True,
        "activity": {"name": "Altars of Sorrow", "destination": {"name": "Nessus"}},
        "resolution": {"exact": True, "matched_name": "Altars of Sorrow"},
    }
    closest = {**contradiction, "resolution": {"exact": False, "matched_name": "Altars of Sorrow"}}
    call = _call("get_activity_details", {"activity_name_or_hash": "Altars of Sorrow"}, "m-1")

    _, _, conflict_trace = chat(
        "Where is Altars of Sorrow?",
        [[call], [_answer("It is on the Moon.")]],
        knowledge(FakeManifestProvider(contradiction)),
    )
    _, _, closest_trace = chat(
        "Where is Altars of Sorrow?",
        [[call], [_answer("It is on the Moon.")]],
        knowledge(FakeManifestProvider(closest)),
    )

    assert conflict_trace["core_knowledge"]["manifest_conflicts"] == [
        {
            "entity_id": "activity.altars_of_sorrow",
            "grounded_located_on": "The Moon",
            "manifest_destination": "Nessus",
        }
    ]
    assert closest_trace["core_knowledge"]["manifest_conflicts"] == []


HAND_CANNON_FIND = _find("find-hc", subtype="Hand Cannon")
ROCKET_FIND = _find("find-rocket", subtype="Rocket Launcher")


@pytest.mark.parametrize(
    ("alias_terms", "arguments", "expected"),
    [
        (("Hand Cannon",), {"subtype": "Hand Cannon"}, "match"),
        (("Hand Cannon",), {"subtype": "Rocket Launcher"}, "mismatch"),
        ((), {"subtype": "Hand Cannon"}, "not_applicable"),
        # A resolved alias that is not in this Guardian's own gear vocabulary is ignored.
        (("Glaive",), {"subtype": "Glaive"}, "not_applicable"),
    ],
)
def test_relevance_uses_registered_aliases_only_through_alias_terms(
    alias_terms: tuple[str, ...], arguments: dict[str, Any], expected: str
) -> None:
    assert (
        forced_alternatives_relevance(
            "Anything better than my HC in my vault?",
            build_context(),
            arguments,
            {},
            alias_terms=alias_terms,
        )
        == expected
    )


@pytest.mark.parametrize("find_call", [HAND_CANNON_FIND, ROCKET_FIND])
def test_registered_aliases_change_only_the_trace_relevance_signal(find_call: Any) -> None:
    message = "Anything better than my HC in my vault?"
    outputs = [[ANALYZE_CALL, find_call], [_answer("Keep your current setup.")]]
    with_core, with_requests, with_trace = chat(message, outputs, context=build_context())
    plain_service, plain_requests = service(outputs=outputs)
    plain_service.knowledge = None
    plain = asyncio.run(plain_service.chat(ChatRequest(message=message), build_context()))
    plain_trace = plain_service.latest_trace()

    expected = "match" if find_call is HAND_CANNON_FIND else "mismatch"
    assert with_trace["build_analysis"]["forced_alternatives_relevance"] == [expected]
    assert plain_trace["build_analysis"]["forced_alternatives_relevance"] == ["not_applicable"]
    assert with_core.message == plain.message
    assert [value["tool_choice"] for value in with_requests.requests] == [
        value["tool_choice"] for value in plain_requests.requests
    ]
    assert with_trace["planning_correction"] == plain_trace["planning_correction"]


# --- Hot path: no network or disk, bounded time -----------------------------------------


def test_chat_hot_path_reads_no_files_and_opens_no_sockets(monkeypatch: Any) -> None:
    import socket

    recommendations, _ = service(outputs=[[concept_call("Altars of Sorrow")], [_answer("Moon.")]])
    context = guardian_context()
    # The event loop needs its own socket pair on Windows, so create it before the guards.
    loop = asyncio.new_event_loop()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("The chat hot path must not read files or open sockets.")

    try:
        with monkeypatch.context() as guarded:
            guarded.setattr(builtins, "open", forbidden)
            guarded.setattr(Path, "read_text", forbidden)
            guarded.setattr(Path, "open", forbidden)
            guarded.setattr(socket, "socket", forbidden)
            guarded.setattr(socket, "create_connection", forbidden)
            answer = loop.run_until_complete(
                recommendations.chat(ChatRequest(message="Where is Altars of Sorrow?"), context)
            )
    finally:
        loop.close()

    assert answer.message == "Moon."
    trace = recommendations.latest_trace()
    assert trace["core_knowledge"]["bundle_attached"] is True
    assert trace["core_knowledge"]["tool_used"] is True


def test_core_grounding_time_on_the_production_corpus() -> None:
    recommendations, _ = service()
    names = [ENGINE.entity_brief(value)["name"] for value in list(ENGINE._entities)]  # noqa: SLF001
    message = " ".join(["Should I run", *names, "with an HC?"])[:2000]
    durations = []
    for _ in range(25):
        started = time.perf_counter()
        recommendations._core_grounding(message, None)  # noqa: SLF001
        durations.append((time.perf_counter() - started) * 1000)

    assert statistics.median(durations) < 15
