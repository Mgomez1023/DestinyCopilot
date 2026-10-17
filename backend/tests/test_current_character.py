"""Deterministic current-character resolution: the selection rule, the resolver default, the
tools that use it, the routing line, and the trace (no model calls)."""

import asyncio
import json
import random
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.ai import RecommendationService
from app.character_selection import (
    CharacterSelectionError,
    GuardianCharacterResolver,
    resolve_current_character,
    selector_source,
)
from app.config import Settings
from app.guardian_tools import (
    CURRENT_DEFAULT_TOOLS,
    GUARDIAN_TOOL_DEFINITIONS,
    GuardianToolService,
)
from app.models import CharacterSummary, ChatRequest, GuardianContext
from evals.profiles import PROFILES
from tests.test_build_analysis import (
    BOTH_EVIDENCE_CHOICE,
    ScriptedBuildResponses,
    _answer,
    _call,
)

BASE = datetime(2026, 10, 17, 18, tzinfo=UTC)


def character(
    class_name: str,
    character_id: str,
    last_played: datetime | None,
    minutes: int = 100,
) -> CharacterSummary:
    return CharacterSummary(
        character_id=character_id,
        class_name=class_name,
        race_name="Human",
        gender_name="Female",
        power=2000,
        last_played=last_played,
        minutes_played_total=minutes,
    )


def context_of(*characters: CharacterSummary) -> GuardianContext:
    value = PROFILES["titan_veteran"]()
    value.characters = list(characters)
    return value


# --- Selection rule ---------------------------------------------------------------------


@pytest.mark.parametrize("current", ["Titan", "Hunter", "Warlock"])
def test_each_class_can_be_the_most_recently_played(current: str) -> None:
    offsets = {"Titan": 10, "Hunter": 20, "Warlock": 30}
    offsets[current] = 0
    value = context_of(
        *(
            character(name, str(index), BASE - timedelta(days=days))
            for index, (name, days) in enumerate(offsets.items(), start=1)
        )
    )

    selected = resolve_current_character(value)

    assert selected is not None
    assert (selected.class_name, selected.reason) == (current, "most_recently_played")
    assert selected.candidates == 3
    assert selected.last_played == BASE


def test_selection_does_not_depend_on_list_order() -> None:
    characters = [
        character("Titan", "1", BASE - timedelta(days=5)),
        character("Hunter", "2", BASE),
        character("Warlock", "3", BASE - timedelta(days=9)),
    ]
    results = set()
    for seed in range(12):
        shuffled = characters[:]
        random.Random(seed).shuffle(shuffled)
        selected = resolve_current_character(context_of(*shuffled))
        assert selected is not None
        results.add(selected.class_name)

    assert results == {"Hunter"}


def test_a_character_with_a_timestamp_beats_one_without() -> None:
    value = context_of(
        character("Titan", "9", None, minutes=9999),
        character("Warlock", "1", BASE - timedelta(days=400), minutes=1),
    )

    selected = resolve_current_character(value)

    assert selected is not None
    assert (selected.class_name, selected.reason) == ("Warlock", "most_recently_played")


def test_last_played_tie_falls_back_to_total_playtime() -> None:
    value = context_of(
        character("Titan", "5", BASE, minutes=100),
        character("Hunter", "2", BASE, minutes=900),
        character("Warlock", "3", BASE - timedelta(days=1), minutes=5000),
    )

    selected = resolve_current_character(value)

    assert selected is not None
    assert (selected.class_name, selected.reason) == ("Hunter", "last_played_tie_playtime")


def test_full_tie_falls_back_to_the_highest_numeric_character_id() -> None:
    value = context_of(
        character("Titan", "2305843009213693999", BASE, minutes=100),
        character("Hunter", "2305843009213694001", BASE, minutes=100),
        character("Warlock", "999", BASE, minutes=100),
    )

    selected = resolve_current_character(value)

    assert selected is not None
    # A numeric comparison, not a string comparison ("999" > "2305..." as text).
    assert (selected.class_name, selected.reason) == ("Hunter", "last_played_tie_character_id")


@pytest.mark.parametrize(
    ("minutes", "expected", "reason"),
    [
        ((100, 500, 300), "Hunter", "last_played_unavailable_playtime"),
        ((100, 100, 100), "Warlock", "last_played_unavailable_character_id"),
    ],
)
def test_no_timestamps_at_all_uses_the_documented_fallback(
    minutes: tuple[int, int, int], expected: str, reason: str
) -> None:
    value = context_of(
        character("Titan", "1", None, minutes[0]),
        character("Hunter", "2", None, minutes[1]),
        character("Warlock", "3", None, minutes[2]),
    )

    selected = resolve_current_character(value)

    assert selected is not None
    assert (selected.class_name, selected.reason) == (expected, reason)
    assert selected.last_played is None


def test_missing_classes_single_character_and_no_characters() -> None:
    two = resolve_current_character(PROFILES["missing_class"]())
    single = resolve_current_character(PROFILES["new_player"]())

    assert two is not None and two.class_name == "Titan" and two.candidates == 2
    assert single is not None
    assert (single.class_name, single.reason, single.candidates) == (
        "Hunter",
        "only_character",
        1,
    )
    assert resolve_current_character(context_of()) is None


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("titan_veteran", "Titan"),
        ("hunter_veteran", "Hunter"),
        ("warlock_veteran", "Warlock"),
        ("returning_player", "Warlock"),
    ],
)
def test_synthetic_profiles_resolve_their_main_character(profile: str, expected: str) -> None:
    selected = resolve_current_character(PROFILES[profile]())

    assert selected is not None and selected.class_name == expected


def test_selector_source() -> None:
    assert selector_source("abc", None) == "explicit_id"
    assert selector_source(None, "Hunter") == "explicit_class"
    assert selector_source(None, None) == "unspecified"
    assert selector_source("", "") == "unspecified"


# --- Resolver ---------------------------------------------------------------------------


def test_resolver_defaults_to_current_only_when_asked_and_explicit_wins() -> None:
    value = PROFILES["hunter_veteran"]()
    resolver = GuardianCharacterResolver(value)

    assert resolver.resolve_one(default_to_current=True).class_name == "Hunter"
    assert resolver.resolve_one(character_class="Titan", default_to_current=True).class_name == (
        "Titan"
    )
    hunter_id = next(c.character_id for c in value.characters if c.class_name == "Hunter")
    titan_id = next(c.character_id for c in value.characters if c.class_name == "Titan")
    assert resolver.resolve_one(character_id=titan_id, default_to_current=True).class_name == (
        "Titan"
    )
    assert resolver.resolve_one(character_id=hunter_id).class_name == "Hunter"
    with pytest.raises(CharacterSelectionError, match="character_id or character_class"):
        resolver.resolve_one()
    with pytest.raises(CharacterSelectionError, match="not both"):
        resolver.resolve_one(
            character_id=hunter_id, character_class="Hunter", default_to_current=True
        )


# --- Tools ------------------------------------------------------------------------------

NULLS = {"character_id": None, "character_class": None}
ANALYZE = {**NULLS, "goal": None, "activity": None, "locked_items": None, "preserve_exotics": False}
FIND = {
    **NULLS,
    "slot": None,
    "item_type": None,
    "subtype": None,
    "damage_type": None,
    "rarity": None,
    "required_perks": None,
    "exotic": None,
    "equipped": None,
    "locations": None,
    "preserve_exotics": True,
    "locked_items": None,
    "limit": 3,
}


def execute(profile: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return GuardianToolService(PROFILES[profile]()).execute(name, arguments)


@pytest.mark.parametrize(
    ("profile", "current"),
    [("titan_veteran", "Titan"), ("hunter_veteran", "Hunter"), ("warlock_veteran", "Warlock")],
)
def test_current_build_tools_target_the_resolved_character(profile: str, current: str) -> None:
    analysis = execute(profile, "analyze_current_build", ANALYZE)
    alternatives = execute(profile, "find_build_alternatives", FIND)
    loadout = execute(profile, "get_equipped_loadout", NULLS)
    details = execute(profile, "get_build_details", NULLS)

    assert analysis["character_class"] == current
    assert alternatives["character_class"] == current
    assert loadout["character"]["class"] == current
    assert details["character"]["class"] == current


def test_explicit_class_still_overrides_the_current_character() -> None:
    for name, base in (
        ("analyze_current_build", ANALYZE),
        ("find_build_alternatives", FIND),
        ("get_equipped_loadout", NULLS),
        ("get_build_details", NULLS),
    ):
        result = execute("titan_veteran", name, {**base, "character_class": "Hunter"})
        selected = result.get("character_class") or result["character"]["class"]
        assert selected == "Hunter", name
    warlock = execute(
        "hunter_veteran", "analyze_current_build", {**ANALYZE, "character_class": "Titan"}
    )
    assert warlock["character_class"] == "Titan"


def test_explicit_class_for_a_missing_character_still_fails_instead_of_defaulting() -> None:
    with pytest.raises(ValueError, match="not found"):
        execute("missing_class", "analyze_current_build", {**ANALYZE, "character_class": "Hunter"})


def test_character_summary_marks_the_current_character() -> None:
    result = execute("hunter_veteran", "get_character_summary", NULLS)

    assert result["current_character"] == "Hunter"
    assert {value["class"]: value["is_current"] for value in result["characters"]} == {
        "Hunter": True,
        "Titan": False,
    }


def test_existing_tool_scoping_is_unchanged() -> None:
    # Content progression still needs an explicit selector.
    with pytest.raises(ValueError, match="character_id or character_class"):
        execute(
            "titan_veteran",
            "get_content_progression",
            {**NULLS, "content_name": None},
        )
    # Optional-selector tools still cover every character with no selector.
    quests = execute("titan_veteran", "get_active_quests", NULLS)
    assert {value["character_class"] for value in quests["quests"]} == {"Titan"}
    summary = execute("titan_veteran", "get_character_summary", NULLS)
    assert len(summary["characters"]) == 3
    # Class-scoped inventory search keeps its filtering and exclusion count.
    hunter = execute(
        "large_vault",
        "search_inventory",
        {**NULLS, "character_class": "Hunter", "item_type": "Armor", "limit": 100},
    )
    armor = [value for value in hunter["items"] if value["item_type"] == "Armor"]
    assert armor and all(value["class_restriction"] in {"Hunter", "Any", None} for value in armor)
    assert hunter["excluded_other_class_count"] > 0
    unscoped = execute(
        "large_vault", "search_inventory", {**NULLS, "item_type": "Armor", "limit": 100}
    )
    assert (
        "excluded_other_class_count" not in unscoped or not unscoped["excluded_other_class_count"]
    )


def test_tool_schemas_state_the_current_default_and_stay_strict() -> None:
    from app.build_analysis import BUILD_TOOL_DEFINITIONS

    definitions = {
        value["name"]: value for value in [*GUARDIAN_TOOL_DEFINITIONS, *BUILD_TOOL_DEFINITIONS]
    }
    for name in CURRENT_DEFAULT_TOOLS:
        parameters = definitions[name]["parameters"]
        assert definitions[name]["strict"] is True
        assert parameters["additionalProperties"] is False
        assert {"character_id", "character_class"} <= set(parameters["required"])
        assert parameters["properties"]["character_id"]["type"] == ["string", "null"]
        assert None in parameters["properties"]["character_class"]["enum"]
        text = json.dumps(parameters).casefold()
        assert "current" in text and "most recently played" in text, name
    assert "most recently played" not in json.dumps(
        definitions["get_content_progression"]["parameters"]
    )


# --- Routing and trace ------------------------------------------------------------------


def turn_instructions(profile: str, scope: str | None) -> str:
    value = PROFILES[profile]()
    service = RecommendationService(Settings(openai_api_key="test-key"))
    responses = ScriptedBuildResponses([[_answer("Fine.")]])
    service.client = SimpleNamespace(responses=responses)
    asyncio.run(service.chat(ChatRequest(message="Hello there"), value))
    # Exercise the routing builder directly for both scopes.
    from app.build_analysis import derive_build_request_context
    from app.response_quality import response_mode_context
    from app.session_preferences import SessionPreferenceContext
    from app.title_progress import derive_title_request_context

    mode = response_mode_context(
        "Hello",
        session_planning=False,
        continuation=False,
        build_request=False,
        focused_build=False,
    )
    return service._turn_instructions(  # noqa: SLF001
        False,
        None,
        scope,
        None,
        mode,
        derive_build_request_context("Hello", SessionPreferenceContext(), []),
        derive_title_request_context("Hello", []),
        resolve_current_character(value),
    )


def test_routing_names_the_current_character_unless_the_user_named_one() -> None:
    implicit = turn_instructions("hunter_veteran", None)
    explicit = turn_instructions("hunter_veteran", "Titan")

    assert "Current character (most recently played): Hunter." in implicit
    assert "pass null for both character_id and character_class" in implicit
    assert "never guess a class" in implicit
    assert "Current character (most recently played)" not in explicit
    assert "Explicit character scope: Titan" in explicit


async def run_build_turn(
    profile: str, message: str, call_arguments: dict[str, Any]
) -> tuple[Any, ScriptedBuildResponses, dict[str, Any]]:
    responses = ScriptedBuildResponses(
        [
            [_call("analyze_current_build", call_arguments, "analysis-1")],
            [_answer("Weakest part: boss damage. Add a heavy option.")],
            [_answer("Weakest part: boss damage. Add a heavy option.")],
        ]
    )
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)
    answer = await service.chat(ChatRequest(message=message), PROFILES[profile]())
    return answer, responses, service.latest_trace() or {}


def analyze_tool_output(responses: ScriptedBuildResponses) -> dict[str, Any]:
    for request in responses.requests[1:]:
        for item in request["input"]:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                return json.loads(item["output"])
    raise AssertionError("no tool output")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "current"),
    [("titan_veteran", "Titan"), ("hunter_veteran", "Hunter"), ("warlock_veteran", "Warlock")],
)
async def test_current_build_prompt_never_requires_the_model_to_guess_a_class(
    profile: str, current: str
) -> None:
    _, responses, trace = await run_build_turn(
        profile, "What's the weakest part of my current build?", ANALYZE
    )

    first = responses.requests[0]
    assert first["tool_choice"] in (
        BOTH_EVIDENCE_CHOICE,
        {"type": "function", "name": "analyze_current_build"},
    )
    assert f"Current character (most recently played): {current}." in first["instructions"]
    assert analyze_tool_output(responses)["character_class"] == current
    assert trace["current_character"]["class"] == current
    assert trace["current_character"]["reason"] == "most_recently_played"
    assert trace["current_character"]["explicit_user_scope"] is None
    assert trace["character_selections"] == [
        {
            "tool": "analyze_current_build",
            "selector": "current_default",
            "class": None,
            "used_current_class": current,
            "used_conversation_class": None,
            "class_without_user_scope": False,
        }
    ]


@pytest.mark.asyncio
async def test_explicit_class_prompt_routes_to_that_class_and_is_traced_as_explicit() -> None:
    _, responses, trace = await run_build_turn(
        "titan_veteran",
        "What's the weakest part of my Hunter build?",
        {**ANALYZE, "character_class": "Hunter"},
    )

    first = responses.requests[0]["instructions"]
    assert "Current character (most recently played)" not in first
    assert "Explicit character scope: Hunter" in first
    assert analyze_tool_output(responses)["character_class"] == "Hunter"
    assert trace["current_character"]["class"] == "Titan"
    assert trace["current_character"]["explicit_user_scope"] == "Hunter"
    assert trace["character_selections"][0]["selector"] == "explicit_class"
    assert trace["character_selections"][0]["class_without_user_scope"] is False


@pytest.mark.asyncio
async def test_a_model_guessed_class_with_no_user_scope_is_flagged_in_the_trace() -> None:
    _, responses, trace = await run_build_turn(
        "titan_veteran",
        "What's the weakest part of my current build?",
        {**ANALYZE, "character_class": "Warlock"},
    )

    # Observability only: the explicit selector still wins, but the guess is recorded.
    assert analyze_tool_output(responses)["character_class"] == "Warlock"
    assert trace["character_selections"][0]["selector"] == "explicit_class"
    assert trace["character_selections"][0]["class_without_user_scope"] is True


@pytest.mark.asyncio
async def test_trace_never_contains_the_raw_character_id() -> None:
    value = PROFILES["titan_veteran"]()
    _, _, trace = await run_build_turn(
        "titan_veteran", "What am I using right now?", NULLS | ANALYZE
    )
    encoded = json.dumps(trace)
    ids = [c.character_id for c in value.characters]

    assert all(identifier not in encoded for identifier in ids)
    assert '"character_id"' not in encoded
    assert len(trace["current_character"]["character_ref"]) == 10
