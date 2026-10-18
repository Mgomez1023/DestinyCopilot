"""Conversation character scope: which character a conversation is about, how an explicit
switch changes it, and what is invalidated or rebuilt when it does."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.ai import RecommendationService
from app.character_selection import (
    CharacterNotFoundError,
    resolve_conversation_scope,
)
from app.config import Settings
from app.guardian_tools import GuardianToolService
from app.models import ChatRequest, ChatTurn
from app.ownership_evidence import (
    OwnershipEvidence,
    evidence_summary,
    find_ownership_claims,
    record_ownership_evidence,
)
from app.session_preferences import derive_session_preferences
from evals.profiles import PROFILES
from tests.test_build_analysis import ScriptedBuildResponses, _answer, _call
from tests.test_current_character import ANALYZE, NULLS

WEAK_STAT_ANSWER = (
    "Your weakest stat is Health at 80 against 100 Grenade, so survivability suffers."
)
PLAN = "What should I do next?"
PLAN_REPLY = "Complete Kepler Uplink, the active Edge of Fate step, on your Titan."
HUNTER_REPLY = "Run Gambit on your Hunter; it fits a short session."
SEARCH = {
    **NULLS,
    "query": None,
    "item_type": None,
    "subtype": None,
    "bucket": None,
    "equipped_only": False,
    "limit": 100,
}


def turns(*pairs: tuple[str, str]) -> list[ChatTurn]:
    return [ChatTurn(role=role, content=text) for role, text in pairs]  # type: ignore[arg-type]


def scope_for(profile: str, history: list[ChatTurn], message: str):
    context = PROFILES[profile]()
    previous = derive_session_preferences(history, "").preferences.character
    derived = derive_session_preferences(history, message)
    return resolve_conversation_scope(
        context,
        previous,
        derived.preferences.character,
        chosen_this_turn="character" in derived.current_turn_updates,
    )


# --- A. Scope persistence ---------------------------------------------------------------------


def test_scope_defaults_to_bungie_current_then_follows_each_explicit_switch() -> None:
    history: list[ChatTurn] = []
    first = scope_for("titan_veteran", history, PLAN)
    assert (first.active, first.source, first.switched) == ("Titan", "current_default", False)
    assert first.bungie_current == "Titan"

    history += turns(("user", PLAN), ("assistant", PLAN_REPLY))
    hunter = scope_for("titan_veteran", history, "Switch to my Hunter.")
    assert (hunter.active, hunter.source, hunter.previous_active, hunter.switched) == (
        "Hunter",
        "explicit_prompt",
        "Titan",
        True,
    )

    history += turns(("user", "Switch to my Hunter."), ("assistant", HUNTER_REPLY))
    persisted = scope_for("titan_veteran", history, "What should I do?")
    assert (persisted.active, persisted.source, persisted.switched) == (
        "Hunter",
        "persisted_conversation_scope",
        False,
    )
    assert persisted.bungie_current == "Titan"  # The Bungie default is never rewritten.

    history += turns(("user", "What should I do?"), ("assistant", HUNTER_REPLY))
    warlock = scope_for("titan_veteran", history, "Let's play my Warlock now.")
    assert (warlock.active, warlock.previous_active, warlock.switched) == (
        "Warlock",
        "Hunter",
        True,
    )
    history += turns(("user", "Let's play my Warlock now."), ("assistant", "Sure."))
    after = scope_for("titan_veteran", history, "What am I using right now?")
    assert (after.active, after.source) == ("Warlock", "persisted_conversation_scope")


@pytest.mark.parametrize(
    "message",
    ["Switch to my Hunter.", "use my Hunter", "Let's play Hunter", "Actually, my Hunter."],
)
def test_confident_switch_phrasings_set_the_scope(message: str) -> None:
    assert scope_for("titan_veteran", [], message).active == "Hunter"


# --- B. Explicit override ----------------------------------------------------------------------


def test_an_explicit_class_wins_for_the_prompt_and_stays_the_scope_until_changed() -> None:
    history = turns(
        ("user", "Switch to my Hunter."),
        ("assistant", HUNTER_REPLY),
    )
    override = scope_for("titan_veteran", history, "Can my Titan use Heart of Inmost Light?")
    assert (override.active, override.source, override.switched) == (
        "Titan",
        "explicit_prompt",
        True,
    )

    history += turns(("user", "Can my Titan use Heart of Inmost Light?"), ("assistant", "Yes."))
    # Documented semantics: an explicit class is sticky, not a one-off.
    follow_up = scope_for("titan_veteran", history, "What am I using right now?")
    assert (follow_up.active, follow_up.source) == ("Titan", "persisted_conversation_scope")


def test_naming_several_classes_without_a_switch_word_does_not_change_scope() -> None:
    history = turns(("user", "Switch to my Hunter."), ("assistant", HUNTER_REPLY))

    comparison = scope_for("titan_veteran", history, "Compare my Titan and my Warlock armor.")

    assert (comparison.active, comparison.switched) == ("Hunter", False)


def test_explicit_selector_still_beats_the_conversation_default() -> None:
    result = GuardianToolService(PROFILES["titan_veteran"](), "Hunter").execute(  # type: ignore[arg-type]
        "analyze_current_build", {**ANALYZE, "character_class": "Titan"}
    )

    assert result["character_class"] == "Titan"


# --- F. Reset ---------------------------------------------------------------------------------


def test_a_new_conversation_and_explicit_resets_return_to_the_bungie_default() -> None:
    fresh = scope_for("titan_veteran", [], "What am I using right now?")
    assert (fresh.active, fresh.source) == ("Titan", "current_default")

    history = turns(("user", "Switch to my Hunter."), ("assistant", HUNTER_REPLY))
    for message in ("Start over. What should I do?", "Any character is fine."):
        reset = scope_for("titan_veteran", history, message)
        assert (reset.active, reset.source) == ("Titan", "current_default"), message


# --- Tools default to the conversation's character -----------------------------------------------


def test_unspecified_selectors_use_the_conversation_character_not_the_last_played() -> None:
    context = PROFILES["titan_veteran"]()
    tools = GuardianToolService(context, "Hunter")  # type: ignore[arg-type]

    assert tools.execute("analyze_current_build", ANALYZE)["character_class"] == "Hunter"
    assert tools.execute("find_build_alternatives", _find())["character_class"] == "Hunter"
    assert tools.execute("get_equipped_loadout", NULLS)["character"]["class"] == "Hunter"
    assert tools.execute("get_build_details", NULLS)["character"]["class"] == "Hunter"
    # Without an explicit scope the Bungie current character still applies (P0-1).
    default = GuardianToolService(context)
    assert default.execute("get_equipped_loadout", NULLS)["character"]["class"] == "Titan"


def _find() -> dict[str, Any]:
    return {
        **NULLS,
        "slot": None,
        "item_type": "Weapon",
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


def test_per_character_views_do_not_leak_other_characters_inside_a_scoped_conversation() -> None:
    context = PROFILES["titan_veteran"]()

    unscoped = GuardianToolService(context).execute("get_active_quests", NULLS)
    hunter = GuardianToolService(context, "Hunter").execute("get_active_quests", NULLS)  # type: ignore[arg-type]
    warlock = GuardianToolService(context, "Warlock").execute(  # type: ignore[arg-type]
        "get_available_activities", NULLS
    )

    assert "The Edge of Fate" in json.dumps(unscoped)  # Existing all-characters semantics.
    assert hunter["quests"] == [] and "Edge of Fate" not in json.dumps(hunter)
    assert warlock["activities"]
    assert all(value["character_ids"] == ["warlock-1"] for value in warlock["activities"])


def test_account_wide_views_keep_their_semantics_inside_a_scoped_conversation() -> None:
    tools = GuardianToolService(PROFILES["titan_veteran"](), "Hunter")  # type: ignore[arg-type]

    summary = tools.execute("get_character_summary", NULLS)
    inventory = tools.execute("search_inventory", SEARCH)

    assert len(summary["characters"]) == 3
    assert {value["class"] for value in summary["characters"] if value["is_current"]} == {"Titan"}
    assert inventory["total_matching"] > 0


def test_a_missing_character_fails_honestly_instead_of_reverting_to_current() -> None:
    tools = GuardianToolService(PROFILES["missing_class"](), "Hunter")  # type: ignore[arg-type]

    with pytest.raises(CharacterNotFoundError):
        tools.execute("get_equipped_loadout", NULLS)
    with pytest.raises(CharacterNotFoundError):
        tools.execute("analyze_current_build", ANALYZE)


# --- E. Equipped evidence belongs to a character ------------------------------------------------


def titan_hunter_evidence() -> OwnershipEvidence:
    context = PROFILES["titan_veteran"]()
    classes = {value.character_id: value.class_name for value in context.characters}
    evidence = OwnershipEvidence()
    record_ownership_evidence(
        evidence,
        "search_inventory",
        GuardianToolService(context).execute("search_inventory", SEARCH),
        classes,
    )
    return evidence


def test_equipped_evidence_records_the_character_and_never_crosses_over() -> None:
    evidence = titan_hunter_evidence()

    assert evidence.equipped_on["synthoceps"] == ["Titan"]
    assert evidence.equipped_on["celestial nighthawk"] == ["Hunter"]
    assert evidence.source_for("Synthoceps", "equipped", "Titan") == "inventory_search"
    assert evidence.source_for("Synthoceps", "equipped", "Hunter") is None
    assert evidence.source_for("Celestial Nighthawk", "equipped", "Titan") is None
    assert evidence.source_for("Celestial Nighthawk", "equipped", "Hunter") is not None
    # Possession is account-wide: the Titan's Exotic is still owned when asking about Hunter.
    assert evidence.source_for("Synthoceps", "possession", "Hunter") == "inventory_search"


def test_a_hunter_equipped_claim_is_unsupported_by_titan_gear() -> None:
    evidence = titan_hunter_evidence()

    equipped = find_ownership_claims("Synthoceps is equipped.", evidence, "Hunter")
    owned = find_ownership_claims("You have Synthoceps.", evidence, "Hunter")
    titan = find_ownership_claims("Synthoceps is equipped.", evidence, "Titan")

    assert [value.supported for value in equipped] == [False]
    assert [value.supported for value in owned] == [True]
    assert [value.supported for value in titan] == [True]


def test_evidence_without_a_character_cannot_prove_a_scoped_claim_against_known_owners() -> None:
    evidence = OwnershipEvidence()
    evidence.add("Rose", "equipped_loadout", equipped=True)  # Result did not say which.
    evidence.add("Rampage", "equipped_loadout", equipped=True, character="Titan")

    assert evidence.source_for("Rose", "equipped", "Hunter") == "equipped_loadout"
    assert evidence.source_for("Rampage", "equipped", "Hunter") is None


def test_the_correction_evidence_lists_only_the_active_characters_equipped_gear() -> None:
    evidence = titan_hunter_evidence()

    hunter_summary = evidence_summary(evidence, "Hunter")

    equipped = hunter_summary.split("equipped on the Hunter: ", 1)[1].split(".")[0]
    assert "Celestial Nighthawk" in equipped and "Synthoceps" not in equipped
    assert "Synthoceps" in hunter_summary  # Still listed as owned.


# --- C and D. End to end: planning and build leakage ------------------------------------------


def chat(
    history: list[ChatTurn], message: str, outputs: list[list[Any]] | None = None
) -> tuple[Any, ScriptedBuildResponses, dict[str, Any]]:
    responses = ScriptedBuildResponses(outputs or [[_answer(HUNTER_REPLY)]] * 3)
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)
    answer = asyncio.run(
        service.chat(ChatRequest(message=message, history=history), PROFILES["titan_veteran"]())
    )
    return answer, responses, service.latest_trace() or {}


def planning_json(instructions: str) -> str:
    return instructions.split("session planning context follows", 1)[1]


@pytest.mark.parametrize(
    ("message", "present", "absent_class"),
    [
        ("Actually, switch to my Hunter.", "Gambit", "Hunter"),
        ("Switch to my Warlock.", "Vanguard Ops", "Warlock"),
    ],
)
def test_a_switch_after_planning_rebuilds_planning_for_the_new_character(
    message: str, present: str, absent_class: str
) -> None:
    history = turns(("user", PLAN), ("assistant", PLAN_REPLY))

    _, responses, trace = chat(history, message)

    request = responses.requests[0]
    planning = planning_json(request["instructions"])
    assert trace["session_planning"] is True
    assert "Edge of Fate" not in planning and "Kepler" not in planning
    assert present in planning
    assert f"Explicit character scope: {absent_class}" in request["instructions"]
    # The Titan reply is replaced rather than left for the model to reuse.
    sent = json.dumps(request["input"])
    assert "Kepler Uplink" not in sent
    assistant = [v for v in request["input"] if v.get("role") == "assistant"]
    assert len(assistant) == 1 and "omitted" in assistant[0]["content"]
    scope = trace["conversation_scope"]
    assert scope == {
        "bungie_current": "Titan",
        "active": absent_class,
        "source": "explicit_prompt",
        "previous_active": "Titan",
        "switched": True,
        "planning_continuation": True,
        "planning_context_rebuilt": True,
        "stale_context_removed": {"assistant_turns": 1},
    }


def test_the_titan_quest_is_present_before_the_switch_as_a_control() -> None:
    _, responses, trace = chat([], PLAN)

    assert "Edge of Fate" in planning_json(responses.requests[0]["instructions"])
    assert trace["conversation_scope"]["source"] == "current_default"
    assert trace["conversation_scope"]["planning_context_rebuilt"] is False


def test_a_switch_outside_a_planning_conversation_does_not_become_planning() -> None:
    history = turns(("user", "What is a dungeon?"), ("assistant", "A three-player activity."))

    _, _, trace = chat(history, "Switch to my Hunter.")

    assert trace["session_planning"] is False
    assert trace["conversation_scope"]["planning_continuation"] is False


def test_later_unspecified_turns_stay_on_the_new_character_and_keep_stale_replies_out() -> None:
    history = turns(
        ("user", PLAN),
        ("assistant", PLAN_REPLY),
        ("user", "Actually, switch to my Hunter."),
        ("assistant", HUNTER_REPLY),
    )

    _, responses, trace = chat(history, "What should I do?")

    request = responses.requests[0]
    assistant = [v["content"] for v in request["input"] if v.get("role") == "assistant"]
    assert "Edge of Fate" not in planning_json(request["instructions"])
    assert "omitted" in assistant[0] and assistant[1] == HUNTER_REPLY
    assert "Kepler" not in json.dumps(request["input"])
    scope = trace["conversation_scope"]
    assert (scope["active"], scope["source"], scope["switched"]) == (
        "Hunter",
        "persisted_conversation_scope",
        False,
    )
    assert scope["stale_context_removed"] == {"assistant_turns": 1}


def test_tools_in_a_switched_conversation_use_the_new_character_and_say_so_in_the_trace() -> None:
    history = turns(("user", PLAN), ("assistant", PLAN_REPLY))
    outputs = [
        [_call("get_active_quests", NULLS, "quests-1")],
        [_answer(HUNTER_REPLY)],
        [_answer(HUNTER_REPLY)],
    ]

    _, responses, trace = chat(history, "Switch to my Hunter.", outputs)

    tool_output = next(
        json.loads(v["output"])
        for request in responses.requests[1:]
        for v in request["input"]
        if isinstance(v, dict) and v.get("type") == "function_call_output"
    )
    assert tool_output["quests"] == []
    assert trace["character_selections"] == [
        {
            "tool": "get_active_quests",
            "selector": "conversation_default",
            "class": None,
            "used_current_class": None,
            "used_conversation_class": "Hunter",
            "class_without_user_scope": False,
        }
    ]


def test_a_build_turn_after_a_switch_analyzes_and_validates_against_the_new_character() -> None:
    history = turns(
        ("user", "What's the weakest part of my current build?"),
        ("assistant", "Your Synthoceps are equipped; Health is low."),
    )
    answer_text = (
        "You have Synthoceps equipped, so your weakest stat is Health at 80 against Grenade.\n"
        "Move one Recuperation mod toward Recovery to even out the stat spread.\n"
        "Keeping Health and Recovery closer together makes the build more forgiving in fights."
    )
    outputs = [
        [
            _call("analyze_current_build", ANALYZE, "analysis-1"),
            _call("search_inventory", SEARCH, "search-1"),
        ],
        [_answer(answer_text)],
        [_answer(answer_text)],
    ]

    answer, responses, trace = chat(history, "Switch to my Hunter and critique my build", outputs)

    analysis = next(
        json.loads(v["output"])
        for request in responses.requests[1:]
        for v in request["input"]
        if isinstance(v, dict)
        and v.get("type") == "function_call_output"
        and "equipped_weapons" in v["output"]
    )
    assert analysis["character_class"] == "Hunter"
    ownership = trace["ownership_validation"]
    assert ownership["equipped_evidence_characters"]["Hunter"] > 0
    assert ownership["equipped_evidence_characters"]["Titan"] > 0
    first = ownership["passes"][0]
    assert first["active_character"] == "Hunter"
    # The Titan's Synthoceps is owned, but not equipped on the Hunter: that sentence is removed.
    assert [(c["item"], c["claim_type"], c["action"]) for c in first["claims"]] == [
        ("Synthoceps", "equipped", "removed")
    ]
    assert "Synthoceps" not in answer.message
    assert "Recuperation" in answer.message
    assert "Explicit character scope: Hunter" in responses.requests[0]["instructions"]


# --- G. Regressions -----------------------------------------------------------------------------


def test_default_current_character_behavior_is_unchanged_without_a_switch() -> None:
    _, responses, trace = chat(
        [],
        "What's the weakest part of my current build?",
        [
            [_call("analyze_current_build", ANALYZE, "analysis-1")],
            [_answer(WEAK_STAT_ANSWER)],
            [_answer(WEAK_STAT_ANSWER)],
        ],
    )

    assert (
        "Current character (most recently played): Titan." in responses.requests[0]["instructions"]
    )
    assert trace["character_selections"][0]["selector"] == "current_default"
    assert trace["conversation_scope"]["source"] == "current_default"
    assert trace["conversation_scope"]["stale_context_removed"] == {"assistant_turns": 0}


def test_a_switch_after_only_a_character_choice_is_not_a_planning_request() -> None:
    history = turns(("user", "Switch to my Hunter."), ("assistant", HUNTER_REPLY))

    _, _, trace = chat(history, "Switch to my Titan.")

    # The switch-continuation rule does not fire. (Planning mode can still come from the
    # existing preference rule that treats any turn after a stated constraint as a
    # follow-up; that rule is unchanged and tracked separately.)
    assert trace["conversation_scope"]["switched"] is True
    assert trace["conversation_scope"]["planning_continuation"] is False


def test_an_explicit_class_question_in_a_build_conversation_stays_a_build_turn() -> None:
    history = turns(("user", "Switch to my Warlock."), ("assistant", "Done."))

    _, _, trace = chat(history, "What about my Hunter build?")

    assert trace["conversation_scope"]["active"] == "Hunter"
    assert trace["conversation_scope"]["planning_continuation"] is False


def test_a_stray_constraint_without_a_planning_request_does_not_make_a_switch_planning() -> None:
    history = turns(("user", "Solo only please."), ("assistant", "Understood."))

    _, _, trace = chat(history, "Switch to my Warlock.")

    assert trace["session_planning"] is False
    assert trace["conversation_scope"]["active"] == "Warlock"
    assert trace["conversation_scope"]["planning_continuation"] is False
