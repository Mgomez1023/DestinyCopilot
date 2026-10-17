"""Deterministic product evals: the parts of correct assistant behavior that do not need a
model, run against the synthetic Guardian profiles and production code paths (CI-safe)."""

import asyncio
import json
from typing import Any

import pytest

from app.ai import RecommendationService
from app.config import Settings
from app.core_knowledge_provider import CoreKnowledgeProvider
from app.destiny_knowledge import DestinyKnowledgeService
from app.guardian_tools import CharacterNotFoundError, GuardianToolService
from app.live_knowledge import volatile_topic
from app.models import ChatTurn
from app.session_planning import build_session_planning_context
from app.session_preferences import derive_session_preferences
from evals.golden import GOLDEN
from evals.profiles import PROFILES, profile_facts
from evals.scenarios import BANK
from tests.test_core_knowledge_integration import ENGINE
from tests.test_manifest_resolution import provider as manifest_provider

SERVICE = RecommendationService(
    Settings(), DestinyKnowledgeService([CoreKnowledgeProvider(ENGINE)])
)


def bundle_for(message: str, candidates: Any = None) -> dict[str, Any]:
    grounding = SERVICE._core_grounding(message, candidates)  # noqa: SLF001
    assert grounding.block is not None, message
    return json.loads(grounding.block.split("\n", 1)[1])


def card(message: str, name: str) -> dict[str, Any]:
    payload = bundle_for(message)
    cards = payload.get("entities", []) + payload.get("rules", [])
    return next(value for value in cards if value["name"] == name)


def facts(value: dict[str, Any]) -> dict[str, Any]:
    return {predicate: fact for predicate, fact, _ in value.get("facts", [])}


def tools(profile: str) -> GuardianToolService:
    return GuardianToolService(PROFILES[profile]())


# --- Grounded knowledge for the golden prompts ------------------------------------------


def test_golden_knowledge_prompts_attach_the_expected_grounded_facts() -> None:
    where = card("Where is Altars of Sorrow?", "Altars of Sorrow")
    assert facts(where)["located_on"] == "The Moon"
    assert where["not_same_as"] == ["Haunted Altars of Sorrow"]
    matchmade = card("Is Altars of Sorrow matchmade?", "Altars of Sorrow")
    assert "matchmaking" in matchmade["unknown"] and "matchmaking" not in facts(matchmade)
    assert (
        "rewards"
        in card("Why should I finish Altars of Sorrow?", "Altars of Sorrow")["live_sensitive"]
    )
    assert (
        facts(card("Can I equip two Exotic armor pieces?", "Exotic armor"))[
            "max_equipped_exotic_armor"
        ]
        == 1
    )
    together = card("Can I equip an Exotic weapon and Exotic armor together?", "Exotic weapon")
    assert facts(together)["exotic_weapon_and_armor_together"] is True
    assert facts(card("What is a dungeon?", "Dungeon"))["fireteam_size"] == 3
    assert facts(card("What is a Nightfall?", "Nightfall"))["kind_of"] == "Strike"


def test_altars_ambiguity_reaches_the_model_explicitly() -> None:
    payload = bundle_for('What does "Altars" refer to?')

    assert "entities" not in payload
    assert len(payload["ambiguous"][0]["candidates"]) == 4


def test_hc_resolves_and_unregistered_shorthand_does_not() -> None:
    hc = card("Use a HC for this build.", "Hand Cannon")
    assert hc["matched"] == {"text": "hc", "via": "shorthand"}
    ar = SERVICE._core_grounding("Is an AR good for my Titan?", None)  # noqa: SLF001
    # "Titan" is a real class mention; the unregistered "AR" must not resolve to anything.
    assert ar.trace["matched_entity_ids"] == ["class.titan"]
    assert SERVICE._core_grounding("Best shotty I own?", None).block is None  # noqa: SLF001


def test_fuzzy_manifest_lookup_reports_not_exact(tmp_path: Any) -> None:
    result = asyncio.run(manifest_provider(tmp_path).get_activity_details("Altars of Sorrow"))

    assert result["activity"]["name"] == "Haunted Altars of Sorrow"
    assert result["resolution"]["exact"] is False


# --- Live vs static routing -------------------------------------------------------------


def test_live_scenarios_route_to_the_live_gate_and_static_ones_do_not() -> None:
    for value in BANK:
        topic = volatile_topic(value.prompt)
        if value.expect.live_sensitive:
            assert topic is not None, value.id
        elif value.tags.get("knowledge") == "stable":
            assert topic is None, (value.id, topic)
    by_id = {value.id: value for value in GOLDEN}
    assert volatile_topic(by_id["golden.nightfall_this_week"].prompt) == "weekly_nightfall"
    assert volatile_topic(by_id["golden.what_is_nightfall"].prompt) is None


# --- Ownership and class evidence -------------------------------------------------------


def test_owned_gear_evidence_lists_only_owned_items() -> None:
    owned = profile_facts(PROFILES["large_vault"]()).owned_items
    result = tools("large_vault").execute(
        "find_build_alternatives",
        {"character_class": "Titan", "subtype": "Hand Cannon", "limit": 8},
    )

    names = [value["name"] for value in result["candidates"]]
    assert names and set(names) <= set(owned)
    assert all(value["subtype"] == "Hand Cannon" for value in result["candidates"])


def test_wrong_class_exotic_armor_is_filtered_for_the_requested_class() -> None:
    result = tools("wrong_class_vault").execute(
        "find_build_alternatives",
        {"character_class": "Titan", "item_type": "Armor", "rarity": "Exotic", "limit": 8},
    )

    assert result["candidates"] == []
    assert result["excluded_other_class_count"] >= 4


def test_missing_class_is_an_explicit_lookup_failure() -> None:
    with pytest.raises(CharacterNotFoundError):
        tools("missing_class").execute("get_active_quests", {"character_class": "Hunter"})


# --- Quest state and planning -----------------------------------------------------------


def test_active_quest_state_is_exact_and_a_single_thread_is_never_active() -> None:
    result = tools("titan_veteran").execute("get_active_quests", {"character_class": "Titan"})
    assert [value["name"] for value in result["quests"]] == ["The Edge of Fate"]

    for name, factory in PROFILES.items():
        context = factory()
        planning = build_session_planning_context(
            context, "What should I do next?", "general", None
        )
        for candidate in planning.candidates:
            assert candidate.name != "A Single Thread", name
            if candidate.active_quest_status == "KNOWN_TRUE":
                active = {quest.name for value in context.characters for quest in value.quests}
                assert candidate.name in active, (name, candidate.name)


def test_hunter_switch_changes_preferences_and_account_context() -> None:
    history = [
        ChatTurn(role="user", content="I only have 30 minutes. Pick one thing for me."),
        ChatTurn(role="assistant", content="Run Vanguard Ops."),
    ]
    derived = derive_session_preferences(history, "Actually what if I wanted to play on my Hunter?")
    context = PROFILES["titan_veteran"]()
    titan = build_session_planning_context(context, "plan", "general", "Titan")
    hunter = build_session_planning_context(
        context, "plan", "general", "Hunter", derived.preferences
    )

    assert derived.preferences.character == "Hunter"
    assert derived.preferences.time_minutes == 30
    titan_names = {value.name for value in titan.candidates}
    hunter_names = {value.name for value in hunter.candidates}
    assert "Gambit" in hunter_names and "Gambit" not in titan_names
    assert "The Edge of Fate" in titan_names and "The Edge of Fate" not in hunter_names


def test_stale_data_profile_surfaces_missing_socket_evidence() -> None:
    result = tools("stale_manifest").execute("analyze_current_build", {"character_class": "Titan"})

    assert any("socket" in value.casefold() for value in result["limitations"])
