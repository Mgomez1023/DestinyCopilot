import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app import response_quality, session_planning
from app.ai import (
    FINAL_ROUND_LIMITATION_MESSAGE,
    MAX_TOOL_CALLS,
    MAX_TOOL_ROUNDS,
    SYSTEM_INSTRUCTIONS,
    RecommendationService,
)
from app.build_analysis import (
    AnalyzeCurrentBuildRequest,
    BuildAnalysisService,
    FindBuildAlternativesRequest,
    class_compatibility,
    classify_plug,
    derive_build_request_context,
    forced_alternatives_relevance,
)
from app.chat_stream import (
    CompletedStreamEvent,
    MessageDeltaStreamEvent,
    StreamStatusReporter,
    response_events,
)
from app.config import Settings
from app.guardian_tools import GuardianToolService
from app.models import (
    CharacterSummary,
    ChatRequest,
    ChatTurn,
    DataAvailability,
    GuardianContext,
    InventorySummary,
    ItemStatSummary,
    ItemSummary,
    SocketedPlugSummary,
)
from app.response_quality import (
    EVIDENCE_VIOLATION_CODES,
    GROUNDING_SAFETY_VIOLATION_CODES,
    STYLE_VIOLATION_CODES,
    BuildResponseValidationContext,
    ResponseQualityValidator,
    build_fallback_kind,
    violation_group,
)
from app.session_planning import PlanningViolation
from app.session_preferences import SessionPreferenceContext
from app.title_progress import TitleRequestContext


def plug(name: str, category: str = "frames", plug_type: str = "Trait") -> SocketedPlugSummary:
    # Manifest shape: socketed plugs are itemType 19 ("Mod"); the useful label lives in
    # itemTypeDisplayName and the role in plugCategoryIdentifier.
    return SocketedPlugSummary(
        item_hash=abs(hash(name)) % 2**32,
        name=name,
        description=f"Manifest description for {name}.",
        item_type="Mod",
        plug_type=plug_type,
        category_identifier=category,
    )


def item(
    name: str,
    item_type: str,
    *,
    subtype: str | None = None,
    bucket: str | None = None,
    damage_type: str | None = None,
    tier: str = "Legendary",
    location: str = "vault",
    character_id: str | None = None,
    equipped: bool = False,
    plugs: list[SocketedPlugSummary] | None = None,
    socket_data: bool | None = True,
    instance_data: bool | None = True,
    stats: list[ItemStatSummary] | None = None,
    empty_sockets: int | None = 0,
    class_restriction: str | None = "default",
) -> ItemSummary:
    values = plugs or []
    # Manifest shape: weapons use classType 3 (any class); armor and subclasses are locked.
    if class_restriction == "default":
        class_restriction = "Any" if item_type == "Weapon" else "Titan"
    return ItemSummary(
        item_hash=abs(hash((name, location, tuple(value.name for value in values)))) % 2**32,
        instance_id=f"private-{name}-{location}",
        name=name,
        item_type=item_type,
        item_subtype=subtype,
        tier=tier,
        class_restriction=class_restriction,  # type: ignore[arg-type]
        # Bungie reports a vaulted copy in the vault bucket ("General"); the slot is the
        # definition's, so both are modeled.
        bucket_name="General" if location in {"vault", "profile"} and bucket else bucket,
        equipment_slot=bucket,
        damage_type=damage_type,
        location=location,  # type: ignore[arg-type]
        character_id=character_id,
        is_equipped=equipped,
        instance_data_available=instance_data,
        socket_data_available=socket_data,
        stat_data_available=True,
        socket_count=len(values) + (empty_sockets or 0),
        empty_socket_count=empty_sockets,
        socketed_plugs=[value.name for value in values],
        socketed_plug_details=values,
        stats=stats or [],
    )


def build_context() -> GuardianContext:
    subclass = item(
        "Sunbreaker",
        "Subclass",
        location="equipped",
        character_id="titan-private-id",
        equipped=True,
        plugs=[
            plug("Thermite Grenade", "shared.solar.grenades", "Solar Grenade"),
            plug("Roaring Flames", "titan.solar.aspects", "Solar Aspect"),
            plug("Ember of Torches", "shared.solar.fragments", "Solar Fragment"),
        ],
    )
    sunshot = item(
        "Sunshot",
        "Weapon",
        subtype="Hand Cannon",
        bucket="Energy Weapons",
        damage_type="Solar",
        tier="Exotic",
        location="equipped",
        character_id="titan-private-id",
        equipped=True,
        plugs=[plug("Sun Blast", "intrinsics", "Intrinsic")],
    )
    hallowfire = item(
        "Hallowfire Heart",
        "Armor",
        subtype="Chest Armor",
        bucket="Chest Armor",
        tier="Exotic",
        location="equipped",
        character_id="titan-private-id",
        equipped=True,
        plugs=[plug("Solar Resistance", "enhancements.v2_chest", "Chest Armor Mod")],
        stats=[ItemStatSummary(name="Resilience", value=22)],
        empty_sockets=1,
    )
    titan = CharacterSummary(
        character_id="titan-private-id",
        class_name="Titan",
        race_name="Exo",
        gender_name="Female",
        power=0,
        subclass=subclass,
        equipped_gear=[subclass, sunshot, hallowfire],
    )
    other_character_item = item(
        "Other Character Rocket",
        "Weapon",
        subtype="Rocket Launcher",
        bucket="Power Weapons",
        damage_type="Solar",
        location="character",
        character_id="hunter-private-id",
        plugs=[plug("Explosive Light")],
    )
    inventory = [
        item(
            "Apex Predator",
            "Weapon",
            subtype="Rocket Launcher",
            bucket="Power Weapons",
            damage_type="Solar",
            plugs=[plug("Reconstruction"), plug("Bait and Switch")],
        ),
        item(
            "Apex Predator",
            "Weapon",
            subtype="Rocket Launcher",
            bucket="Power Weapons",
            damage_type="Solar",
            location="character",
            character_id="titan-private-id",
            plugs=[plug("Tracking Module"), plug("Explosive Light")],
        ),
        item(
            "Gjallarhorn",
            "Weapon",
            subtype="Rocket Launcher",
            bucket="Power Weapons",
            damage_type="Solar",
            tier="Exotic",
            plugs=[plug("Wolfpack Rounds", "intrinsics", "Intrinsic")],
        ),
        item(
            "Missing Roll Rocket",
            "Weapon",
            subtype="Rocket Launcher",
            bucket="Power Weapons",
            damage_type=None,
            socket_data=False,
            instance_data=False,
            empty_sockets=None,
        ),
        item(
            "Unknown",
            "Unknown",
            bucket="Power Weapons",
            socket_data=False,
            instance_data=False,
            empty_sockets=None,
        ),
        other_character_item,
    ]
    return GuardianContext(
        bungie_display_name="Sanitized",
        membership_id="private-membership",
        membership_type=3,
        platform_name="Steam",
        characters=[titan],
        inventory=InventorySummary(
            total_items=len(inventory),
            vault_items=4,
            character_items=2,
            unique_item_hashes=len(inventory),
            items=inventory,
            returned_items=len(inventory),
        ),
        data_availability=DataAvailability(
            fetched_at=datetime(2026, 9, 23, tzinfo=UTC),
            unavailable_components={"ItemSockets": "partial"},
            notes=["Some inventory components were incomplete."],
        ),
    )


def test_current_build_is_compact_factual_and_has_no_universal_score_or_ids() -> None:
    result = BuildAnalysisService(build_context()).analyze_current_build(
        AnalyzeCurrentBuildRequest(
            character_id="titan-private-id",
            goal="solo PvE",
            activity=None,
            locked_items=["Sunshot"],
            preserve_exotics=True,
        )
    )

    encoded = json.dumps(result)
    assert result["character_class"] == "Titan"
    assert result["subclass"]["aspects"][0]["name"] == "Roaring Flames"
    assert result["subclass"]["fragments"][0]["name"] == "Ember of Torches"
    assert result["armor_stat_totals"] == {"Resilience": 22}
    assert result["observations"]["observable_empty_sockets"] == 1
    assert {value["name"] for value in result["locked_items"]} == {
        "Sunshot",
        "Hallowfire Heart",
    }
    assert all(value["ownership"] == "verified_owned" for value in result["locked_items"])
    assert "No universal numeric or tier score" in result["scoring"]
    assert "titan-private-id" not in encoded
    assert "private-Sunshot" not in encoded
    assert "item_hash" not in encoded


def test_build_tools_resolve_titan_class_for_analysis_and_owned_alternatives() -> None:
    service = BuildAnalysisService(build_context())

    analysis = service.analyze_current_build(AnalyzeCurrentBuildRequest(character_class="Titan"))
    alternatives = service.find_build_alternatives(
        FindBuildAlternativesRequest(
            character_class="Titan",
            item_type="Weapon",
            subtype="Hand Cannon",
            preserve_exotics=True,
        )
    )

    assert analysis["character_class"] == "Titan"
    assert alternatives["character_class"] == "Titan"


@pytest.mark.parametrize(
    ("category", "item_type", "role"),
    [
        # Every category below appears in the Bungie Manifest; every plug is itemType 19.
        ("frames", "Mod", "perk"),
        ("intrinsics", "Mod", "perk"),
        ("origins", "Mod", "perk"),
        ("barrels", "Mod", "perk"),
        ("tubes", "Mod", "perk"),
        ("magazines", "Mod", "perk"),
        ("magazines_gl", "Mod", "perk"),
        ("scopes", "Mod", "perk"),
        ("bowstrings", "Mod", "perk"),
        ("catalysts", "Mod", "perk"),
        ("shader", "Mod", "cosmetic"),
        ("armor_skins_titan_arms", "Mod", "cosmetic"),
        ("exotic_new_grenade_launcher0_skins", "Mod", "cosmetic"),
        ("mementos", "Mod", "cosmetic"),
        ("v400.plugs.weapons.masterworks.trackers", "Mod", "cosmetic"),
        ("enhancements.v2_arms", "Mod", "mod"),
        ("v400.weapon.mod_guns", "Mod", "mod"),
        ("core.gear_systems.armor_tiering.plugs.tuning.mods", "Mod", "mod"),
        ("titan.solar.supers", "Mod", "ability"),
        ("warlock.prism.class_abilities", "Mod", "ability"),
        ("hunter.arc.movement", "Mod", "ability"),
        ("hunter.strand.melee", "Mod", "ability"),
        ("shared.solar.grenades", "Mod", "ability"),
        ("titan.prism.prism_grenade", "Mod", "ability"),
        ("titan.solar.aspects", "Mod", "aspect"),
        ("shared.void.fragments", "Mod", "fragment"),
        ("v300.plugs.masterworks.generic.armor.super_dr", "Mod", "other"),
        ("v460.plugs.armor.masterworks", "Mod", "other"),
        ("crafting.recipes.empty_socket", "Mod", "other"),
        (None, "Mod", "mod"),
        (None, None, "other"),
    ],
)
def test_plug_classifier_uses_real_manifest_categories(
    category: str | None, item_type: str | None, role: str
) -> None:
    assert classify_plug(category, item_type) == role


def _context_with_rolled_hand_cannon() -> GuardianContext:
    context = build_context()
    titan = context.characters[0]
    hand_cannon = item(
        "Test Hand Cannon",
        "Weapon",
        subtype="Hand Cannon",
        bucket="Kinetic Weapons",
        damage_type="Kinetic",
        location="equipped",
        character_id="titan-private-id",
        equipped=True,
        plugs=[
            plug("Adaptive Frame", "intrinsics", "Intrinsic"),
            plug("Hammer-Forged Rifling", "barrels", "Barrel"),
            plug("Ricochet Rounds", "magazines", "Magazine"),
            plug("Rampage", "frames", "Trait"),
            plug("Outlaw", "frames", "Trait"),
            plug("Veist Stinger", "origins", "Origin Trait"),
            plug("Backup Mag", "v400.weapon.mod_guns", "Weapon Mod"),
            plug("Rivalry Blacksand", "shader", "Shader"),
            plug("Kill Tracker", "v400.plugs.weapons.masterworks.trackers", ""),
        ],
    )
    titan.equipped_gear.append(hand_cannon)
    return context


def test_weapon_traits_barrels_and_magazines_are_perks_not_mods() -> None:
    result = BuildAnalysisService(_context_with_rolled_hand_cannon()).analyze_current_build(
        AnalyzeCurrentBuildRequest(character_class="Titan")
    )

    weapon = next(
        value for value in result["equipped_weapons"] if value["name"] == "Test Hand Cannon"
    )
    assert [value["name"] for value in weapon["relevant_perks"]] == [
        "Adaptive Frame",
        "Hammer-Forged Rifling",
        "Ricochet Rounds",
        "Rampage",
        "Outlaw",
        "Veist Stinger",
    ]
    by_name = {value["name"]: value for value in weapon["socketed_plugs"]}
    assert by_name["Rampage"]["type"] == "Trait"
    assert by_name["Rampage"]["role"] == "perk"
    assert by_name["Backup Mag"]["role"] == "mod"
    assert by_name["Rivalry Blacksand"]["role"] == "cosmetic"
    assert by_name["Kill Tracker"]["role"] == "cosmetic"
    armor_mod_names = {value["name"] for value in result["armor_mods"]}
    assert armor_mod_names == {"Solar Resistance"}


def test_build_details_mods_exclude_weapon_perks_and_cosmetics() -> None:
    result = GuardianToolService(_context_with_rolled_hand_cannon()).get_build_details(
        character_class="Titan"
    )

    assert result["mods"] == ["Solar Resistance", "Backup Mag"]
    sockets = result["other_build_affecting_sockets"]
    assert {"Rampage", "Outlaw", "Hammer-Forged Rifling", "Ricochet Rounds"} <= set(sockets)
    assert "Rivalry Blacksand" not in sockets
    assert "Kill Tracker" not in sockets


def test_owned_alternatives_exclude_other_class_armor_and_disclose_unknown_class() -> None:
    context = build_context()
    context.inventory.items.extend(
        [
            item(
                "Celestial Nighthawk",
                "Armor",
                subtype="Helmet",
                bucket="Helmet",
                tier="Exotic",
                class_restriction="Hunter",
            ),
            item(
                "Helm of Saint-14",
                "Armor",
                subtype="Helmet",
                bucket="Helmet",
                tier="Exotic",
                class_restriction="Titan",
            ),
            item(
                "Unresolved Helm",
                "Armor",
                subtype="Helmet",
                bucket="Helmet",
                class_restriction=None,
            ),
        ]
    )

    result = BuildAnalysisService(context).find_build_alternatives(
        FindBuildAlternativesRequest(
            character_class="Titan",
            item_type="Armor",
            slot="Helmet",
            preserve_exotics=False,
        )
    )

    names = {value["name"] for value in result["candidates"]}
    assert names == {"Helm of Saint-14", "Unresolved Helm"}
    assert result["excluded_other_class_count"] == 1
    assert result["total_matching_owned_copies"] == 2
    by_name = {value["name"]: value for value in result["candidates"]}
    assert by_name["Helm of Saint-14"]["class_restriction"] == "Titan"
    assert by_name["Unresolved Helm"]["class_restriction"] is None
    assert any("restricted to another class" in value for value in result["limitations"])
    assert any("no Manifest class restriction" in value for value in result["limitations"])


def test_unresolved_character_class_never_hides_class_locked_gear() -> None:
    restricted = item("Helm of Saint-14", "Armor", class_restriction="Titan")
    other = item("Celestial Nighthawk", "Armor", class_restriction="Hunter")
    any_class = item("Rocket", "Weapon")

    assert class_compatibility(restricted, "Unknown class") == "unknown"
    assert class_compatibility(other, "Unknown class") == "unknown"
    assert class_compatibility(any_class, "Unknown class") == "compatible"
    assert class_compatibility(restricted, "titan") == "compatible"
    assert class_compatibility(other, "Titan") == "incompatible"


def test_any_class_weapons_remain_owned_alternatives_without_class_notes() -> None:
    result = BuildAnalysisService(build_context()).find_build_alternatives(
        FindBuildAlternativesRequest(character_class="Titan", subtype="Rocket Launcher")
    )

    assert result["returned"] > 0
    assert result["excluded_other_class_count"] == 0
    assert all(value["class_restriction"] == "Any" for value in result["candidates"])
    assert not any("class" in value for value in result["limitations"])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"slot": "Power Weapons"}, 3),
        ({"subtype": "Rocket Launcher"}, 3),
        ({"damage_type": "Solar"}, 2),
        ({"rarity": "Exotic", "preserve_exotics": False}, 1),
        ({"exotic": False}, 3),
        ({"locations": ["character"]}, 1),
        ({"required_perks": ["Explosive Light"]}, 1),
    ],
)
def test_owned_alternative_filters(overrides: dict[str, Any], expected: int) -> None:
    values: dict[str, Any] = {
        "character_id": "titan-private-id",
        "slot": None,
        "item_type": "Weapon",
        "subtype": None,
        "damage_type": None,
        "rarity": None,
        "required_perks": None,
        "exotic": None,
        "equipped": False,
        "locations": None,
        "preserve_exotics": True,
        "locked_items": None,
        "limit": 8,
    }
    values.update(overrides)
    result = BuildAnalysisService(build_context()).find_build_alternatives(
        FindBuildAlternativesRequest(**values)
    )

    assert result["total_matching_owned_copies"] == expected
    assert all(value["name"] != "Other Character Rocket" for value in result["candidates"])


def test_duplicate_owned_copies_keep_actual_rolls_and_are_bounded() -> None:
    result = BuildAnalysisService(build_context()).find_build_alternatives(
        FindBuildAlternativesRequest(
            character_id="titan-private-id",
            slot="Power Weapons",
            item_type="Weapon",
            subtype="Rocket Launcher",
            damage_type="Solar",
            rarity="Legendary",
            required_perks=None,
            exotic=False,
            equipped=False,
            locations=None,
            preserve_exotics=True,
            locked_items=None,
            limit=1,
        )
    )

    assert result["total_matching_owned_copies"] == 2
    assert result["returned"] == 1
    assert result["truncated"] is True
    assert result["candidates"][0]["copy_number"] == 1
    assert result["candidates"][0]["socketed_plug_names"] == [
        "Tracking Module",
        "Explosive Light",
    ]


def test_missing_roll_data_and_unknown_damage_fail_safely() -> None:
    result = BuildAnalysisService(build_context()).find_build_alternatives(
        FindBuildAlternativesRequest(
            character_id="titan-private-id",
            slot="Power Weapons",
            item_type="Weapon",
            subtype=None,
            damage_type=None,
            rarity=None,
            required_perks=None,
            exotic=False,
            equipped=False,
            locations=None,
            preserve_exotics=True,
            locked_items=None,
            limit=8,
        )
    )

    missing = next(
        value for value in result["candidates"] if value["name"] == "Missing Roll Rocket"
    )
    assert missing["damage_type"] is None
    assert missing["data_quality"]["instance"] == "not_returned"
    assert missing["data_quality"]["roll_or_perk_data"] == "not_returned"
    assert any("cannot be judged" in value for value in result["limitations"])
    assert any("unresolved Manifest" in value for value in result["limitations"])
    assert "Guardian component unavailable: ItemSockets" in result["limitations"]


def test_locked_slot_and_exotic_preservation_prevent_incompatible_candidates() -> None:
    service = BuildAnalysisService(build_context())
    locked = service.find_build_alternatives(
        FindBuildAlternativesRequest(
            character_id="titan-private-id",
            slot="Energy Weapons",
            item_type="Weapon",
            subtype=None,
            damage_type=None,
            rarity=None,
            required_perks=None,
            exotic=None,
            equipped=None,
            locations=None,
            preserve_exotics=True,
            locked_items=["Sunshot"],
            limit=8,
        )
    )
    heavies = service.find_build_alternatives(
        FindBuildAlternativesRequest(
            character_id="titan-private-id",
            slot="Power Weapons",
            item_type="Weapon",
            subtype=None,
            damage_type=None,
            rarity=None,
            required_perks=None,
            exotic=None,
            equipped=False,
            locations=None,
            preserve_exotics=True,
            locked_items=["Sunshot"],
            limit=8,
        )
    )

    assert locked["candidates"] == []
    assert any("explicitly preserved" in value for value in locked["limitations"])
    assert "Gjallarhorn" not in {value["name"] for value in heavies["candidates"]}


def test_tool_inputs_are_strict_and_bounded() -> None:
    definitions = {value["name"]: value for value in GuardianToolService.definitions()}
    assert definitions["analyze_current_build"]["strict"] is True
    assert (
        definitions["find_build_alternatives"]["parameters"]["properties"]["limit"]["maximum"] == 8
    )
    with pytest.raises(ValidationError):
        FindBuildAlternativesRequest(
            character_id="titan-private-id",
            limit=9,
            unexpected="nope",  # type: ignore[call-arg]
        )


def test_build_request_constraints_capture_named_and_exotic_locks() -> None:
    preferences = SessionPreferenceContext(activity_mode="pve", fireteam="solo")
    named = derive_build_request_context("Build around Sunshot for solo PvE.", preferences)
    exotic = derive_build_request_context("Don't change my Exotic.", preferences)
    detailed = derive_build_request_context("Give me a complete detailed rebuild.", preferences)

    assert named.locked_items == ["Sunshot"]
    assert named.activity_mode == "pve"
    assert named.fireteam == "solo"
    assert exotic.preserve_equipped_exotics is True
    assert detailed.normal_change_limit is None


def test_build_locks_persist_across_history_until_explicitly_released() -> None:
    preferences = SessionPreferenceContext(primary_goal="build_improvement")
    history = [ChatTurn(role="user", content="Build around Sunshot and keep Hallowfire Heart.")]

    retained = derive_build_request_context("Make it easy to use.", preferences, history)
    released = derive_build_request_context("You can replace Sunshot now.", preferences, history)
    exotic = derive_build_request_context(
        "Make it solo-friendly.",
        preferences,
        [ChatTurn(role="user", content="Do this without changing my Exotic.")],
    )

    assert set(retained.locked_items) == {"Sunshot", "Hallowfire Heart"}
    assert released.locked_items == ["Hallowfire Heart"]
    assert exotic.preserve_equipped_exotics is True


@pytest.mark.parametrize(
    (
        "message",
        "requires_current",
        "requires_owned",
        "focused",
    ),
    [
        ("Check my Titan and give suggestions.", True, False, False),
        ("Check my weapons and suggest changes.", True, False, False),
        ("What should I use if I'm jumping into Iron Banner?", True, False, True),
        (
            "What should I use if I'm jumping into Iron Banner? Check my Guardian and give "
            "personalized recommendations.",
            True,
            False,
            True,
        ),
        ("I want to use a different hand cannon. Check my vault for a good one.", True, True, True),
        ("Anything better in my vault?", True, True, True),
        ("What gun should I run?", True, False, True),
        ("Find me a good hand cannon that I own.", False, True, True),
        ("What hand cannon that I own should I use?", True, True, True),
        ("Recommend a shotgun that I own.", False, True, True),
        ("Check my rolls.", False, True, True),
        ("What should I equip for PvP?", True, False, True),
        ("What perks are on my Thorn?", False, True, True),
    ],
)
def test_natural_personalized_gear_requests_route_to_bounded_build_evidence(
    message: str,
    requires_current: bool,
    requires_owned: bool,
    focused: bool,
) -> None:
    result = derive_build_request_context(message, SessionPreferenceContext())

    assert result.is_build_request is True
    assert result.requires_current_build_analysis is requires_current
    assert result.requires_owned_inventory is requires_owned
    assert result.focused_recommendation is focused


@pytest.mark.parametrize(
    "message",
    [
        "Is Thorn good?",
        "What does Explosive Payload do?",
        "What hand cannons are good this season?",
        "Should I keep doing this quest or start the campaign?",
    ],
)
def test_public_item_questions_are_not_personalized_build_requests(message: str) -> None:
    result = derive_build_request_context(message, SessionPreferenceContext())

    assert result.is_build_request is False
    assert result.requires_current_build_analysis is False
    assert result.requires_owned_inventory is False


@pytest.mark.parametrize(
    "message",
    [
        "Check perks too.",
        "Anything better in my vault?",
        "What else do I have?",
        "Nah, I don't wanna use Thorn. What else do I have?",
    ],
)
def test_natural_followups_keep_recent_personalized_build_context(message: str) -> None:
    history = [
        ChatTurn(
            role="user",
            content="What should I use for PvP? Check my Guardian and give suggestions.",
        )
    ]

    result = derive_build_request_context(message, SessionPreferenceContext(), history)

    assert result.is_build_request is True
    assert result.is_followup is True
    assert result.requires_owned_inventory is True


def test_owned_roll_followups_preserve_the_exact_live_conversation_context() -> None:
    opening = ChatTurn(
        role="user",
        content=(
            "Check my Titan and my vault. I'm playing Iron Banner and want to use a different "
            "hand cannon. Compare the hand cannons I actually own, check the perks on my "
            "copies, and recommend the best one for PvP with one backup."
        ),
    )
    rejection = "Nah I don't want that one. What else do I have?"

    second_turn = derive_build_request_context(
        rejection,
        SessionPreferenceContext(),
        [opening],
    )
    third_turn = derive_build_request_context(
        "Which exact copy has the better roll, and what perks make it better?",
        SessionPreferenceContext(),
        [opening, ChatTurn(role="user", content=rejection)],
    )

    assert second_turn.is_build_request is True
    assert second_turn.is_followup is True
    assert second_turn.followup_kind == "owned_inventory"
    assert second_turn.requires_owned_inventory is True
    assert third_turn.is_build_request is True
    assert third_turn.is_followup is True
    assert third_turn.followup_kind == "owned_inventory"
    assert third_turn.requires_owned_inventory is True
    assert third_turn.requires_current_build_analysis is False


def test_roll_comparison_without_recent_personalized_context_stays_general() -> None:
    result = derive_build_request_context(
        "Which exact copy has the better roll?",
        SessionPreferenceContext(),
    )

    assert result.is_build_request is False
    assert result.is_followup is False
    assert result.requires_owned_inventory is False


def test_assistant_text_alone_cannot_create_personalized_roll_context() -> None:
    result = derive_build_request_context(
        "Which roll is better?",
        SessionPreferenceContext(),
        [
            ChatTurn(
                role="assistant",
                content="I checked your vault and compared your owned hand cannon rolls.",
            )
        ],
    )

    assert result.is_build_request is False
    assert result.is_followup is False
    assert result.requires_owned_inventory is False


def test_explanation_followup_preserves_build_context_without_forcing_inventory() -> None:
    history = [
        ChatTurn(
            role="user",
            content="Check my rolls and recommend the best hand cannon that I own for PvP.",
        )
    ]

    result = derive_build_request_context(
        "Why that one?",
        SessionPreferenceContext(),
        history,
    )

    assert result.is_build_request is True
    assert result.is_followup is True
    assert result.followup_kind == "explanation"
    assert result.requires_owned_inventory is False
    assert result.requires_current_build_analysis is False


def test_detailed_whole_build_review_is_not_focused_or_inventory_forced() -> None:
    result = derive_build_request_context(
        "Review my whole Titan build in detail.", SessionPreferenceContext()
    )

    assert result.is_build_request is True
    assert result.requires_current_build_analysis is True
    assert result.requires_owned_inventory is False
    assert result.focused_recommendation is False
    assert result.normal_change_limit is None


def test_rejected_item_is_released_from_prior_build_lock() -> None:
    result = derive_build_request_context(
        "Nah, I don't wanna use Thorn. What else do I have?",
        SessionPreferenceContext(),
        [ChatTurn(role="user", content="Build around Thorn.")],
    )

    assert result.is_followup is True
    assert result.locked_items == []
    assert result.requires_owned_inventory is True


@pytest.mark.parametrize(
    ("message", "goal", "activity", "activity_mode", "fireteam"),
    [
        ("Make my loadout work for a campaign.", None, None, "pve", "either"),
        ("Make this better for solo PvE.", "solo_pve", None, "pve", "solo"),
        ("Prep me for The Shattered Throne.", None, "The Shattered Throne", "pve", "either"),
        ("Build me for a raid-style boss encounter.", None, None, "pve", "group"),
        ("Fix my PvP build.", "pvp", None, "pvp", "either"),
        ("Make an easy-to-use casual add clear build.", "add_clear", None, "pve", "either"),
        ("Make this good for boss DPS.", "boss_dps", None, "pve", "group"),
    ],
)
def test_activity_build_context_keeps_explicit_constraints(
    message: str,
    goal: str | None,
    activity: str | None,
    activity_mode: str,
    fireteam: str,
) -> None:
    preferences = SessionPreferenceContext(
        activity_mode=activity_mode,  # type: ignore[arg-type]
        fireteam=fireteam,  # type: ignore[arg-type]
    )

    result = derive_build_request_context(message, preferences)

    assert result.is_build_request is True
    assert result.goal == goal
    assert result.activity == activity
    assert result.activity_mode == activity_mode
    assert result.fireteam == fireteam


class BuildResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            output = [
                SimpleNamespace(
                    type="function_call",
                    name="analyze_current_build",
                    arguments=(
                        '{"character_id":"titan-private-id","goal":"solo PvE",'
                        '"activity":null,"locked_items":["Sunshot"],'
                        '"preserve_exotics":true}'
                    ),
                    call_id="build-1",
                )
            ]
            text = ""
        else:
            output = [
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text=(
                                "Keep Sunshot. Your current setup supports a Solar-focused review."
                            ),
                            annotations=[],
                        )
                    ],
                )
            ]
            text = "Keep Sunshot. Your current setup supports a Solar-focused review."
        return SimpleNamespace(
            output=output,
            output_text=text,
            status="completed",
            incomplete_details=None,
            error=None,
            usage=None,
        )


@pytest.mark.asyncio
async def test_build_tool_preserves_stateless_function_call_loop_and_safe_output() -> None:
    responses = BuildResponses()
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)

    answer = await service.chat(
        ChatRequest(message="Build around Sunshot for solo PvE."), build_context()
    )

    assert "Keep Sunshot" in answer.message
    assert len(responses.requests) == 2
    assert "previous_response_id" not in responses.requests[1]
    output = next(
        value
        for value in responses.requests[1]["input"]
        if isinstance(value, dict) and value.get("type") == "function_call_output"
    )
    assert output["call_id"] == "build-1"
    assert "titan-private-id" not in output["output"]
    assert "No universal numeric or tier score" in output["output"]
    trace = service.latest_trace()
    assert trace is not None
    assert trace["build_analysis"]["analysis_used"] is True


@pytest.mark.asyncio
async def test_iron_banner_runtime_call_resolves_titan_by_class() -> None:
    class TitanBuildResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                output = [
                    SimpleNamespace(
                        type="function_call",
                        name="analyze_current_build",
                        arguments=(
                            '{"character_id":null,"character_class":"Titan",'
                            '"goal":"pvp","activity":"Iron Banner",'
                            '"locked_items":null,"preserve_exotics":false}'
                        ),
                        call_id="titan-build-1",
                    )
                ]
                text = ""
            else:
                text = "For Iron Banner, keep Sunshot and tune the rest of your Titan setup."
                output = [
                    SimpleNamespace(
                        type="message",
                        id=f"titan-build-message-{len(self.requests)}",
                        role="assistant",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
                    )
                ]
            return SimpleNamespace(
                output=output,
                output_text=text,
                status="completed",
                incomplete_details=None,
                error=None,
                usage=None,
            )

    responses = TitanBuildResponses()
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=responses)

    answer = await service.chat(
        ChatRequest(
            message=(
                "What should I use if I'm jumping into Iron Banner? Check my Titan and give "
                "personalized recommendations."
            )
        ),
        build_context(),
    )

    assert answer.message.startswith("For Iron Banner")
    assert len(responses.requests) == 2
    tool_output = next(
        item
        for item in responses.requests[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert json.loads(tool_output["output"])["character_class"] == "Titan"
    assert "Active or nearly complete objectives" not in answer.message
    analysis_trace = service.latest_trace()["tool_trace"][0]
    assert analysis_trace["success"] is True
    assert analysis_trace["request"]["character_class"] == "Titan"
    assert analysis_trace["result_summary"]["character_class"] == "Titan"
    assert analysis_trace["result_summary"]["equipped_weapon_names"] == ["Sunshot"]
    assert analysis_trace["result_summary"]["equipped_exotic_names"] == [
        "Sunshot",
        "Hallowfire Heart",
    ]
    assert analysis_trace["result_summary"]["roll_data_complete"] is True


@pytest.mark.asyncio
async def test_vault_hand_cannon_runtime_calls_use_authenticated_guardian_data() -> None:
    class VaultResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                calls = [
                    SimpleNamespace(
                        type="function_call",
                        name="analyze_current_build",
                        arguments=(
                            '{"character_id":null,"character_class":"Titan",'
                            '"goal":null,"activity":null,"locked_items":null,'
                            '"preserve_exotics":false}'
                        ),
                        call_id="vault-analysis",
                    ),
                    SimpleNamespace(
                        type="function_call",
                        name="find_build_alternatives",
                        arguments=(
                            '{"character_id":null,"character_class":"Titan","slot":null,'
                            '"item_type":"Weapon","subtype":"Hand Cannon",'
                            '"damage_type":null,"rarity":null,"required_perks":null,'
                            '"exotic":null,"equipped":false,"locations":["vault"],'
                            '"preserve_exotics":true,"locked_items":null,"limit":8}'
                        ),
                        call_id="vault-options",
                    ),
                ]
                text = ""
            else:
                text = "I checked your Titan and found several returned hand cannon copies."
                calls = [
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
                    )
                ]
            return SimpleNamespace(
                output=calls,
                output_text=text,
                status="completed",
                incomplete_details=None,
                error=None,
                usage=None,
            )

    responses = VaultResponses()
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)
    context = build_context()
    context.inventory.items.extend(
        item(
            f"Vault Hand Cannon {index}",
            "Weapon",
            subtype="Hand Cannon",
            bucket="Kinetic Weapons",
            location="vault",
            plugs=[plug(f"Returned Perk {index}")],
        )
        for index in range(10)
    )

    answer = await service.chat(
        ChatRequest(
            message=(
                "I want to use a different hand cannon. Check my vault for a good one. "
                "Check perks too."
            ),
            history=[ChatTurn(role="user", content="Keep this scoped to my Titan.")],
        ),
        context,
    )

    assert "several returned hand cannon copies" in answer.message
    assert "sign in" not in answer.message.casefold()
    assert "link bungie" not in answer.message.casefold()
    assert "Pass character_class=Titan" in responses.requests[0]["instructions"]
    trace = service.latest_trace()
    assert trace["build_analysis"]["alternative_search_used"] is True
    alternative_trace = next(
        value for value in trace["tool_trace"] if value["name"] == "find_build_alternatives"
    )
    assert alternative_trace["success"] is True
    assert alternative_trace["request"]["character_class"] == "Titan"
    assert alternative_trace["request"]["item_type"] == "Weapon"
    assert alternative_trace["request"]["subtype"] == "Hand Cannon"
    assert alternative_trace["request"]["locations"] == ["vault"]
    assert alternative_trace["request"]["limit"] == 8
    assert alternative_trace["result_summary"]["returned"] == 8
    assert alternative_trace["result_summary"]["total_matching"] == 10
    assert alternative_trace["result_summary"]["truncated"] is True
    assert len(alternative_trace["result_summary"]["item_names"]) == 8
    encoded_trace = json.dumps(alternative_trace)
    assert "character_id" not in encoded_trace
    assert "instance_id" not in encoded_trace
    assert "membership_id" not in encoded_trace
    assert "private-" not in encoded_trace


@pytest.mark.asyncio
async def test_authenticated_character_lookup_failure_is_not_an_authentication_failure() -> None:
    service = RecommendationService(Settings(openai_api_key="test-key"))

    result = await service._execute_tool_call(  # noqa: SLF001 - regression boundary
        GuardianToolService(build_context()),
        "analyze_current_build",
        '{"character_id":null,"character_class":"Warlock"}',
    )

    assert result == {
        "error": {
            "code": "character_not_found",
            "message": "A Warlock was not found in Guardian data.",
            "authentication_required": False,
        }
    }
    assert "sign in" not in json.dumps(result).casefold()


def test_search_inventory_trace_is_sanitized_and_bounded() -> None:
    result = {
        "items": [
            {
                "name": f"Hand Cannon {index}",
                "character_id": "private-character",
                "instance_id": f"private-instance-{index}",
                "perks_and_sockets": ["Private full perk payload"],
            }
            for index in range(12)
        ],
        "total_matching": 12,
        "limit": 12,
        "truncated": False,
    }

    trace = RecommendationService._safe_tool_trace(  # noqa: SLF001 - trace contract
        "search_inventory",
        "guardian",
        json.dumps(
            {
                "query": "private user query",
                "character_id": "private-character",
                "character_class": None,
                "item_type": "Weapon",
                "subtype": "Hand Cannon",
                "bucket": None,
                "equipped_only": False,
                "limit": 12,
            }
        ),
        result,
    )

    assert trace["success"] is True
    assert trace["request"] == {
        "bucket": None,
        "character_class": None,
        "equipped_only": False,
        "item_type": "Weapon",
        "limit": 12,
        "subtype": "Hand Cannon",
    }
    assert trace["result_summary"] == {
        "item_names": [f"Hand Cannon {index}" for index in range(8)],
        "returned": 12,
        "total_matching": 12,
        "truncated": False,
    }
    encoded = json.dumps(trace)
    assert "private user query" not in encoded
    assert "private-character" not in encoded
    assert "private-instance" not in encoded
    assert "perks_and_sockets" not in encoded


def test_failed_guardian_tool_trace_records_only_safe_error_type() -> None:
    trace = RecommendationService._safe_tool_trace(  # noqa: SLF001 - trace contract
        "analyze_current_build",
        "guardian",
        '{"character_id":"private-character","character_class":null}',
        {
            "error": {
                "code": "character_not_found",
                "message": "Private character private-character was not found.",
                "authentication_required": False,
            }
        },
    )

    assert trace == {
        "name": "analyze_current_build",
        "category": "guardian",
        "request": {"character_class": None},
        "success": False,
        "error_type": "character_not_found",
    }
    assert "private-character" not in json.dumps(trace)


@pytest.mark.asyncio
async def test_build_validation_failure_uses_build_fallback_not_session_planning_copy() -> None:
    class InvalidBuildResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            text = "Use whatever objective is closest to completion."
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        id=f"invalid-build-{len(self.requests)}",
                        role="assistant",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
                    )
                ],
                output_text=text,
                status="completed",
                incomplete_details=None,
                error=None,
                usage=None,
            )

    responses = InvalidBuildResponses()
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=responses)

    answer = await service.chat(
        ChatRequest(
            message=(
                "What should I use if I'm jumping into Iron Banner? Check my Titan and give "
                "personalized recommendations."
            )
        ),
        build_context(),
    )

    assert answer.message == (
        "I couldn't inspect the required build data, so I can't give a grounded personalized "
        "loadout recommendation yet."
    )
    assert "Active or nearly complete objectives" not in answer.message
    assert service.latest_trace()["planning_correction"]["fallback"] == "build_evidence"


@pytest.mark.asyncio
async def test_build_quality_failure_uses_the_existing_single_correction_retry() -> None:
    class CorrectionResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="analyze_current_build",
                            arguments='{"character_id":"titan-private-id"}',
                            call_id="analysis-1",
                        )
                    ],
                    output_text="",
                    status="completed",
                    incomplete_details=None,
                    error=None,
                    usage=None,
                )
            text = (
                "Build score: 82/100. This is A Tier."
                if len(self.requests) == 2
                else "Keep Sunshot and focus the remaining slots on your stated goal."
            )
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        id=f"message-{len(self.requests)}",
                        role="assistant",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
                    )
                ],
                output_text=text,
                status="completed",
                incomplete_details=None,
                error=None,
                usage=None,
            )

    responses = CorrectionResponses()
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)

    answer = await service.chat(ChatRequest(message="Is my Titan build good?"), build_context())

    assert answer.message == "Keep Sunshot and focus the remaining slots on your stated goal."
    assert len(responses.requests) == 3
    assert responses.requests[2]["tool_choice"] == "none"
    trace = service.latest_trace()
    assert trace is not None
    assert trace["planning_correction"] == {
        "attempted": True,
        "violation_codes": ["fake_build_score"],
        "succeeded": True,
    }


@pytest.mark.asyncio
async def test_missing_required_build_evidence_uses_safe_correction_path() -> None:
    class MissingEvidenceResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            text = (
                "Use the hand cannon in your vault; it is better than your current weapon."
                if len(self.requests) == 1
                else (
                    "I can't make a personalized hand-cannon choice until I can check your "
                    "current setup and verify the owned copies in your vault."
                )
            )
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        id=f"message-{len(self.requests)}",
                        role="assistant",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
                    )
                ],
                output_text=text,
                status="completed",
                incomplete_details=None,
                error=None,
                usage=None,
            )

    responses = MissingEvidenceResponses()
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)

    answer = await service.chat(
        ChatRequest(message="Anything better in my vault?"), build_context()
    )

    assert answer.message.startswith("I can't make a personalized")
    assert len(responses.requests) == 2
    instructions = responses.requests[0]["instructions"]
    assert '"requires_current_build_analysis":true' in instructions
    assert '"requires_owned_inventory":true' in instructions
    assert "typical useful answer is about 50-120 words" in instructions
    trace = service.latest_trace()
    assert trace is not None
    assert trace["response_mode"] == "build_advice"
    assert trace["build_analysis"]["focused_recommendation"] is True
    assert trace["build_analysis"]["requires_current_build_analysis"] is True
    assert trace["build_analysis"]["requires_owned_inventory"] is True
    assert set(trace["planning_correction"]["violation_codes"]) == {
        "missing_current_build_analysis",
        "missing_owned_inventory_evidence",
    }
    assert trace["planning_correction"]["succeeded"] is True


def test_build_routing_requires_guardian_ownership_and_external_current_grounding() -> None:
    assert "call analyze_current_build" in SYSTEM_INSTRUCTIONS
    assert "find_build_alternatives" in SYSTEM_INSTRUCTIONS
    assert "Ownership of every immediate alternative MUST" in SYSTEM_INSTRUCTIONS
    assert '"Best DPS," "meta,"' in SYSTEM_INSTRUCTIONS
    assert "require Live or Web grounding" in SYSTEM_INSTRUCTIONS


class ScriptedBuildResponses:
    """Return scripted outputs in order and record each request's arguments."""

    def __init__(self, outputs: list[list[Any]]) -> None:
        self.outputs = outputs
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        output = self.outputs[len(self.requests) - 1]
        text = "".join(
            part.text for value in output if value.type == "message" for part in value.content
        )
        return SimpleNamespace(
            output=output,
            output_text=text,
            status="completed",
            incomplete_details=None,
            error=None,
            usage=None,
        )


def _call(name: str, arguments: dict[str, Any], call_id: str) -> Any:
    return SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id
    )


def _answer(text: str, index: int = 1) -> Any:
    return SimpleNamespace(
        type="message",
        id=f"answer-{index}",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
    )


TITAN = {"character_id": None, "character_class": "Titan"}
ANALYZE_CALL = _call(
    "analyze_current_build",
    {**TITAN, "goal": None, "activity": None, "locked_items": None, "preserve_exotics": False},
    "analysis-1",
)
FIND_CALL = _call(
    "find_build_alternatives",
    {
        **TITAN,
        "slot": "Power Weapons",
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
        "limit": 5,
    },
    "find-1",
)
FUNCTION_CHOICE = {
    "analyze_current_build": {"type": "function", "name": "analyze_current_build"},
    "find_build_alternatives": {"type": "function", "name": "find_build_alternatives"},
}
BOTH_EVIDENCE_CHOICE = {
    "type": "allowed_tools",
    "mode": "required",
    "tools": [
        {"type": "function", "name": "analyze_current_build"},
        {"type": "function", "name": "find_build_alternatives"},
    ],
}
GROUNDED_VAULT_ANSWER = (
    "Keep Sunshot. Your vault Apex Predator with Reconstruction is the strongest owned Power "
    "option for this goal."
)


_LAST_SERVICE: dict[str, RecommendationService] = {}


async def _chat(message: str, outputs: list[list[Any]]) -> tuple[Any, Any, Any]:
    responses = ScriptedBuildResponses(outputs)
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)
    _LAST_SERVICE["value"] = service
    answer = await service.chat(ChatRequest(message=message), build_context())
    return answer, responses, service.latest_trace()


@pytest.mark.asyncio
async def test_required_build_evidence_is_forced_up_front_in_one_parallel_round() -> None:
    answer, responses, trace = await _chat(
        "Anything better in my vault?",
        [[ANALYZE_CALL, FIND_CALL], [_answer(GROUNDED_VAULT_ANSWER)]],
    )

    assert answer.message == GROUNDED_VAULT_ANSWER
    assert len(responses.requests) == 2
    assert responses.requests[0]["tool_choice"] == BOTH_EVIDENCE_CHOICE
    assert responses.requests[0]["parallel_tool_calls"] is True
    assert responses.requests[1]["tool_choice"] == "auto"
    assert trace["build_analysis"]["forced_evidence_tools"] == [
        "analyze_current_build",
        "find_build_alternatives",
    ]
    assert trace["planning_correction"]["attempted"] is False


@pytest.mark.asyncio
async def test_missing_ownership_evidence_is_forced_after_analysis_only_round() -> None:
    answer, responses, _ = await _chat(
        "Anything better in my vault?",
        [[ANALYZE_CALL], [FIND_CALL], [_answer(GROUNDED_VAULT_ANSWER)]],
    )

    assert answer.message == GROUNDED_VAULT_ANSWER
    assert [value["tool_choice"] for value in responses.requests] == [
        BOTH_EVIDENCE_CHOICE,
        FUNCTION_CHOICE["find_build_alternatives"],
        "auto",
    ]


@pytest.mark.asyncio
async def test_only_current_build_analysis_is_forced_when_inventory_is_not_required() -> None:
    final = "Your Sunshot and Hallowfire Heart setup suits solo PvE."
    answer, responses, trace = await _chat(
        "Is my Titan build good?", [[ANALYZE_CALL], [_answer(final)]]
    )

    assert answer.message == final
    assert responses.requests[0]["tool_choice"] == FUNCTION_CHOICE["analyze_current_build"]
    assert responses.requests[1]["tool_choice"] == "auto"
    assert trace["build_analysis"]["forced_evidence_tools"] == ["analyze_current_build"]


@pytest.mark.asyncio
async def test_evidence_already_gathered_by_another_tool_is_not_forced_again() -> None:
    search_call = _call(
        "search_inventory",
        {
            **TITAN,
            "query": None,
            "item_type": "Weapon",
            "subtype": None,
            "bucket": None,
            "equipped_only": False,
            "limit": 10,
        },
        "search-1",
    )

    _, responses, _ = await _chat(
        "Anything better in my vault?",
        [[ANALYZE_CALL, search_call], [_answer(GROUNDED_VAULT_ANSWER)]],
    )

    # search_inventory already established ownership, so nothing else is forced.
    assert len(responses.requests) == 2
    assert responses.requests[1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_failed_forced_evidence_call_is_not_forced_twice() -> None:
    failing_analysis = _call(
        "analyze_current_build",
        {
            "character_id": "missing-character",
            "character_class": None,
            "goal": None,
            "activity": None,
            "locked_items": None,
            "preserve_exotics": False,
        },
        "analysis-error",
    )
    acknowledged = "I can't verify your current setup, so I can't judge this build yet."

    answer, responses, _ = await _chat(
        "Is my Titan build good?", [[failing_analysis], [_answer(acknowledged)]]
    )

    assert answer.message == acknowledged
    assert responses.requests[0]["tool_choice"] == FUNCTION_CHOICE["analyze_current_build"]
    assert responses.requests[1]["tool_choice"] == "auto"


def test_title_forcing_keeps_priority_and_final_round_allows_no_tools() -> None:
    title = TitleRequestContext(
        title_intent_present=True, intent_resolved=True, retrieval_required=True
    )
    needs_both = BuildResponseValidationContext(
        is_build_request=True,
        requires_current_build_analysis=True,
        requires_owned_inventory=True,
    )
    no_title = TitleRequestContext()
    choose = RecommendationService._round_tool_choice

    assert choose(1, title, needs_both, set()) == (
        {"type": "function", "name": "get_title_progress"},
        (),
    )
    assert choose(2, title, needs_both, set())[0] == BOTH_EVIDENCE_CHOICE
    assert choose(MAX_TOOL_ROUNDS, no_title, needs_both, set()) == ("none", ())
    already = {"analyze_current_build", "find_build_alternatives"}
    assert choose(1, no_title, needs_both, already) == ("auto", ())
    assert choose(1, no_title, BuildResponseValidationContext(), set()) == ("auto", ())


def test_evidence_forcing_does_not_change_tool_budgets() -> None:
    assert MAX_TOOL_ROUNDS == 4
    assert MAX_TOOL_CALLS == 10


@pytest.mark.asyncio
async def test_forced_evidence_still_respects_the_tool_round_limit() -> None:
    looping = [[_call("get_character_summary", TITAN, f"loop-{index}")] for index in range(3)]
    looping[0] = [ANALYZE_CALL, FIND_CALL]

    answer, responses, trace = await _chat(
        "Anything better in my vault?", [*looping, [_answer("Best from what I checked: Sunshot.")]]
    )

    assert len(responses.requests) == MAX_TOOL_ROUNDS
    assert answer.message.startswith("Best from what I checked")
    assert trace["tool_loop"]["round_limit_reached"] is True


@pytest.mark.asyncio
async def test_remaining_evidence_violation_still_uses_safe_build_answer() -> None:
    draft = "Use the hand cannon in your vault? Or keep your current one?"

    answer, responses, trace = await _chat(
        "Anything better in my vault?", [[_answer(draft, 1)], [_answer(draft, 2)]]
    )

    assert answer.message == (
        "I couldn't retrieve a grounded shortlist of your owned gear and its returned perk "
        "data, so I don't want to guess."
    )
    assert answer.sources == []
    assert len(responses.requests) == 2
    assert responses.requests[1]["tool_choice"] == "none"
    assert trace["planning_correction"]["fallback"] == "build_evidence"
    remaining = trace["planning_correction"]["remaining_violation_codes"]
    assert "missing_owned_inventory_evidence" in remaining


@pytest.mark.parametrize(
    ("message", "draft", "code"),
    [
        ("Is my Titan build good?", "Build score: 82/100. This is A Tier.", "fake_build_score"),
        (
            "Build around Sunshot for solo PvE.",
            "Replace Sunshot with Gjallarhorn for more damage.",
            "locked_item_replaced",
        ),
    ],
)
@pytest.mark.asyncio
async def test_grounding_violation_stays_blocked_with_accurate_wording(
    message: str, draft: str, code: str
) -> None:
    answer, responses, trace = await _chat(
        message, [[ANALYZE_CALL], [_answer(draft, 1)], [_answer(draft, 2)]]
    )

    assert answer.message == ResponseQualityValidator.safe_build_grounding_answer()
    assert draft not in answer.message
    assert answer.sources == []
    assert responses.requests[2]["tool_choice"] == "none"
    assert trace["planning_correction"]["fallback"] == "build_grounding"
    assert code in trace["planning_correction"]["remaining_violation_codes"]


@pytest.mark.asyncio
async def test_style_only_violation_after_correction_returns_corrected_text() -> None:
    draft = "Your Sunshot setup suits solo PvE. Want add-clear tips? Or boss tips?"
    corrected = "Your Sunshot setup suits solo PvE. Want add-clear tips? Or boss damage tips?"

    answer, responses, trace = await _chat(
        "Is my Titan build good?",
        [[ANALYZE_CALL], [_answer(draft, 1)], [_answer(corrected, 2)]],
    )

    assert answer.message == corrected
    assert len(responses.requests) == 3
    assert responses.requests[2]["tool_choice"] == "none"
    assert trace["planning_correction"]["succeeded"] is False
    assert trace["planning_correction"]["fallback"] == "style_only_accepted"
    assert trace["planning_correction"]["remaining_violation_codes"] == ["too_many_followups"]


def test_build_fallback_wording_never_claims_retrieval_failed_for_grounding() -> None:
    wording = ResponseQualityValidator.safe_build_grounding_answer().casefold()

    assert "retrieve" not in wording
    assert "inspect" not in wording
    assert "couldn't put together a recommendation i could fully verify" in wording


def test_every_validator_code_has_exactly_one_explicit_fallback_group() -> None:
    sources = [
        Path(response_quality.__file__).read_text(encoding="utf-8"),
        Path(session_planning.__file__).read_text(encoding="utf-8"),
    ]
    emitted = {code for source in sources for code in re.findall(r'code="([a-z_]+)"', source)}
    groups = [EVIDENCE_VIOLATION_CODES, STYLE_VIOLATION_CODES, GROUNDING_SAFETY_VIOLATION_CODES]

    assert emitted == set().union(*groups)
    assert sum(len(value) for value in groups) == len(emitted)
    assert violation_group("future_unlisted_check") == "grounding_safety"


def test_build_fallback_kind_precedence_is_evidence_then_grounding_then_style() -> None:
    def kind(*codes: str) -> str:
        return build_fallback_kind([PlanningViolation(code=code, correction="") for code in codes])

    assert kind("too_many_followups", "locked_item_replaced") == "grounding_safety"
    assert kind("extreme_verbosity", "missing_current_build_analysis") == "evidence"
    assert kind("unsupported_ownership_claim", "missing_owned_inventory_evidence") == "evidence"
    assert kind("inventory_dump", "too_many_build_changes") == "style_only"


class _ScriptedStream:
    def __init__(self, response: Any) -> None:
        self.response = response

    async def __aenter__(self) -> "_ScriptedStream":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def __aiter__(self) -> Any:
        async def iterate() -> Any:
            yield SimpleNamespace(type="response.output_text.done")

        return iterate()

    async def get_final_response(self) -> Any:
        return self.response


class ScriptedStreamingBuildResponses(ScriptedBuildResponses):
    """Serve scripted outputs through the SDK streaming interface only."""

    def stream(self, **kwargs: Any) -> _ScriptedStream:
        self.requests.append(kwargs)
        output = self.outputs[len(self.requests) - 1]
        text = "".join(
            part.text for value in output if value.type == "message" for part in value.content
        )
        return _ScriptedStream(
            SimpleNamespace(
                output=output,
                output_text=text,
                status="completed",
                incomplete_details=None,
                error=None,
                usage=None,
            )
        )

    async def create(self, **kwargs: Any) -> Any:
        raise AssertionError("A streaming chat must not use the non-streaming request path.")


@pytest.mark.asyncio
async def test_streaming_path_receives_forced_evidence_tool_choices_and_completes() -> None:
    responses = ScriptedStreamingBuildResponses(
        [[ANALYZE_CALL], [FIND_CALL], [_answer(GROUNDED_VAULT_ANSWER)]]
    )
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)
    statuses: list[Any] = []

    async def emit(event: Any) -> None:
        statuses.append(event)

    reporter = StreamStatusReporter(emit)
    response = await service.chat(
        ChatRequest(message="Anything better in my vault?"),
        build_context(),
        stream_status=reporter,
    )
    events = list(response_events(response, reporter.response_mode or "direct_fact"))
    service.mark_stream_delivery(
        reporter.trace_id,
        final_response_streamed=True,
        stream_completed=True,
        emitted_status_categories=list(reporter.emitted_categories),
    )

    assert [value["tool_choice"] for value in responses.requests] == [
        BOTH_EVIDENCE_CHOICE,
        FUNCTION_CHOICE["find_build_alternatives"],
        "auto",
    ]
    assert all(value["parallel_tool_calls"] is True for value in responses.requests)
    assert response.message == GROUNDED_VAULT_ANSWER
    assert "loadout" in reporter.emitted_categories
    assert "inventory" in reporter.emitted_categories
    assert reporter.emitted_categories[-1] == "final"
    assert [event.stage for event in statuses][-1] == "final"
    deltas = [event.delta for event in events if isinstance(event, MessageDeltaStreamEvent)]
    assert "".join(deltas) == GROUNDED_VAULT_ANSWER
    assert isinstance(events[-1], CompletedStreamEvent)
    assert events[-1].response_mode == "build_advice"
    trace = service.latest_trace()
    assert trace["streaming"] is True
    assert trace["stream_completed"] is True
    assert trace["final_response_streamed"] is True
    assert trace["build_analysis"]["forced_evidence_tools"] == [
        "analyze_current_build",
        "find_build_alternatives",
    ]
    assert trace["planning_correction"]["attempted"] is False


def _find(call_id: str, *, slot: str | None = None, subtype: str | None = None) -> Any:
    item_type = "Armor" if slot and "Weapons" not in slot else "Weapon"
    return _call(
        "find_build_alternatives",
        {**FIND_CALL_ARGUMENTS, "slot": slot, "subtype": subtype, "item_type": item_type},
        call_id,
    )


FIND_CALL_ARGUMENTS = json.loads(FIND_CALL.arguments)
ROCKET_RESULT = {"candidates": [{"subtype": "Rocket Launcher", "slot": "Power Weapons"}]}
HAND_CANNON_RESULT = {"candidates": [{"subtype": "Hand Cannon", "slot": "Energy Weapons"}]}


@pytest.mark.parametrize(
    ("message", "arguments", "result", "expected"),
    [
        ("What rocket launcher should I use?", {"subtype": "Rocket Launcher"}, {}, "match"),
        ("Which rocket launchers do I own?", {"subtype": "Rocket Launcher"}, {}, "match"),
        ("What rocket launcher should I use?", {"slot": "Power Weapons"}, ROCKET_RESULT, "match"),
        ("Best power weapon for me?", {"slot": "Power Weapons"}, {}, "match"),
        (
            "What rocket launcher should I use?",
            {"subtype": "Hand Cannon"},
            HAND_CANNON_RESULT,
            "mismatch",
        ),
        (
            "What rocket launcher should I use?",
            {"slot": None, "subtype": None},
            HAND_CANNON_RESULT,
            "mismatch",
        ),
        ("Anything better in my vault?", {"subtype": "Rocket Launcher"}, {}, "not_applicable"),
        ("Rocket launcher or hand cannon?", {"subtype": "Hand Cannon"}, {}, "not_applicable"),
        # Shorthand is interpreted only through alias_terms from the core knowledge engine.
        ("Is my HC any good?", {"subtype": "Rocket Launcher"}, {}, "not_applicable"),
        (
            "What rocket launcher should I use?",
            {"subtype": "Rocket Launcher"},
            {"error": "lookup failed"},
            "not_applicable",
        ),
    ],
)
def test_forced_alternatives_relevance_uses_only_owned_slot_and_subtype_vocabulary(
    message: str, arguments: dict[str, Any], result: dict[str, Any], expected: str
) -> None:
    assert forced_alternatives_relevance(message, build_context(), arguments, result) == expected


@pytest.mark.asyncio
async def test_relevance_signal_is_trace_only_and_never_changes_the_answer() -> None:
    message = "Do I own a better rocket launcher than my current one in my vault?"
    answer_text = "Apex Predator with Reconstruction in your vault is your best owned option."
    runs = []
    for find_call in (
        _find("find-match", subtype="Rocket Launcher"),
        _find("find-mismatch", subtype="Hand Cannon"),
    ):
        runs.append(await _chat(message, [[ANALYZE_CALL, find_call], [_answer(answer_text)]]))

    matched, matched_requests, matched_trace = runs[0]
    mismatched, mismatched_requests, mismatched_trace = runs[1]
    assert matched_trace["build_analysis"]["forced_alternatives_relevance"] == ["match"]
    assert mismatched_trace["build_analysis"]["forced_alternatives_relevance"] == ["mismatch"]
    assert matched.message == mismatched.message == answer_text
    assert matched.sources == mismatched.sources
    assert [value["tool_choice"] for value in matched_requests.requests] == [
        value["tool_choice"] for value in mismatched_requests.requests
    ]
    assert matched_trace["planning_correction"] == mismatched_trace["planning_correction"]
    assert "Hand Cannon" not in json.dumps(mismatched_trace["build_analysis"])


@pytest.mark.asyncio
async def test_unforced_alternatives_calls_get_no_relevance_signal() -> None:
    _, _, trace = await _chat(
        "Is my Titan build good?",
        [[ANALYZE_CALL], [_find("unforced", subtype="Rocket Launcher")], [_answer("Solid.")]],
    )

    assert trace["build_analysis"]["forced_alternatives_relevance"] == []


FULL_LOADOUT = "Give me a complete detailed loadout from what I own for my Titan."
WEAPON_SLOTS = ("Kinetic Weapons", "Energy Weapons", "Power Weapons")
ARMOR_SLOTS = ("Helmet", "Gauntlets", "Chest Armor", "Leg Armor", "Class Armor")


@pytest.mark.asyncio
async def test_full_loadout_tool_pressure_is_observable_without_budget_changes() -> None:
    answer, responses, trace = await _chat(
        FULL_LOADOUT,
        [
            [ANALYZE_CALL, *(_find(f"weapon-{slot}", slot=slot) for slot in WEAPON_SLOTS)],
            [_find(f"armor-{slot}", slot=slot) for slot in ARMOR_SLOTS],
            [_answer("Keep Sunshot and Hallowfire Heart, and run Apex Predator in Power.")],
        ],
    )

    # A slot-by-slot full loadout uses 9 of 10 calls and 3 of 4 rounds when the model
    # batches perfectly; any extra lookup or a less parallel plan reaches a limit.
    assert answer.message.startswith("Keep Sunshot")
    assert len(responses.requests) == 3
    assert trace["tool_budget"] == {
        "max_rounds": 4,
        "max_tool_calls": 10,
        "rounds_used": 3,
        "tool_calls_used": 9,
        "limit_reached": None,
    }


@pytest.mark.asyncio
async def test_full_loadout_that_exceeds_the_call_budget_answers_from_evidence() -> None:
    answer, responses, trace = await _chat(
        FULL_LOADOUT,
        [
            [ANALYZE_CALL, *(_find(f"weapon-{slot}", slot=slot) for slot in WEAPON_SLOTS)],
            [
                *(_find(f"armor-{slot}", slot=slot) for slot in ARMOR_SLOTS),
                _call("get_build_details", TITAN, "subclass-details"),
                _call("get_equipped_loadout", TITAN, "loadout-details"),
            ],
            [_answer("Weapons are verified; I could not check every armor slot.")],
        ],
    )

    assert answer.message.startswith("Weapons are verified")
    assert trace["tool_budget"]["limit_reached"] == "tool_calls"
    assert trace["tool_budget"]["tool_calls_used"] == MAX_TOOL_CALLS
    assert trace["tool_budget"]["rounds_used"] == 3
    assert responses.requests[2]["tool_choice"] == "none"
    skipped = [
        value
        for value in responses.requests[2]["input"]
        if isinstance(value, dict)
        and value.get("type") == "function_call_output"
        and "tool_budget_exhausted" in value["output"]
    ]
    assert len(skipped) == 1
    assert trace["tool_loop"]["tool_limit_reached"] is True
    assert trace["tool_loop"]["round_limit_reached"] is False


@pytest.mark.asyncio
async def test_sequential_full_loadout_that_exhausts_rounds_still_answers() -> None:
    answer, responses, trace = await _chat(
        FULL_LOADOUT,
        [
            [ANALYZE_CALL],
            [_find("weapon-kinetic", slot="Kinetic Weapons")],
            [_find("weapon-energy", slot="Energy Weapons")],
            [_answer("Kinetic and Energy are checked; Power Weapons could not be verified.")],
        ],
    )

    assert "could not be verified" in answer.message
    assert len(responses.requests) == MAX_TOOL_ROUNDS
    assert [value["tool_choice"] for value in responses.requests][-1] == "none"
    assert trace["tool_budget"]["limit_reached"] == "rounds"
    assert trace["tool_budget"]["rounds_used"] == MAX_TOOL_ROUNDS
    loop = trace["tool_loop"]
    assert loop["round_limit_reached"] is True
    assert loop["tool_limit_reached"] is False
    assert loop["final_answer_forced"] is True
    assert loop["evidence_at_force"]["tool_calls_made"] == 3
    assert loop["final_answer_succeeded"] is True
    assert loop["final_answer_source"] == "model"


LOOP_MESSAGES = [
    "Is my Titan build good?",
    "Anything better in my vault?",
    "What should I do tonight on my Titan?",
    "How does Sunshot work, and do I own it?",
]


@pytest.mark.parametrize("message", LOOP_MESSAGES)
@pytest.mark.asyncio
async def test_repeated_tool_requests_end_in_a_toolless_grounded_final_round(
    message: str,
) -> None:
    looping = [[_call("get_character_summary", TITAN, f"loop-{index}")] for index in range(3)]
    looping[0] = [ANALYZE_CALL, FIND_CALL]

    answer, responses, trace = await _chat(message, [*looping, [_answer(GROUNDED_VAULT_ANSWER)]])

    assert answer.message
    assert len(responses.requests) == MAX_TOOL_ROUNDS
    final = responses.requests[-1]
    assert final["tool_choice"] == "none"
    assert "no further tool calls are possible" in final["instructions"]
    assert "do not invent" in final["instructions"]
    assert all("no further tool calls" not in v["instructions"] for v in responses.requests[:-1])
    assert trace["tool_loop"]["final_answer_forced"] is True
    assert trace["tool_loop"]["final_answer_succeeded"] is True
    assert trace["tool_loop"]["evidence_at_force"]["tool_calls_made"] == 4


@pytest.mark.asyncio
async def test_tool_calls_in_the_final_round_are_never_executed() -> None:
    answer, responses, trace = await _chat(
        "What should I do tonight on my Titan?",
        [[_call("get_character_summary", TITAN, f"loop-{index}")] for index in range(4)],
    )

    assert answer.message == FINAL_ROUND_LIMITATION_MESSAGE
    assert len(responses.requests) == MAX_TOOL_ROUNDS
    assert trace["tool_budget"]["tool_calls_used"] == 3
    assert trace["tool_loop"]["final_answer_succeeded"] is False
    assert trace["tool_loop"]["final_answer_source"] == "limitation_fallback"


@pytest.mark.asyncio
async def test_insufficient_evidence_final_answer_is_a_limitation_not_invention() -> None:
    limitation = "I could only verify your equipped Titan loadout; I couldn't check the vault."
    looping = [[_call("get_character_summary", TITAN, f"loop-{index}")] for index in range(3)]

    answer, _, trace = await _chat(
        "What should I do tonight on my Titan?", [*looping, [_answer(limitation)]]
    )

    assert answer.message == limitation
    assert trace["tool_loop"]["final_answer_succeeded"] is True


@pytest.mark.parametrize("message", LOOP_MESSAGES)
@pytest.mark.asyncio
async def test_normal_two_and_three_round_flows_never_force_a_final_answer(message: str) -> None:
    _, one, one_trace = await _chat(
        message, [[ANALYZE_CALL, FIND_CALL], [_answer(GROUNDED_VAULT_ANSWER)]]
    )
    _, two, two_trace = await _chat(
        message,
        [
            [ANALYZE_CALL, FIND_CALL],
            [_call("get_character_summary", TITAN, "c1")],
            [_answer(GROUNDED_VAULT_ANSWER)],
        ],
    )

    assert len(one.requests) == 2 and len(two.requests) == 3
    for trace, responses in ((one_trace, one), (two_trace, two)):
        assert trace["tool_loop"]["final_answer_forced"] is False
        assert trace["tool_loop"]["evidence_at_force"] is None
        assert trace["tool_budget"]["limit_reached"] is None
        assert all("no further tool calls" not in v["instructions"] for v in responses.requests)
