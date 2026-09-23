import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.ai import SYSTEM_INSTRUCTIONS, RecommendationService
from app.build_analysis import (
    AnalyzeCurrentBuildRequest,
    BuildAnalysisService,
    FindBuildAlternativesRequest,
    derive_build_request_context,
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
from app.session_preferences import SessionPreferenceContext


def plug(name: str, category: str = "weapon.perks") -> SocketedPlugSummary:
    return SocketedPlugSummary(
        item_hash=abs(hash(name)) % 2**32,
        name=name,
        description=f"Manifest description for {name}.",
        item_type="Trait",
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
) -> ItemSummary:
    values = plugs or []
    return ItemSummary(
        item_hash=abs(hash((name, location, tuple(value.name for value in values)))) % 2**32,
        instance_id=f"private-{name}-{location}",
        name=name,
        item_type=item_type,
        item_subtype=subtype,
        tier=tier,
        bucket_name=bucket,
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
            plug("Thermite Grenade", "shared.grenades"),
            plug("Roaring Flames", "shared.aspects"),
            plug("Ember of Torches", "shared.fragments"),
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
        plugs=[plug("Sun Blast", "intrinsics")],
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
        plugs=[plug("Solar Resistance", "enhancements.mods")],
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
            plugs=[plug("Wolfpack Rounds", "intrinsics")],
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
async def test_build_quality_failure_uses_the_existing_single_correction_retry() -> None:
    class CorrectionResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            text = (
                "Build score: 82/100. This is A Tier."
                if len(self.requests) == 1
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
    assert len(responses.requests) == 2
    assert responses.requests[1]["tool_choice"] == "none"
    trace = service.latest_trace()
    assert trace is not None
    assert trace["planning_correction"] == {
        "attempted": True,
        "violation_codes": ["fake_build_score"],
        "succeeded": True,
    }


def test_build_routing_requires_guardian_ownership_and_external_current_grounding() -> None:
    assert "call analyze_current_build" in SYSTEM_INSTRUCTIONS
    assert "find_build_alternatives" in SYSTEM_INSTRUCTIONS
    assert "Ownership of every immediate alternative MUST" in SYSTEM_INSTRUCTIONS
    assert '"Best DPS," "meta,"' in SYSTEM_INSTRUCTIONS
    assert "require Live or Web grounding" in SYSTEM_INSTRUCTIONS
