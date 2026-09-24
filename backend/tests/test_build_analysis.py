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
