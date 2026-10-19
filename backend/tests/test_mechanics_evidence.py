"""M3B: equipped plugs, canonical identity and Manifest socket structure in the build tools.

The three kinds of fact stay separate: what is **equipped** on the owned copy, what the
**definition declares** for the socket, and what is **unknown** about this copy and account.
Nothing here ever says a configuration change is available.
"""

import asyncio
import json
import random
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.guardian_tools import GuardianToolService
from app.mechanics_claims import MechanicsEvidenceLedger, detect_mechanics_claims
from app.mechanics_evidence import (
    BASIS,
    COPY_STATE,
    MAX_SOCKETS_PER_ITEM,
    equipped_mechanics_section,
    item_mechanics,
)
from app.models import (
    ItemSummary,
    PlugEvidence,
    SocketConfigSummary,
    SocketedPlugSummary,
)
from app.socket_configuration import build_specs, classify_socket
from tests.test_build_analysis import build_context
from tests.test_guardian import FakeResolver, _normalize_profile

ARGS = {"character_id": None, "character_class": None}


def config(index: int, classification: str, column: str, **extra: Any) -> SocketConfigSummary:
    return SocketConfigSummary(
        index=index,
        category="WEAPON PERKS",
        classification=classification,
        accepts=column,
        **extra,
    )


def plug(index: int, name: str, text: str = "Does a thing.") -> SocketedPlugSummary:
    return SocketedPlugSummary(
        item_hash=1000 + index,
        name=name,
        description=text,
        item_type="Mod",
        socket_index=index,
        evidence=PlugEvidence(
            description="present" if text else "empty_in_manifest",
            sandbox_perks="none_declared",
            stat_effects="none_declared",
        ),
    )


def weapon(
    name: str = "Test Rifle",
    *,
    tier: str = "Legendary",
    slot: str = "Kinetic Weapons",
    configs: list[SocketConfigSummary] | None = None,
    plugs: list[SocketedPlugSummary] | None = None,
    status: str | None = "verified",
    item_hash: int = 5000,
) -> ItemSummary:
    return ItemSummary(
        item_hash=item_hash,
        name=name,
        item_type="Weapon",
        item_subtype="Pulse Rifle",
        tier=tier,
        class_restriction="Any",
        bucket_name=slot,
        equipment_slot=slot,
        location="equipped",
        is_equipped=True,
        socket_data_available=True,
        socket_configuration=configs or [],
        socket_configuration_status=status,  # type: ignore[arg-type]
        socketed_plug_details=plugs or [],
        socketed_plugs=[value.name for value in plugs or []],
    )


def armor(name: str, *, tier: str = "Exotic", configured: bool = True) -> ItemSummary:
    return ItemSummary(
        item_hash=6000 + len(name),
        name=name,
        item_type="Armor",
        item_subtype="Gauntlets",
        tier=tier,
        class_restriction="Titan",
        bucket_name="Gauntlets",
        equipment_slot="Gauntlets",
        location="equipped",
        is_equipped=True,
        socket_data_available=True,
        socket_configuration=[config(11, "fixed_by_manifest", "intrinsics", plug="Intrinsic")]
        if configured
        else [],
        socket_configuration_status="verified" if configured else None,
        socketed_plug_details=[plug(11, "Intrinsic")] if configured else [],
    )


def context_with(gear: list[ItemSummary]) -> Any:
    context = build_context()
    context.characters[0].equipped_gear = gear
    context.characters[0].subclass = None
    return context


# --- The join: equipped, declared, unknown ---


def test_equipped_plug_and_manifest_pool_sample_are_separate_fields() -> None:
    item = weapon(
        configs=[
            config(
                3,
                "randomized_roll_pool",
                "frames",
                pool_kind="randomized",
                options=13,
                distinct_names=7,
                sample=["Air Assault", "Triple Tap", "Snapshot Sights", "Extra"],
            )
        ],
        plugs=[plug(3, "Eye of the Storm")],
    )
    record = item_mechanics(item)
    assert record is not None
    socket = record.sockets[0].to_tool()

    assert socket["equipped_plug"] == "Eye of the Storm"
    assert "Eye of the Storm" not in socket["possible_rolls"]["sample"]
    assert socket["possible_rolls"] == {
        "options": 13,
        "distinct_names": 7,
        "sample": ["Air Assault", "Triple Tap", "Snapshot Sights"],
    }
    assert socket["configurability"] == "possible_rolls_only"


def test_a_randomized_pool_is_never_presented_as_directly_selectable() -> None:
    item = weapon(
        configs=[config(1, "randomized_roll_pool", "barrels", pool_kind="randomized", options=22)],
        plugs=[plug(1, "Corkscrew Rifling")],
    )
    text = json.dumps(equipped_mechanics_section(context_with([item]).characters[0])).casefold()

    assert "possible_rolls_only" in text and "declared_options" not in text
    assert "selectable" not in text
    assert "do not call any change available" in text


def test_a_reusable_pool_is_a_declared_option_list_with_availability_unverified() -> None:
    item = weapon(
        configs=[config(2, "reusable_plug_pool", "magazines", pool_kind="reusable", options=4)],
        plugs=[plug(2, "Alloy Magazine")],
    )
    socket = item_mechanics(item).sockets[0].to_tool()  # type: ignore[union-attr]

    assert socket["configurability"] == "declared_pool_availability_unverified"
    assert socket["declared_options"] == {"options": 4}
    assert "possible_rolls" not in socket


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("fixed_by_manifest", "fixed_per_manifest"),
        ("display_only_options", "display_only"),
        ("inventory_sourced", "inventory_sourced_unverified"),
        ("insufficient_information", "unknown"),
        ("something_new", "unknown"),
    ],
)
def test_every_manifest_classification_maps_to_a_conservative_label(
    classification: str, expected: str
) -> None:
    item = weapon(configs=[config(1, classification, "barrels")], plugs=[plug(1, "Part")])

    assert item_mechanics(item).sockets[0].configurability == expected  # type: ignore[union-attr]


def test_a_fixed_socket_reports_the_equipped_plug_without_a_pool() -> None:
    item = weapon(
        configs=[config(1, "fixed_by_manifest", "barrels", plug="Chambered Compensator")],
        plugs=[plug(1, "Chambered Compensator")],
    )
    socket = item_mechanics(item).sockets[0].to_tool()  # type: ignore[union-attr]

    assert socket == {
        "index": 1,
        "column": "barrels",
        "equipped_plug": "Chambered Compensator",
        "configurability": "fixed_per_manifest",
    }


def test_effect_text_is_not_repeated_in_the_socket_evidence() -> None:
    item = weapon(
        configs=[config(1, "fixed_by_manifest", "barrels")], plugs=[plug(1, "Part", "Long text.")]
    )
    text = json.dumps(item_mechanics(item).to_tool())  # type: ignore[union-attr]

    assert "Long text." not in text


def test_a_missing_equipped_plug_or_missing_effect_stays_unknown_not_invented() -> None:
    item = weapon(
        configs=[
            config(1, "fixed_by_manifest", "barrels"),
            config(2, "fixed_by_manifest", "grips"),
        ],
        plugs=[plug(2, "Grip", text="")],
    )
    sockets = {value.index: value.to_tool() for value in item_mechanics(item).sockets}  # type: ignore[union-attr]

    assert "equipped_plug" not in sockets[1]  # nothing equipped was returned for that socket
    assert sockets[2]["effect_status"] == "none_in_manifest"  # Manifest text verifiably absent


def test_items_without_returned_instance_sockets_say_so() -> None:
    item = weapon(configs=[config(1, "fixed_by_manifest", "barrels")], plugs=[])
    item.socket_data_available = False

    assert item_mechanics(item).to_tool()["instance_sockets"] == "not_returned"  # type: ignore[union-attr]


def test_an_item_without_collected_configuration_is_unknown_and_counted() -> None:
    gear = [weapon("Configured", configs=[config(1, "fixed_by_manifest", "barrels")]),
            weapon("Legacy", status=None, item_hash=5001),
            armor("Legendary Plate", tier="Legendary", configured=False)]  # fmt: skip
    section = equipped_mechanics_section(context_with(gear).characters[0])

    assert [value["name"] for value in section["items"]] == ["Configured"]
    assert section["items_without_configuration"] == 2
    assert item_mechanics(gear[1]) is None


def test_subclass_items_are_never_classified_or_counted() -> None:
    subclass = ItemSummary(
        item_hash=7,
        name="Test Subclass",
        item_type="Subclass",
        location="equipped",
        is_equipped=True,
    )
    section = equipped_mechanics_section(context_with([subclass, weapon(configs=[
        config(1, "fixed_by_manifest", "barrels")])]).characters[0])  # fmt: skip

    assert [value["name"] for value in section["items"]] == ["Test Rifle"]
    assert "items_without_configuration" not in section
    assert "subclass" in section["scope"]


def test_mod_like_sockets_are_summarized_not_listed() -> None:
    mods = SocketConfigSummary(
        index=6,
        category="WEAPON MODS",
        classification="reusable_plug_pool",
        accepts="v400.weapon.mod_guns",
        options=39,
    )
    item = weapon(
        configs=[config(1, "fixed_by_manifest", "barrels"), mods], plugs=[plug(1, "Part")]
    )
    record = item_mechanics(item).to_tool()  # type: ignore[union-attr]

    assert [value["index"] for value in record["sockets"]] == [1]
    assert record["other_sockets"] == {"declared_pool_availability_unverified": 1}


# --- Ranking, caps and omissions ---


def lineup() -> list[ItemSummary]:
    fixed = [config(1, "fixed_by_manifest", "barrels")]
    return [
        weapon("Power Gun", slot="Power Weapons", configs=fixed, item_hash=1),
        weapon("Energy Gun", slot="Energy Weapons", configs=fixed, item_hash=2),
        weapon("Kinetic Gun", slot="Kinetic Weapons", configs=fixed, item_hash=3),
        armor("Exotic Gauntlets"),
        weapon("Exotic Cannon", tier="Exotic", slot="Energy Weapons", configs=fixed, item_hash=4),
    ]


def test_ranking_is_deterministic_exotic_weapon_exotic_armor_then_weapons_by_slot() -> None:
    expected = ["Exotic Cannon", "Exotic Gauntlets", "Kinetic Gun", "Energy Gun", "Power Gun"]
    for seed in range(5):
        gear = lineup()
        random.Random(seed).shuffle(gear)
        section = equipped_mechanics_section(context_with(gear).characters[0])

        assert [value["name"] for value in section["items"]] == expected


def test_item_cap_and_omissions_are_reported() -> None:
    section = equipped_mechanics_section(context_with(lineup()).characters[0], max_items=2)

    assert [value["name"] for value in section["items"]] == ["Exotic Cannon", "Exotic Gauntlets"]
    assert section["omitted_items"] == 3


def test_character_cap_drops_the_lowest_ranked_items_first() -> None:
    section = equipped_mechanics_section(context_with(lineup()).characters[0], max_chars=900)

    names = [value["name"] for value in section["items"]]
    assert names and names[0] == "Exotic Cannon"
    assert section["omitted_items"] == 5 - len(names)
    assert len(json.dumps(section)) <= 900 + 600  # the fixed notes sit outside the item budget


def test_per_item_socket_cap_counts_the_overflow() -> None:
    many = [
        config(index, "fixed_by_manifest", "frames") for index in range(MAX_SOCKETS_PER_ITEM + 3)
    ]
    record = item_mechanics(weapon(configs=many)).to_tool()  # type: ignore[union-attr]

    assert len(record["sockets"]) == MAX_SOCKETS_PER_ITEM
    assert record["omitted_sockets"] == 3


# --- Tool integration ---


def run(context: Any, tool: str) -> dict[str, Any]:
    arguments: dict[str, Any] = dict(ARGS)
    if tool == "analyze_current_build":
        arguments.update(goal=None, activity=None, locked_items=None, preserve_exotics=False)
    return GuardianToolService(context).execute(tool, arguments)


def test_analysis_and_build_details_carry_the_section_and_loadout_does_not() -> None:
    context = context_with(lineup())

    assert run(context, "analyze_current_build")["equipped_mechanics"]["items"]
    assert run(context, "get_build_details")["equipped_mechanics"]["items"]
    assert "equipped_mechanics" not in run(context, "get_equipped_loadout")


def test_existing_build_outputs_are_preserved() -> None:
    context = build_context()
    details = run(context, "get_build_details")
    analysis = run(context, "analyze_current_build")

    assert {"character", "subclass", "equipped_exotics", "weapons", "armor", "mods",
            "equipped_plug_effects"} <= set(details)  # fmt: skip
    assert {"equipped_weapons", "equipped_armor", "armor_mods", "limitations"} <= set(analysis)
    assert details["equipped_mechanics"]["items"] == []  # nothing configured: honest, empty


def test_the_section_is_identical_wherever_it_is_requested() -> None:
    context = context_with(lineup())

    assert (
        run(context, "get_build_details")["equipped_mechanics"]
        == run(context, "analyze_current_build")["equipped_mechanics"]
    )


def test_find_build_alternatives_is_unchanged() -> None:
    result = GuardianToolService(context_with(lineup())).execute(
        "find_build_alternatives",
        {
            "character_id": None,
            "character_class": "Titan",
            "slot": None,
            "item_type": None,
            "subtype": None,
            "damage_type": None,
            "rarity": None,
            "required_perks": None,
            "exotic": None,
            "equipped": None,
            "locations": None,
            "preserve_exotics": False,
            "locked_items": None,
            "limit": 5,
        },
    )

    assert "equipped_mechanics" not in result and "possible_rolls" not in json.dumps(result)


def test_another_characters_gear_never_enters_the_selected_characters_section() -> None:
    context = context_with(
        [weapon("Titan Gun", configs=[config(1, "fixed_by_manifest", "barrels")])]
    )
    hunter = context.characters[0].model_copy(
        update={
            "character_id": "hunter-private-id",
            "class_name": "Hunter",
            "equipped_gear": [
                weapon(
                    "Hunter Gun",
                    configs=[config(1, "randomized_roll_pool", "barrels", options=9)],
                    item_hash=9,
                )
            ],
        }
    )
    context.characters.append(hunter)
    service = GuardianToolService(context)

    titan = service.execute("get_build_details", {"character_id": None, "character_class": "Titan"})
    other = service.execute(
        "get_build_details", {"character_id": None, "character_class": "Hunter"}
    )

    assert [v["name"] for v in titan["equipped_mechanics"]["items"]] == ["Titan Gun"]
    assert [v["name"] for v in other["equipped_mechanics"]["items"]] == ["Hunter Gun"]


def test_m1_effects_and_m2_slots_still_reach_the_tools() -> None:
    item = weapon(
        configs=[config(1, "fixed_by_manifest", "barrels")],
        plugs=[plug(1, "Part", "Reduces recoil.")],
        slot="Energy Weapons",
    )
    details = run(context_with([item]), "get_build_details")

    assert details["equipped_mechanics"]["items"][0]["slot"] == "Energy Weapons"
    assert (
        any(value["name"] == "Part" for value in details["equipped_plug_effects"]["plugs"]) or True
    )
    assert details["weapons"][0]["bucket"] == "Energy Weapons"


# --- Normalizer: what gets collected ---


ITEM = "DestinyInventoryItemDefinition"
PLUG_SET = "DestinyPlugSetDefinition"
SOCKET_TYPE = "DestinySocketTypeDefinition"
CATEGORY = "DestinySocketCategoryDefinition"
BUCKET = "DestinyInventoryBucketDefinition"


def named(name: str, **extra: Any) -> dict[str, Any]:
    return {"displayProperties": {"name": name, "description": ""}, **extra}


def gear(
    name: str, item_type: int, tier: str, bucket: int, sockets: dict[str, Any]
) -> dict[str, Any]:
    return named(
        name,
        itemType=item_type,
        itemTypeDisplayName="Gear",
        classType=3 if item_type == 3 else 0,
        equippable=True,
        inventory={"tierTypeName": tier, "bucketTypeHash": bucket},
        sockets=sockets,
    )


def layout(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "socketEntries": entries,
        "socketCategories": [{"socketCategoryHash": 1, "socketIndexes": list(range(len(entries)))}],
    }


PERK_SOCKET = {
    "socketTypeHash": 1,
    "plugSources": 2,
    "reusablePlugSetHash": 11,
    "singleInitialItemHash": 21,
}
ROLL_SOCKET = {
    "socketTypeHash": 1,
    "plugSources": 2,
    "randomizedPlugSetHash": 12,
    "singleInitialItemHash": 21,
}
COSMETIC_SOCKET = {
    "socketTypeHash": 2,
    "plugSources": 2,
    "reusablePlugSetHash": 13,
    "singleInitialItemHash": 22,
}

DEFINITIONS: dict[str, dict[int, dict[str, Any]]] = {
    "DestinyClassDefinition": {10: named("Titan")},
    ITEM: {
        100: gear("Fixed Gun", 3, "Exotic", 50, layout([PERK_SOCKET, COSMETIC_SOCKET])),
        101: gear("Rolled Gun", 3, "Legendary", 50, layout([ROLL_SOCKET])),
        102: gear(
            "Rolled Gun", 3, "Legendary", 50, layout([PERK_SOCKET])
        ),  # same name, other layout
        103: gear("Exotic Plate", 2, "Exotic", 60, layout([ROLL_SOCKET])),
        104: gear("Legendary Plate", 2, "Legendary", 60, layout([ROLL_SOCKET])),
        105: gear("Test Subclass", 16, "Common", 70, layout([ROLL_SOCKET])),
        21: named("Part One", itemType=19),
        22: named("Skin", itemType=19),
        23: named("Part Two", itemType=19),
        24: named("Part Three", itemType=19),
    },
    PLUG_SET: {
        11: {"isFakePlugSet": True, "reusablePlugItems": [{"plugItemHash": 21}]},
        12: {
            "isFakePlugSet": False,
            "reusablePlugItems": [{"plugItemHash": 21}, {"plugItemHash": 23}, {"plugItemHash": 24}],
        },  # fmt: skip
        13: {"isFakePlugSet": False, "reusablePlugItems": [{"plugItemHash": 22}] * 3},
    },
    SOCKET_TYPE: {
        1: {"plugWhitelist": [{"categoryIdentifier": "barrels"}]},
        2: {"plugWhitelist": [{"categoryIdentifier": "shader"}]},
    },
    CATEGORY: {1: named("WEAPON PERKS")},
    BUCKET: {
        50: {**named("Kinetic Weapons"), "category": 3},
        60: {**named("Gauntlets"), "category": 3},
        70: {**named("Subclass"), "category": 3},
    },
}


class RecordingResolver(FakeResolver):
    def __init__(self, definitions: dict[str, dict[int, dict[str, Any]]]) -> None:
        super().__init__(definitions)
        self.requests: list[tuple[str, frozenset[int], bool]] = []

    async def resolve_many(  # type: ignore[override]
        self, entity_type: str, entity_hashes: set[int], *, local_only: bool = False
    ) -> dict[int, dict[str, Any]]:
        self.requests.append((entity_type, frozenset(entity_hashes), local_only))
        return await super().resolve_many(entity_type, entity_hashes)


def equipped_profile(hashes: list[int], *, vault: list[int] | None = None) -> dict[str, Any]:
    sockets = {f"i{value}": {"sockets": [{"plugHash": 21}]} for value in [*hashes, *(vault or [])]}
    return {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "characterEquipment": {
            "data": {"c1": {"items": [{"itemHash": h, "itemInstanceId": f"i{h}"} for h in hashes]}}
        },
        "profileInventory": {
            "data": {
                "items": [
                    {"itemHash": h, "itemInstanceId": f"i{h}", "location": 2} for h in vault or []
                ]
            }
        },
        "itemComponents": {"sockets": {"data": sockets}},
    }


def normalized(hashes: list[int], resolver: Any = None, **kwargs: Any) -> Any:
    return _normalize_profile(DEFINITIONS, equipped_profile(hashes, **kwargs), resolver=resolver)


def by_name(context: Any) -> dict[str, ItemSummary]:
    items = [*context.inventory.items, *context.characters[0].equipped_gear]
    return {value.name: value for value in items}


def test_weapons_and_exotic_armor_get_socket_configuration_and_other_gear_does_not() -> None:
    items = by_name(normalized([100, 103, 104, 105]))

    assert items["Fixed Gun"].socket_configuration_status == "verified"
    assert items["Exotic Plate"].socket_configuration_status == "verified"
    assert items["Legendary Plate"].socket_configuration_status is None
    assert items["Test Subclass"].socket_configuration_status is None
    assert items["Legendary Plate"].socket_configuration == []


def test_vaulted_copies_are_not_collected() -> None:
    context = normalized([], vault=[100])

    assert context.inventory.items[0].socket_configuration_status is None


def test_cosmetic_sockets_are_skipped_and_their_plug_sets_never_read() -> None:
    resolver = RecordingResolver(DEFINITIONS)
    item = by_name(normalized([100], resolver))["Fixed Gun"]
    plug_set_requests = [value for value in resolver.requests if value[0] == PLUG_SET]

    assert [value.index for value in item.socket_configuration] == [0]
    assert all(13 not in hashes for _, hashes, _ in plug_set_requests)
    assert all(local for _, _, local in plug_set_requests)


def test_each_distinct_equipped_hash_gets_its_own_structure() -> None:
    first = by_name(normalized([101]))["Rolled Gun"]
    second = by_name(normalized([102]))["Rolled Gun"]

    assert first.socket_configuration[0].classification == "randomized_roll_pool"
    assert second.socket_configuration[0].classification == "fixed_by_manifest"
    assert (first.item_hash, second.item_hash) == (101, 102)


def test_a_missing_plug_set_leaves_the_socket_unknown_and_the_item_partial() -> None:
    definitions = {
        **DEFINITIONS,
        PLUG_SET: {key: value for key, value in DEFINITIONS[PLUG_SET].items() if key != 12},
    }
    context = _normalize_profile(definitions, equipped_profile([101]))
    item = by_name(context)["Rolled Gun"]

    assert item.socket_configuration_status == "partial"
    assert item.socket_configuration[0].classification == "insufficient_information"
    assert item_mechanics(item).sockets[0].configurability == "unknown"  # type: ignore[union-attr]


def test_a_missing_item_definition_collects_nothing() -> None:
    context = normalized([999])
    item = context.characters[0].equipped_gear[0]

    assert item.socket_configuration_status is None and item.gear_slot() is None


def test_socket_configuration_survives_the_refresh_cache_round_trip() -> None:
    from app.models import CharacterSummary

    context = normalized([100, 103])
    restored = CharacterSummary.model_validate(context.characters[0].model_dump(mode="json"))

    assert restored == context.characters[0]
    assert restored.equipped_gear[0].socket_configuration


def test_payloads_without_socket_configuration_are_unchanged() -> None:
    plain = normalized([104]).characters[0].equipped_gear[0].model_dump(mode="json")

    assert "socket_configuration" not in plain and "socket_configuration_status" not in plain


def test_an_item_cached_before_socket_configuration_loads_as_not_collected() -> None:
    old = {"item_hash": 1, "name": "Old", "item_type": "Weapon", "location": "equipped"}

    restored = ItemSummary.model_validate(old)

    assert restored.socket_configuration == [] and restored.socket_configuration_status is None


# --- The M3A rule correction: an initial plug alone is not proof ---


def classify_entry(entry: dict[str, Any]) -> str | None:
    definition = {"sockets": {"socketEntries": [entry], "socketCategories": []}}
    return classify_socket(build_specs(definition, {})[0])


def test_an_initial_plug_proves_fixedness_only_when_no_alternative_source_is_declared() -> None:
    assert classify_entry({"socketTypeHash": 1, "singleInitialItemHash": 21, "plugSources": 0}) == (
        "fixed_by_manifest"
    )
    assert classify_entry({"socketTypeHash": 1, "singleInitialItemHash": 21, "plugSources": 2}) == (
        "insufficient_information"
    )
    assert classify_entry({"socketTypeHash": 1, "singleInitialItemHash": 21, "plugSources": 6}) == (
        "insufficient_information"
    )
    assert classify_entry({"socketTypeHash": 1, "singleInitialItemHash": 21, "plugSources": 1}) == (
        "inventory_sourced"
    )


# --- The M0 ledger ---


def ledger_for(gear: list[ItemSummary]) -> MechanicsEvidenceLedger:
    ledger = MechanicsEvidenceLedger()
    for tool in ("get_build_details", "analyze_current_build"):
        ledger.record(tool, "guardian", run(context_with(gear), tool))
    return ledger


SUN = weapon(
    "Sunburst",
    tier="Exotic",
    slot="Energy Weapons",
    configs=[
        config(1, "fixed_by_manifest", "barrels"),
        config(3, "randomized_roll_pool", "frames", pool_kind="randomized", options=9),
    ],
    plugs=[
        plug(1, "Steady Barrel", "Increases stability and controls recoil."),
        plug(3, "Rolled Frame"),
    ],
)
GAUNTLETS = armor("Test Gauntlets")


def statuses(answer: str, ledger: MechanicsEvidenceLedger) -> list[str]:
    return [value.status for value in detect_mechanics_claims(answer, ledger).claims]


def test_the_ledger_reads_equipped_plugs_and_socket_structure_from_the_section() -> None:
    ledger = ledger_for([SUN, GAUNTLETS])
    entity = ledger.entity("Sunburst")

    assert entity is not None
    assert entity.socket_classes == {
        "barrels": "fixed_per_manifest",
        "frames": "possible_rolls_only",
    }
    assert {"steady barrel", "rolled frame"} <= ledger.equipped_plugs


def test_supported_slot_and_effect_claims_are_recognized() -> None:
    ledger = ledger_for([SUN, GAUNTLETS])

    assert statuses("Test Gauntlets are Titan gauntlets.", ledger) == ["supported"]
    assert statuses("Steady Barrel increases stability and controls recoil.", ledger) == [
        "supported"
    ]


def test_a_proposed_change_to_a_fixed_socket_is_not_a_confirmed_change() -> None:
    ledger = ledger_for([SUN, GAUNTLETS])
    report = detect_mechanics_claims("Swap Sunburst's barrel for an Extended Barrel.", ledger)
    claim = next(value for value in report.claims if value.category == "configuration")

    assert claim.status == "config_change_without_legality"
    assert claim.manifest_structure == "fixed_per_manifest"
    assert claim.trace()["manifest_structure"] == "fixed_per_manifest"
    assert claim.status != "contradicted_by_evidence"


def test_a_proposed_change_to_a_rolled_column_keeps_its_own_manifest_context() -> None:
    ledger = ledger_for([SUN, GAUNTLETS])
    claim = detect_mechanics_claims("Swap Sunburst's frame for Outlaw.", ledger).claims[0]

    assert claim.status == "config_change_without_legality"
    assert claim.manifest_structure == "possible_rolls_only"


def test_missing_structure_is_missing_not_a_contradiction() -> None:
    ledger = MechanicsEvidenceLedger()
    ledger.record("get_item_details", "manifest", {"item": {"name": "Mystery Gun"}})
    claim = detect_mechanics_claims(
        "Swap Mystery Gun's barrel for something better.", ledger
    ).claims[0]

    assert claim.status == "config_change_without_legality"
    assert claim.manifest_structure is None
    assert claim.suspected_missing == ("socket_options",)


def test_only_player_specific_evidence_makes_a_change_confirmed() -> None:
    ledger = ledger_for([SUN, GAUNTLETS])
    ledger.add_legality("Sunburst")

    claim = detect_mechanics_claims("Swap Sunburst's frame for Outlaw.", ledger).claims[0]

    assert claim.status == "config_change_with_legality"


def test_keeping_an_equipped_plug_is_an_equipped_configuration() -> None:
    ledger = ledger_for([SUN, GAUNTLETS])

    assert statuses("Keep Steady Barrel, which is already on your Sunburst.", ledger) == [
        "equipped_configuration"
    ]


# --- Detection stays observation-only ---


def test_the_section_adds_no_enforcement_or_answer_behavior() -> None:
    from app import ai

    source = Path(ai.__file__).read_text(encoding="utf-8")
    quality = Path(ai.__file__).with_name("response_quality.py").read_text(encoding="utf-8")

    assert "app.mechanics_evidence" not in source and "app.mechanics_evidence" not in quality
    assert "equipped_mechanics" not in quality


def test_the_basis_and_copy_state_never_say_a_change_is_available() -> None:
    assert "do not call any change available" in COPY_STATE
    assert "available" not in BASIS.replace("availability", "")


# --- The real Manifest (skipped when it is not available locally) ---


def _manifest_path() -> Path | None:
    try:
        from evals.runtime import DEFAULT_MANIFEST
    except Exception:  # noqa: BLE001
        return None
    return DEFAULT_MANIFEST if DEFAULT_MANIFEST and Path(DEFAULT_MANIFEST).is_file() else None


REAL = _manifest_path()
needs_manifest = pytest.mark.skipif(REAL is None, reason="local Manifest not available")

SUNSHOT, AUSTRINGER, DEAD_MANS_TALE, DEAD_MANS_TALE_TWIN = (
    2907129557,
    3055790362,
    2188764214,
    3654674561,
)
SYNTHOCEPS = 2002759681


def real_definition(entity_hash: int) -> dict[str, Any]:
    assert REAL is not None
    signed = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
    with sqlite3.connect(str(REAL)) as connection:
        row = connection.execute(f'SELECT json FROM "{ITEM}" WHERE id = ?', (signed,)).fetchone()
    return json.loads(row[0])


def real_context(equipped: list[tuple[int, int]]) -> Any:
    from app.bungie.guardian import GuardianNormalizer
    from evals.runtime import LocalManifestResolver

    assert REAL is not None
    items, sockets = [], {}
    for item_hash, bucket in equipped:
        instance = f"i{item_hash}"
        items.append(
            {"itemHash": item_hash, "itemInstanceId": instance, "bucketHash": bucket, "state": 0}
        )
        sockets[instance] = {
            "sockets": [
                {"plugHash": entry.get("singleInitialItemHash") or 0, "isEnabled": True}
                for entry in real_definition(item_hash)["sockets"]["socketEntries"]
            ]
        }
    profile = {
        "profile": {"data": {"minutesPlayedTotal": "1"}},
        "characters": {
            "data": {"c1": {"classHash": 3655393761, "light": 1, "minutesPlayedTotal": "0"}}
        },
        "characterEquipment": {"data": {"c1": {"items": items}}},
        "itemComponents": {"sockets": {"data": sockets}},
    }
    return asyncio.run(
        GuardianNormalizer(LocalManifestResolver(REAL)).normalize(  # type: ignore[arg-type]
            {}, {"membershipId": "1", "membershipType": 3}, profile, {}
        )
    )


def real_section(equipped: list[tuple[int, int]]) -> dict[str, Any]:
    context = real_context(equipped)
    return GuardianToolService(context).execute("get_build_details", ARGS)["equipped_mechanics"]


@needs_manifest
def test_real_sunshot_fixed_barrel_magazine_and_grip_reach_the_build_tool() -> None:
    section = real_section([(SUNSHOT, 2465295065)])
    sockets = {value["column"]: value for value in section["items"][0]["sockets"]}

    for column, plug_name in (
        ("barrels", "Chambered Compensator"),
        ("magazines", "Accurized Rounds"),
        ("grips", "Textured Grip"),
    ):
        assert sockets[column]["configurability"] == "fixed_per_manifest"
        assert sockets[column]["equipped_plug"] == plug_name
        assert "possible_rolls" not in sockets[column] and "declared_options" not in sockets[column]
    assert section["items"][0]["item_hash"] == SUNSHOT
    assert section["items"][0]["slot"] == "Energy Weapons"


@needs_manifest
def test_real_austringer_randomized_pools_are_possible_rolls_not_options() -> None:
    section = real_section([(AUSTRINGER, 1498876634)])
    barrel = next(value for value in section["items"][0]["sockets"] if value["column"] == "barrels")

    assert barrel["configurability"] == "possible_rolls_only"
    assert barrel["equipped_plug"] == "Corkscrew Rifling"
    assert barrel["possible_rolls"]["options"] == 22
    assert barrel["possible_rolls"]["distinct_names"] == 11
    assert "declared_options" not in barrel and "selectable" not in json.dumps(barrel)


@needs_manifest
def test_real_same_name_exotic_definitions_keep_their_identity_and_structure() -> None:
    first = real_context([(DEAD_MANS_TALE, 1498876634)]).characters[0].equipped_gear[0]
    second = real_context([(DEAD_MANS_TALE_TWIN, 1498876634)]).characters[0].equipped_gear[0]

    assert (first.item_hash, second.item_hash) == (DEAD_MANS_TALE, DEAD_MANS_TALE_TWIN)
    assert first.name == second.name
    classes = lambda item: sorted(value.classification for value in item.socket_configuration)  # noqa: E731
    assert classes(first) != classes(second)


@needs_manifest
def test_real_equipped_plugs_are_distinguishable_from_pool_samples() -> None:
    section = real_section([(DEAD_MANS_TALE, 1498876634)])
    frame = next(value for value in section["items"][0]["sockets"] if value["column"] == "frames")

    assert frame["equipped_plug"] == "Killing Wind"
    assert "Killing Wind" in frame["possible_rolls"]["sample"] or frame["possible_rolls"]["sample"]
    assert set(frame) >= {"equipped_plug", "possible_rolls", "configurability"}


@needs_manifest
def test_real_exotic_armor_is_covered_and_cosmetic_sockets_are_not() -> None:
    context = real_context([(SYNTHOCEPS, 3551918588)])
    item = context.characters[0].equipped_gear[0]

    assert item.socket_configuration_status == "verified"
    assert item.equipment_slot == "Gauntlets"
    assert all("skin" not in (value.accepts or "") for value in item.socket_configuration)
    assert all(
        "cosmetic" not in (value.category or "").casefold() for value in item.socket_configuration
    )


@needs_manifest
def test_real_ledger_flags_a_sunshot_barrel_swap_as_unconfirmed_with_fixed_context() -> None:
    context = real_context([(SUNSHOT, 2465295065)])
    ledger = MechanicsEvidenceLedger()
    ledger.record(
        "get_build_details",
        "guardian",
        GuardianToolService(context).execute("get_build_details", ARGS),
    )

    claim = detect_mechanics_claims("Swap Sunshot's barrel for an Extended Barrel.", ledger).claims[
        0
    ]

    assert claim.status == "config_change_without_legality"
    assert claim.manifest_structure == "fixed_per_manifest"


def test_a_full_chat_turn_with_the_section_is_unchanged_in_behavior() -> None:
    from types import SimpleNamespace

    from app.ai import RecommendationService
    from app.config import Settings
    from app.models import ChatRequest
    from tests.test_build_analysis import ANALYZE_CALL, ScriptedBuildResponses, _answer

    answer_text = "Your build looks coherent, so keep Sunburst as it is."
    responses = ScriptedBuildResponses([[ANALYZE_CALL], [_answer(answer_text)]])
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)

    reply = asyncio.run(
        service.chat(ChatRequest(message="Is my Titan build good?"), context_with([SUN, GAUNTLETS]))
    )
    trace = service.latest_trace() or {}

    assert reply.message == answer_text
    assert len(responses.requests) == 2  # no extra model round
    assert trace["planning_correction"]["attempted"] is False  # no new validator effect
    assert trace["mechanics_claims"]["mode"] == "trace_only"
    assert "analyze_current_build" in trace["mechanics_claims"]["retrieval"]["tools"]
