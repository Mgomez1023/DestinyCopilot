"""M5B: account- and character-scoped plug-set evidence.

Synthetic fixtures mirror the verified response shape: `profilePlugSets.data.plugs[hash]` and
`characterPlugSets.data[characterId].plugs[hash]` (nested, not flat). Real-Manifest tests at
the end check the socket joins against actual definitions and are skipped without a Manifest.
No test makes a network request.
"""

import copy
import json
from typing import Any

import pytest

from app.build_analysis import AnalyzeCurrentBuildRequest, BuildAnalysisService
from app.bungie.client import BungieClient
from app.config import Settings
from app.configuration_evidence import (
    MAX_GROUPS,
    configuration_groups,
    configuration_options,
    configuration_section,
)
from app.guardian_refresh import CACHE_SCHEMA_VERSION, SLICE_COMPONENTS, GuardianSlice
from app.guardian_tools import GuardianToolService
from app.mechanics_evidence import equipped_mechanics_section
from app.models import (
    CharacterSummary,
    GuardianContext,
    PlugSetCollection,
    PlugSetEntry,
    RequirementSource,
)
from app.objective_requirements import parse_requirements
from app.plug_set_evidence import (
    MAX_SET_ENTRIES,
    build_collection,
    resolve_reasons,
    returned_sets,
    socket_plug_set_refs,
)
from app.requirement_fit import compare_requirement, loadout_evidence
from tests.test_guardian import _normalize_profile
from tests.test_guardian_refresh import make_service
from tests.test_objective_requirements import needs_manifest, real

ITEM = "DestinyInventoryItemDefinition"
PLUG_SET = "DestinyPlugSetDefinition"
SOCKET_TYPE = "DestinySocketTypeDefinition"
CATEGORY = "DestinySocketCategoryDefinition"

INVENTORY, REUSABLE, PROFILE_SET, CHARACTER_SET = 1, 2, 4, 8
ABILITY_SET, ASPECT_SET, FRAGMENT_SET, MOD_SET, COSMETIC_SET, STRAY_SET = (
    5000,
    5001,
    5002,
    5010,
    5999,
    5555,
)


def named(name: str, **extra: Any) -> dict[str, Any]:
    return {"displayProperties": {"name": name, "description": ""}, **extra}


def plug_item(
    name: str, category: str, *, insertion: list[str] = (), enabled: list[str] = ()
) -> Any:  # noqa: B006
    return named(
        name,
        itemType=19,
        plug={
            "plugCategoryIdentifier": category,
            "insertionRules": [{"failureMessage": text} for text in insertion],
            "enabledRules": [{"failureMessage": text} for text in enabled],
        },
    )


def socket(set_hash: int, sources: int, type_hash: int) -> dict[str, Any]:
    return {"socketTypeHash": type_hash, "plugSources": sources, "reusablePlugSetHash": set_hash}


def manifest_set(*hashes: int) -> dict[str, Any]:
    return {"isFakePlugSet": False, "reusablePlugItems": [{"plugItemHash": h} for h in hashes]}


def build_definitions() -> dict[str, dict[int, dict[str, Any]]]:
    def subclass(name: str) -> dict[str, Any]:
        return named(
            name,
            itemType=16,
            equippable=True,
            sockets={
                "socketEntries": [
                    socket(ABILITY_SET, REUSABLE | PROFILE_SET, 900),
                    socket(ASPECT_SET, PROFILE_SET, 901),
                    socket(ASPECT_SET, PROFILE_SET, 901),
                    socket(FRAGMENT_SET, PROFILE_SET, 902),
                    socket(FRAGMENT_SET, PROFILE_SET, 902),
                ],
                "socketCategories": [
                    {"socketCategoryHash": 700, "socketIndexes": [0]},
                    {"socketCategoryHash": 701, "socketIndexes": [1, 2]},
                    {"socketCategoryHash": 702, "socketIndexes": [3, 4]},
                ],
            },
        )

    def armor(name: str) -> dict[str, Any]:
        return named(
            name,
            itemType=2,
            equippable=True,
            inventory={"tierTypeName": "Legendary"},
            sockets={
                "socketEntries": [
                    socket(MOD_SET, INVENTORY | PROFILE_SET | CHARACTER_SET, 903),
                    socket(COSMETIC_SET, PROFILE_SET | CHARACTER_SET, 904),
                ],
                "socketCategories": [
                    {"socketCategoryHash": 703, "socketIndexes": [0]},
                    {"socketCategoryHash": 704, "socketIndexes": [1]},
                ],
            },
        )

    return {
        "DestinyClassDefinition": {10: named("Titan"), 11: named("Hunter")},
        ITEM: {
            100: subclass("Test Titan Subclass"),
            101: subclass("Test Hunter Subclass"),
            200: armor("Test Titan Arms"),
            201: armor("Test Hunter Arms"),
            301: plug_item("Thermite Test", "shared.solar.grenades"),
            302: plug_item("Storm Test", "shared.arc.grenades"),
            311: plug_item("Aspect One", "titan.solar.aspects"),
            312: plug_item("Aspect Two", "titan.solar.aspects"),
            313: plug_item(
                "Aspect Three", "titan.solar.aspects", insertion=["Aspect locked.", "Needs rank."]
            ),
            321: plug_item("Fragment One", "shared.solar.fragments"),
            322: plug_item(
                "Fragment Two", "shared.solar.fragments", enabled=["First rule.", "Wrong aspect."]
            ),
            323: plug_item("Fragment Three", "shared.solar.fragments"),
            331: plug_item(
                "Mod One", "enhancements.v2_arms", enabled=["Not enough energy.", "Other."]
            ),
            332: plug_item("Mod Two", "enhancements.v2_arms", insertion=["Locked mod."]),
            341: plug_item("Shader One", "shader"),
        },
        PLUG_SET: {
            ABILITY_SET: manifest_set(301, 302),
            ASPECT_SET: manifest_set(311, 312, 313),
            FRAGMENT_SET: manifest_set(321, 322, 323),
            MOD_SET: manifest_set(331, 332),
            COSMETIC_SET: manifest_set(341),
        },
        SOCKET_TYPE: {
            900: {"plugWhitelist": [{"categoryIdentifier": "shared.solar.grenades"}]},
            901: {"plugWhitelist": [{"categoryIdentifier": "titan.solar.aspects"}]},
            902: {"plugWhitelist": [{"categoryIdentifier": "shared.solar.fragments"}]},
            903: {"plugWhitelist": [{"categoryIdentifier": "enhancements.v2_arms"}]},
            904: {"plugWhitelist": [{"categoryIdentifier": "shader"}]},
        },
        CATEGORY: {
            700: named("ABILITIES"),
            701: named("ASPECTS"),
            702: named("FRAGMENTS"),
            703: named("ARMOR MODS"),
            704: named("ARMOR COSMETICS"),
        },
    }


def entry(plug: int, *, can_insert: Any = True, enabled: Any = True, **extra: Any) -> dict:
    value: dict[str, Any] = {"plugItemHash": plug}
    if can_insert is not None:
        value["canInsert"] = can_insert
    if enabled is not None:
        value["enabled"] = enabled
    return {**value, **extra}


def profile_sets() -> dict[str, list[dict[str, Any]]]:
    return {
        str(ABILITY_SET): [entry(301), entry(302)],
        str(ASPECT_SET): [
            entry(311),
            entry(312),
            entry(313, can_insert=False, insertFailIndexes=[0]),
        ],
        str(FRAGMENT_SET): [
            entry(321),
            entry(322, enabled=False, enableFailIndexes=[1]),
        ],
        str(MOD_SET): [entry(331)],
        str(COSMETIC_SET): [entry(341)],
        str(STRAY_SET): [entry(999)],
    }


def character_sets(*, blocked: bool) -> dict[str, list[dict[str, Any]]]:
    mod_two = entry(332, can_insert=False, insertFailIndexes=[0, 5]) if blocked else entry(332)
    return {str(MOD_SET): [entry(331), mod_two], str(COSMETIC_SET): [entry(341)]}


def socket_state(plug: int | None, **extra: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"isEnabled": True, "isVisible": True, **extra}
    if plug is not None:
        value["plugHash"] = plug
    return value


def build_profile(
    *,
    with_profile_sets: bool = True,
    with_character_sets: bool = True,
    profile_overrides: dict[str, Any] | None = None,
    character_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plugs = copy.deepcopy(profile_sets())
    plugs.update(profile_overrides or {})
    titan_sets = character_sets(blocked=True)
    hunter_sets = character_sets(blocked=False)
    for character_id, overrides in (character_overrides or {}).items():
        target = titan_sets if character_id == "c-titan" else hunter_sets
        target.update(overrides)
    profile: dict[str, Any] = {
        "responseMintedTimestamp": "2026-10-19T00:00:00Z",
        "characters": {
            "data": {
                "c-titan": {"classHash": 10, "classType": 0, "light": 1800},
                "c-hunter": {"classHash": 11, "classType": 1, "light": 1790},
            }
        },
        "characterEquipment": {
            "data": {
                "c-titan": {
                    "items": [
                        {"itemHash": 100, "itemInstanceId": "i-sub-t"},
                        {"itemHash": 200, "itemInstanceId": "i-arm-t"},
                    ]
                },
                "c-hunter": {
                    "items": [
                        {"itemHash": 101, "itemInstanceId": "i-sub-h"},
                        {"itemHash": 201, "itemInstanceId": "i-arm-h"},
                    ]
                },
            }
        },
        "itemComponents": {
            "sockets": {
                "data": {
                    "i-sub-t": {
                        "sockets": [
                            socket_state(301),
                            socket_state(311),
                            socket_state(312),
                            socket_state(322, isEnabled=False, enableFailIndexes=[1]),
                            socket_state(None),
                        ]
                    },
                    "i-sub-h": {
                        "sockets": [socket_state(301), socket_state(311), socket_state(None)]
                    },
                    "i-arm-t": {
                        "sockets": [socket_state(331, isEnabled=False, enableFailIndexes=[0, 9])]
                    },
                    "i-arm-h": {"sockets": [socket_state(331)]},
                }
            }
        },
    }
    if with_profile_sets:
        profile["profilePlugSets"] = {"data": {"plugs": plugs}, "privacy": 1}
    if with_character_sets:
        profile["characterPlugSets"] = {
            "data": {
                "c-titan": {"plugs": titan_sets},
                "c-hunter": {"plugs": hunter_sets},
            },
            "privacy": 1,
        }
    return profile


def normalized(**options: Any) -> GuardianContext:
    return _normalize_profile(build_definitions(), build_profile(**options))


@pytest.fixture
def built_context() -> GuardianContext:
    return normalized()


def character_of(context: GuardianContext, class_name: str) -> CharacterSummary:
    return next(value for value in context.characters if value.class_name == class_name)


def groups_by_category(character: CharacterSummary, context: GuardianContext) -> dict[str, Any]:
    return {
        value["category"]: value
        for value in configuration_groups(character, context.profile_plug_sets)
    }


# --- Response shape and normalization ---------------------------------------------------------


def test_the_response_is_read_as_nested_data_plugs_not_a_flat_dictionary() -> None:
    profile = build_profile()

    assert set(returned_sets(profile, "profile") or {}) >= {ASPECT_SET, MOD_SET}
    titan = returned_sets(profile, "character", "c-titan") or {}
    assert set(titan) == {MOD_SET, COSMETIC_SET}
    # A flat mapping, or a character id that is not in the data, is "not returned".
    assert returned_sets({"profilePlugSets": {"data": {str(ASPECT_SET): []}}}, "profile") is None
    assert returned_sets(profile, "character", "c-unknown") is None
    assert returned_sets(profile, "character", None) is None


def test_profile_scoped_sets_are_retained_once_under_the_context() -> None:
    context = normalized()
    collection = context.profile_plug_sets

    assert collection.status == "returned" and collection.source == "GetProfile:ItemSockets"
    assert collection.minted_at == "2026-10-19T00:00:00Z"
    assert {value.plug_set_hash for value in collection.sets} == {
        ABILITY_SET,
        ASPECT_SET,
        FRAGMENT_SET,
        MOD_SET,
    }
    assert all(value.scope == "profile" and value.character_id is None for value in collection.sets)


def test_character_scoped_sets_stay_with_their_character() -> None:
    context = normalized()
    titan, hunter = character_of(context, "Titan"), character_of(context, "Hunter")

    assert [value.plug_set_hash for value in titan.plug_sets.sets] == [MOD_SET]
    assert titan.plug_sets.sets[0].character_id == "c-titan"
    assert hunter.plug_sets.sets[0].character_id == "c-hunter"
    titan_blocked = [e for e in titan.plug_sets.sets[0].entries if e.can_insert is False]
    hunter_blocked = [e for e in hunter.plug_sets.sets[0].entries if e.can_insert is False]
    assert len(titan_blocked) == 1 and hunter_blocked == []


def test_unreferenced_and_cosmetic_plug_sets_are_dropped() -> None:
    context = normalized()
    retained = {value.plug_set_hash for value in context.profile_plug_sets.sets}
    retained |= {value.plug_set_hash for c in context.characters for value in c.plug_sets.sets}

    assert STRAY_SET not in retained and COSMETIC_SET not in retained
    assert not any(
        ref.plug_set_hash == COSMETIC_SET
        for c in context.characters
        for item in c.equipped_gear
        for ref in item.plug_set_sockets
    )


def test_plug_set_sockets_are_exact_hash_and_source_joins() -> None:
    context = normalized()
    refs = {
        (item.name, ref.socket_index): ref
        for item in character_of(context, "Titan").equipped_gear
        for ref in item.plug_set_sockets
    }

    ability = refs[("Test Titan Subclass", 0)]
    assert ability.plug_set_hash == ABILITY_SET
    assert ability.sources == ["reusable_plug_items", "profile_plug_set"]  # combined, not exclusive
    assert ability.category == "ABILITIES" and ability.manifest_options == 2
    assert refs[("Test Titan Subclass", 1)].plug_set_hash == ASPECT_SET
    assert refs[("Test Titan Subclass", 2)].plug_set_hash == ASPECT_SET
    mods = refs[("Test Titan Arms", 0)]
    assert mods.sources == ["inventory", "profile_plug_set", "character_plug_set"]
    assert mods.manifest_options == 2


def test_manifest_counts_and_unmatched_returned_plugs_are_recorded() -> None:
    context = normalized(
        profile_overrides={str(ASPECT_SET): [entry(311), entry(777)]}  # 777 is not in the set
    )
    aspects = context.profile_plug_sets.get(ASPECT_SET)

    assert aspects is not None
    assert aspects.manifest_options == 3 and aspects.manifest_status == "resolved"
    assert aspects.unmatched_entries == 1 and aspects.listed == 2


def test_a_set_with_no_manifest_definition_is_unresolved_not_dropped() -> None:
    definitions = build_definitions()
    del definitions[PLUG_SET][FRAGMENT_SET]
    context = _normalize_profile(definitions, build_profile())

    fragments = context.profile_plug_sets.get(FRAGMENT_SET)
    assert fragments is not None and fragments.manifest_status == "unresolved"
    assert fragments.manifest_options is None and fragments.listed == 2


# --- Missing, empty and uncollected data ------------------------------------------------------


def test_missing_components_are_not_returned_never_an_empty_set() -> None:
    context = normalized(with_profile_sets=False, with_character_sets=False)
    titan = character_of(context, "Titan")

    assert context.profile_plug_sets.status == "not_returned"
    assert titan.plug_sets.status == "not_returned"
    assert context.profile_plug_sets.sets == [] and titan.plug_sets.sets == []
    groups = groups_by_category(titan, context)
    assert groups["ASPECTS"]["profile"] == {"status": "not_returned"}
    assert "unknown" in groups["ASPECTS"]["states"]
    assert "account_listed" not in groups["ASPECTS"]["states"]


def test_a_referenced_set_missing_from_returned_data_is_unknown() -> None:
    plugs = profile_sets()
    del plugs[str(FRAGMENT_SET)]
    context = normalized(profile_overrides={})
    profile = build_profile()
    profile["profilePlugSets"]["data"]["plugs"].pop(str(FRAGMENT_SET))
    context = _normalize_profile(build_definitions(), profile)

    assert context.profile_plug_sets.missing_set_hashes == [FRAGMENT_SET]
    group = groups_by_category(character_of(context, "Titan"), context)["FRAGMENTS"]
    assert group["profile"] == {"status": "set_not_returned"}
    assert "unknown" in group["states"]


def test_a_returned_empty_set_differs_from_an_uncollected_one() -> None:
    context = normalized(profile_overrides={str(FRAGMENT_SET): []})
    titan = character_of(context, "Titan")
    fragments = context.profile_plug_sets.get(FRAGMENT_SET)

    assert fragments is not None and fragments.listed == 0 and fragments.entries == []
    group = groups_by_category(titan, context)["FRAGMENTS"]
    assert group["profile"]["status"] == "returned" and group["profile"]["listed"] == 0
    assert "account_listed" not in group["states"]
    assert PlugSetCollection().status == "not_collected"


def test_a_character_absent_from_character_plug_set_data_is_not_returned() -> None:
    profile = build_profile()
    del profile["characterPlugSets"]["data"]["c-hunter"]
    context = _normalize_profile(build_definitions(), profile)

    assert character_of(context, "Hunter").plug_sets.status == "not_returned"
    assert character_of(context, "Titan").plug_sets.status == "returned"


def test_entries_are_capped_with_flagged_entries_kept_first() -> None:
    many = [entry(10_000 + index) for index in range(MAX_SET_ENTRIES + 50)]
    many.append(entry(313, can_insert=False, insertFailIndexes=[0]))
    profile = {"profilePlugSets": {"data": {"plugs": {str(ASPECT_SET): many}}}}

    collection = build_collection(profile, "profile", None, {ASPECT_SET}, {}, {})
    evidence = collection.get(ASPECT_SET)

    assert evidence is not None
    assert evidence.listed == MAX_SET_ENTRIES + 51
    retained = len(evidence.entries) + len(evidence.clear_plug_hashes)
    assert retained == MAX_SET_ENTRIES and evidence.omitted_entries == 51
    assert evidence.entries[0].plug_item_hash == 313  # flagged entries survive the cap


# --- canInsert versus enabled, and failure indexes --------------------------------------------


def test_can_insert_false_is_a_restriction_and_enabled_false_is_not() -> None:
    context = normalized()
    groups = groups_by_category(character_of(context, "Titan"), context)

    aspects, fragments = groups["ASPECTS"]["profile"], groups["FRAGMENTS"]["profile"]
    assert [v["plug"] for v in aspects["insertion_blocked_reported"]] == ["Aspect Three"]
    assert "not_enabled_reported" not in aspects
    assert fragments["not_enabled_reported"] == ["Fragment Two"]
    assert "insertion_blocked_reported" not in fragments
    assert "insertion_blocked_reported" in groups["ASPECTS"]["states"]
    assert "insertion_blocked_reported" not in groups["FRAGMENTS"]["states"]


def test_insertion_and_enablement_indexes_resolve_against_their_own_rule_arrays() -> None:
    context = normalized()
    aspects = context.profile_plug_sets.get(ASPECT_SET)
    fragments = context.profile_plug_sets.get(FRAGMENT_SET)

    assert aspects is not None and fragments is not None
    blocked = next(v for v in aspects.entries if v.plug_item_hash == 313)
    assert blocked.insert_fail_indexes == [0] and blocked.insert_fail_reasons == ["Aspect locked."]
    assert blocked.name == "Aspect Three"
    disabled = next(v for v in fragments.entries if v.plug_item_hash == 322)
    # enableFailIndexes [1] reads enabledRules[1], not insertionRules.
    assert disabled.enable_fail_reasons == ["Wrong aspect."] and disabled.insert_fail_reasons == []


def test_unresolvable_failure_indexes_stay_unresolved() -> None:
    plug = plug_item("P", "x", insertion=["Only rule."], enabled=["Enabled rule."])

    assert resolve_reasons([0], plug, "insertionRules") == ["Only rule."]
    assert resolve_reasons([1], plug, "insertionRules") == [None]  # out of range
    assert resolve_reasons([-1], plug, "enabledRules") == [None]
    assert resolve_reasons([0], None, "insertionRules") == [None]  # definition unavailable
    assert resolve_reasons([0], {"plug": {"insertionRules": "bad"}}, "insertionRules") == [None]
    assert resolve_reasons([0], {"plug": {"insertionRules": [{}]}}, "insertionRules") == [None]
    assert resolve_reasons([0, 1], plug, "insertionRules") == ["Only rule.", None]


def test_an_unresolved_reason_is_reported_as_unresolved_not_dropped() -> None:
    context = normalized()
    titan = character_of(context, "Titan")
    mods = groups_by_category(titan, context)["ARMOR MODS"]

    blocked = mods["character"]["insertion_blocked_reported"][0]
    assert blocked["plug"] == "Mod Two"
    assert blocked["reasons"] == ["Locked mod.", None]  # index 5 does not exist
    assert blocked["reasons_unresolved"] is True


def test_missing_flags_are_not_manufactured() -> None:
    profile = {"profilePlugSets": {"data": {"plugs": {str(ASPECT_SET): [{"plugItemHash": 311}]}}}}
    collection = build_collection(profile, "profile", None, {ASPECT_SET}, {}, {})
    only = collection.sets[0].entries[0]

    assert only.can_insert is None and only.enabled is None
    assert PlugSetEntry(plug_item_hash=1).model_dump(mode="json") == {"plug_item_hash": 1}


def test_equipped_plug_enable_failures_are_kept_on_the_socket_evidence() -> None:
    context = normalized()
    subclass = character_of(context, "Titan").subclass
    assert subclass is not None
    fragment = next(v for v in subclass.socketed_plug_details if v.name == "Fragment Two")

    assert fragment.is_enabled is False
    assert fragment.enable_fail_indexes == [1] and fragment.enable_fail_reasons == ["Wrong aspect."]
    healthy = next(v for v in subclass.socketed_plug_details if v.name == "Aspect One")
    assert healthy.enable_fail_indexes == []
    assert "enable_fail_indexes" not in healthy.model_dump(mode="json")


def test_equipped_enable_failure_with_an_unresolvable_index_stays_unresolved() -> None:
    context = normalized()
    arms = next(
        v for v in character_of(context, "Titan").equipped_gear if v.name == "Test Titan Arms"
    )
    mod = next(v for v in arms.socketed_plug_details if v.name == "Mod One")

    assert mod.enable_fail_indexes == [0, 9]
    assert mod.enable_fail_reasons == ["Not enough energy.", None]


# --- Subclass and armor coverage ---------------------------------------------------------------


def test_subclass_abilities_aspects_and_fragments_each_get_their_own_group() -> None:
    context = normalized()
    groups = groups_by_category(character_of(context, "Titan"), context)

    assert set(groups) == {"ABILITIES", "ASPECTS", "FRAGMENTS", "ARMOR MODS"}
    assert groups["ASPECTS"]["sockets"] == 2 and groups["FRAGMENTS"]["sockets"] == 2
    assert groups["ASPECTS"]["kind"] == "subclass" and groups["ARMOR MODS"]["kind"] == "armor"
    assert [v["plug"] for v in groups["ASPECTS"]["equipped"]] == ["Aspect One", "Aspect Two"]
    assert groups["ASPECTS"]["profile"]["listed"] == 3
    assert "character" not in groups["ASPECTS"]  # only the profile source is declared
    assert groups["ABILITIES"]["sources"] == ["reusable_plug_items", "profile_plug_set"]
    assert "instance reusable plugs not retrieved" in groups["ABILITIES"]["limitations"]


def test_combined_sources_report_every_declared_scope_separately() -> None:
    context = normalized()
    mods = groups_by_category(character_of(context, "Titan"), context)["ARMOR MODS"]

    assert mods["profile"]["listed"] == 1 and mods["character"]["listed"] == 2
    assert {"account_listed", "character_listed", "equipped_now"} <= set(mods["states"])
    assert "inventory-sourced plugs not read" in mods["limitations"]


def test_fragment_and_armor_mod_legality_is_never_inferred_from_listing() -> None:
    context = normalized()
    titan = character_of(context, "Titan")
    groups = groups_by_category(titan, context)

    assert "actionable" not in groups["FRAGMENTS"]  # stated once, on the section
    assert any("slot capacity" in text for text in groups["FRAGMENTS"]["limitations"])
    assert any("energy" in text.casefold() for text in groups["ARMOR MODS"]["limitations"])
    section = configuration_section(titan, context.profile_plug_sets)
    assert section["actionable"] == "unknown" and len(section["groups"]) == 4
    options = configuration_options(titan, context.profile_plug_sets)
    assert options and all(value.actionable == "unknown" for value in options)


def test_options_keep_provenance_per_state_and_never_use_instance_states() -> None:
    context = normalized()
    titan = character_of(context, "Titan")
    options = configuration_options(titan, context.profile_plug_sets)
    aspect = next(v for v in options if v.plug_name == "Aspect One")

    assert aspect.state_names() == ["manifest_possible", "equipped_now", "account_listed"]
    sources = {value.state: value.source for value in aspect.states}
    assert sources["manifest_possible"].kind == "manifest_plug_set"
    assert sources["manifest_possible"].plug_set_hash == ASPECT_SET
    assert sources["equipped_now"].kind == "socket_component"
    assert sources["equipped_now"].socket_index == 1 and sources["equipped_now"].item_hash == 100
    assert sources["account_listed"].kind == "profile_plug_set"
    assert sources["account_listed"].minted_at == "2026-10-19T00:00:00Z"
    assert all("instance_listed" not in value.state_names() for value in options)
    assert aspect.instance_id == "i-sub-t"  # internal handle
    assert "instance_id" not in aspect.model_dump(mode="json")


def test_a_scope_that_was_not_returned_yields_unknown_not_a_listed_state() -> None:
    context = normalized(with_profile_sets=False)
    titan = character_of(context, "Titan")
    aspect = next(
        v
        for v in configuration_options(titan, context.profile_plug_sets)
        if v.plug_name == "Aspect One"
    )

    assert "account_listed" not in aspect.state_names() and "unknown" in aspect.state_names()


def test_an_equipped_enable_failure_is_not_a_blocked_insertion() -> None:
    context = normalized()
    titan = character_of(context, "Titan")
    option = next(
        v
        for v in configuration_options(titan, context.profile_plug_sets)
        if v.plug_name == "Fragment Two"
    )

    assert "insertion_blocked_reported" not in option.state_names()
    equipped = next(v for v in option.states if v.state == "equipped_now")
    assert equipped.enabled is False and equipped.reasons == ["Wrong aspect."]


# --- Character isolation ----------------------------------------------------------------------


def test_character_scoped_evidence_never_crosses_characters() -> None:
    context = normalized()
    titan = character_of(context, "Titan")
    hunter = character_of(context, "Hunter")

    titan_mods = groups_by_category(titan, context)["ARMOR MODS"]["character"]
    hunter_mods = groups_by_category(hunter, context)["ARMOR MODS"]["character"]
    assert "insertion_blocked_reported" in titan_mods
    assert "insertion_blocked_reported" not in hunter_mods
    assert hunter_mods["listed"] == 2


def test_the_tool_returns_only_the_requested_characters_configuration() -> None:
    context = normalized()
    service = GuardianToolService(context)

    hunter = service.execute(
        "get_build_details", {"character_id": None, "character_class": "Hunter"}
    )
    titan = service.execute("get_build_details", {"character_id": None, "character_class": "Titan"})
    hunter_mods = next(
        v for v in hunter["configuration_evidence"]["groups"] if v["category"] == "ARMOR MODS"
    )
    titan_mods = next(
        v for v in titan["configuration_evidence"]["groups"] if v["category"] == "ARMOR MODS"
    )

    assert "insertion_blocked_reported" not in hunter_mods["character"]
    assert "insertion_blocked_reported" in titan_mods["character"]
    assert "Test Titan" not in json.dumps(hunter["configuration_evidence"])


def test_profile_scoped_evidence_is_shared_between_characters() -> None:
    context = normalized()

    aspects = [
        groups_by_category(character_of(context, name), context)["ASPECTS"]["profile"]
        for name in ("Titan", "Hunter")
    ]
    assert aspects[0] == aspects[1]


# --- Tool integration, bounds and privacy ------------------------------------------------------


def test_build_tools_carry_a_compact_configuration_section() -> None:
    context = normalized()
    details = GuardianToolService(context).execute(
        "get_build_details", {"character_id": None, "character_class": "Titan"}
    )
    analysis = BuildAnalysisService(context).analyze_current_build(
        AnalyzeCurrentBuildRequest(
            character_id=None, character_class="Titan", goal="solo PvE", activity=None
        )
    )

    for result in (details, analysis):
        section = result["configuration_evidence"]
        assert section["groups"] and "basis" in section and "scope" in section
        assert len(json.dumps(section)) <= 3800
        assert "did not retrieve" not in section["basis"]
    assert "does not prove a change can be made now" in details["configuration_evidence"]["basis"]


def test_other_tools_do_not_carry_configuration_evidence() -> None:
    context = normalized()
    service = GuardianToolService(context)

    for name in ("get_equipped_loadout", "get_active_quests", "get_character_summary"):
        result = service.execute(name, {"character_id": None, "character_class": "Titan"})
        assert "configuration_evidence" not in json.dumps(result)


def test_a_character_without_plug_set_sockets_gets_a_light_section() -> None:
    character = CharacterSummary(
        character_id="x", class_name="Titan", race_name="a", gender_name="b", power=1
    )
    section = configuration_section(character, PlugSetCollection())

    assert section["groups"] == [] and "basis" not in section and "note" in section


def test_the_section_is_bounded_and_counts_omitted_groups() -> None:
    context = normalized()
    titan = character_of(context, "Titan")

    small = configuration_section(titan, context.profile_plug_sets, max_groups=2)
    assert len(small["groups"]) == 2 and small["omitted_groups"] == 2
    tiny = configuration_section(titan, context.profile_plug_sets, max_chars=900)
    assert tiny["omitted_groups"] >= 1
    assert MAX_GROUPS >= 4


def test_no_instance_account_or_membership_identifiers_reach_the_model() -> None:
    context = normalized()
    result = GuardianToolService(context).execute(
        "get_build_details", {"character_id": None, "character_class": "Titan"}
    )
    encoded = json.dumps(result["configuration_evidence"])

    for private in ("i-sub-t", "i-arm-t", "c-titan", "c-hunter", "123"):
        assert private not in encoded


# --- Existing behavior is unchanged -------------------------------------------------------------


def test_equipped_mechanics_are_identical_with_and_without_plug_set_evidence() -> None:
    with_sets = character_of(normalized(), "Titan")
    without = character_of(normalized(with_profile_sets=False, with_character_sets=False), "Titan")

    assert equipped_mechanics_section(with_sets) == equipped_mechanics_section(without)


def test_requirement_fit_still_describes_only_the_equipped_loadout() -> None:
    # An Arc grenade is *listed* for the account, but the equipped grenade is Solar.
    context = normalized()
    titan = character_of(context, "Titan")
    source = RequirementSource(definition="DestinyInventoryItemDefinition", hash=1, field="f")
    requirement = parse_requirements(
        "Defeat combatants with Arc ability final blows.", kind="description", source=source
    )[0]

    listed = context.profile_plug_sets.get(ABILITY_SET)
    assert listed is not None and 302 in listed.clear_plug_hashes
    result = compare_requirement(requirement, loadout_evidence(titan))
    without = compare_requirement(
        requirement,
        loadout_evidence(character_of(normalized(with_profile_sets=False), "Titan")),
    )

    assert result.status == "verified_mismatch" == without.status
    assert result.equipped == []


def test_existing_item_and_character_serialization_omits_empty_plug_set_fields() -> None:
    character = CharacterSummary(
        character_id="x", class_name="Titan", race_name="a", gender_name="b", power=1
    )
    context = normalized()
    sword = next(v for v in character_of(context, "Titan").equipped_gear if v.item_type == "Armor")

    assert "plug_set_sockets" in sword.model_dump(mode="json")
    plain = sword.model_copy(update={"plug_set_sockets": []})
    assert "plug_set_sockets" not in plain.model_dump(mode="json")
    assert character.plug_sets.status == "not_collected"


# --- Cache -------------------------------------------------------------------------------------


def test_cache_schema_covers_plug_sets() -> None:
    assert CACHE_SCHEMA_VERSION >= 8


def test_old_payloads_load_as_not_collected() -> None:
    old = {
        "character_id": "x",
        "class_name": "Titan",
        "race_name": "a",
        "gender_name": "b",
        "power": 1,
    }

    restored = CharacterSummary.model_validate(old)

    assert restored.plug_sets.status == "not_collected"
    assert (
        GuardianContext.model_validate(
            {
                "bungie_display_name": "n",
                "membership_id": "m",
                "membership_type": 3,
                "platform_name": "p",
                "data_availability": {"fetched_at": "2026-10-19T00:00:00Z"},
            }
        ).profile_plug_sets.status
        == "not_collected"
    )


def test_plug_set_evidence_round_trips_through_json() -> None:
    context = normalized()
    restored = GuardianContext.model_validate(context.model_dump(mode="json"))

    assert restored.profile_plug_sets == context.profile_plug_sets
    assert character_of(restored, "Titan").plug_sets == character_of(context, "Titan").plug_sets
    assert character_of(restored, "Titan").subclass == character_of(context, "Titan").subclass


@pytest.mark.asyncio
async def test_plug_set_evidence_survives_the_refresh_cache_and_stays_isolated(
    built_context: GuardianContext,
) -> None:
    context = built_context
    service, guardian, _ = make_service(context=context)

    await service.get_context("session-a", "token")
    warm = await service.get_context("session-a", "token")

    assert guardian.full_calls == 1 and warm.cache_hit
    assert warm.context.profile_plug_sets == context.profile_plug_sets
    titan = character_of(warm.context, "Titan")
    hunter = character_of(warm.context, "Hunter")
    assert titan.plug_sets == character_of(context, "Titan").plug_sets
    assert titan.plug_sets != hunter.plug_sets
    assert [v.name for v in titan.equipped_gear] == [
        v.name for v in character_of(context, "Titan").equipped_gear
    ]


# --- No new Bungie requests ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_profile_request_asks_for_the_same_components_as_before() -> None:
    captured: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {"ErrorCode": 1, "Response": {}}

    class Http:
        async def get(self, url: str, headers: Any = None, params: Any = None) -> Response:
            captured.update(url=url, params=params)
            return Response()

    client = BungieClient(Settings(), Http())  # type: ignore[arg-type]
    await client.get_profile(3, "1", "token")
    requested = set(captured["params"]["components"].split(","))

    assert "ItemSockets" in requested
    assert not requested & {"ItemReusablePlugs", "ItemPlugStates", "ItemPlugObjectives"}
    assert not requested & {"CharacterLoadouts", "CurrencyLookups"}
    assert len(requested) == 16
    for components in SLICE_COMPONENTS.values():
        assert not components & {"ItemReusablePlugs", "ItemPlugStates", "CharacterLoadouts"}
    assert "ItemSockets" in SLICE_COMPONENTS[GuardianSlice.EQUIPMENT]


def test_normalization_has_no_client_to_call() -> None:
    from app.bungie.guardian import GuardianNormalizer

    normalizer = GuardianNormalizer(object())  # type: ignore[arg-type]

    assert not hasattr(normalizer, "client")


# --- Real Manifest -----------------------------------------------------------------------------


@needs_manifest
def test_real_subclass_sockets_join_to_profile_plug_sets_by_exact_hash() -> None:
    definition = real("DestinyInventoryItemDefinition", 2550323932)  # a Titan subclass
    skip: set[int] = set()
    refs = socket_plug_set_refs(definition, skip)
    by_index = {value.socket_index: value for value in refs}

    assert by_index[5].plug_set_hash == by_index[6].plug_set_hash == 4030440911  # two aspects
    assert by_index[5].sources == ["profile_plug_set"]
    assert {by_index[index].plug_set_hash for index in (7, 8, 9, 10, 11, 12)} == {1319395210}
    assert by_index[4].sources == ["reusable_plug_items", "profile_plug_set"]  # grenades
    plug_set = real("DestinyPlugSetDefinition", 4030440911)
    assert len(plug_set["reusablePlugItems"]) == 5
    with_options = socket_plug_set_refs(definition, skip, None, {4030440911: plug_set})
    assert {v.manifest_options for v in with_options if v.plug_set_hash == 4030440911} == {5}


@needs_manifest
def test_real_armor_mod_sockets_declare_inventory_profile_and_character_sources() -> None:
    definition = real("DestinyInventoryItemDefinition", 241462142)  # an Exotic gauntlets
    refs = socket_plug_set_refs(definition)
    mod = next(v for v in refs if v.socket_index == 1)

    assert mod.sources == ["inventory", "profile_plug_set", "character_plug_set"]
    assert mod.plug_set_hash == 3899745242
    # Perk sockets with a randomized pool and no plug source are not plug-set sockets.
    assert 6 not in {v.socket_index for v in refs}


@needs_manifest
def test_real_weapon_perk_sockets_are_not_treated_as_plug_set_evidence() -> None:
    definition = real("DestinyInventoryItemDefinition", 2429822977)  # a randomized weapon
    refs = socket_plug_set_refs(definition)

    # Its perk columns draw on reusable items and randomized sets, not profile/character sets.
    assert {1, 2, 3, 4}.isdisjoint({v.socket_index for v in refs})


# --- Truncation is never verified absence ---------------------------------------------------


def test_a_truncated_set_reports_lower_bounds_and_unknown_for_unretained_plugs() -> None:
    many = [entry(10_000 + index) for index in range(MAX_SET_ENTRIES + 25)]
    profile = build_profile(profile_overrides={str(FRAGMENT_SET): many})
    context = _normalize_profile(build_definitions(), profile)
    titan = character_of(context, "Titan")

    evidence = context.profile_plug_sets.get(FRAGMENT_SET)
    assert evidence is not None and evidence.omitted_entries == 25
    report = groups_by_category(titan, context)["FRAGMENTS"]["profile"]
    assert report["entries_not_retained"] == 25 and report["counts_are_lower_bounds"] is True
    assert report["listed"] == MAX_SET_ENTRIES + 25  # the total Bungie listed is still stated

    # An equipped plug whose entry was not retained is unknown, not "not listed".
    options = configuration_options(titan, context.profile_plug_sets)
    fragment = next(v for v in options if v.plug_name == "Fragment Two")
    unretained = [v for v in fragment.states if v.source.kind == "profile_plug_set"]
    assert [v.state for v in unretained] == ["unknown"]
    assert "account_listed" not in fragment.state_names()


def test_an_untruncated_set_makes_no_lower_bound_claim() -> None:
    context = normalized()
    report = groups_by_category(character_of(context, "Titan"), context)["ASPECTS"]["profile"]

    assert "counts_are_lower_bounds" not in report and "entries_not_retained" not in report
