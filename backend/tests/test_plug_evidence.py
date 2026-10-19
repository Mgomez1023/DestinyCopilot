"""M1: complete equipped-plug evidence from the Manifest, through to the build tools.

The definitions here are synthetic and generic. They reproduce the Manifest shapes the
P1-3A audit found (a plug whose item description is empty while a linked sandbox perk holds
the effect, a blank-name plug, duplicate perk links) without naming any production item.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from app.build_analysis import (
    bound_plug_mechanics,
    effect_status,
    equipped_plug_effects,
)
from app.bungie.manifest import DefinitionResolver
from app.guardian_tools import GuardianToolService
from app.mechanics_claims import MechanicsEvidenceLedger, detect_mechanics_claims
from app.models import (
    PlugEvidence,
    PlugStatEffect,
    SandboxPerkSummary,
    SocketedPlugSummary,
)
from tests.test_build_analysis import build_context
from tests.test_guardian import FakeResolver, _normalize_profile

ITEM = "DestinyInventoryItemDefinition"
PERK = "DestinySandboxPerkDefinition"
STAT = "DestinyStatDefinition"
CATEGORY = "DestinySocketCategoryDefinition"


def named(name: str, description: str = "", **extra: Any) -> dict[str, Any]:
    return {"displayProperties": {"name": name, "description": description}, **extra}


def plug_definition(
    name: str, description: str = "", category: str = "enhancements.v2_arms", **extra: Any
) -> dict[str, Any]:
    return named(
        name,
        description,
        itemType=19,
        itemTypeDisplayName="Armor Mod",
        plug={"plugCategoryIdentifier": category, **extra.pop("plug", {})},
        **extra,
    )


DEFINITIONS: dict[str, dict[int, dict[str, Any]]] = {
    "DestinyClassDefinition": {10: named("Titan")},
    ITEM: {
        100: named(
            "Test Gauntlets",
            itemType=2,
            itemTypeDisplayName="Gauntlets",
            classType=0,
            inventory={"tierTypeName": "Exotic"},
            sockets={
                "socketCategories": [
                    {"socketCategoryHash": 7001, "socketIndexes": [0, 1, 2]},
                    {"socketCategoryHash": 7002, "socketIndexes": [3, 4, 5, 6, 7, 8, 9]},
                ]
            },
        ),
        # Empty item description; the effect lives in two linked sandbox perks (one repeated).
        200: plug_definition(
            "Blank Description Mod",
            "",
            perks=[{"perkHash": 900}, {"perkHash": 900}, {"perkHash": 901}],
            investmentStats=[{"statTypeHash": 800, "value": 3}],
            plug={"energyCost": {"energyCost": 2, "energyType": 3}},
        ),
        # Both layers present and different.
        201: plug_definition(
            "Both Layers Perk",
            "Long conditional text for the plug.",
            "intrinsics",
            perks=[{"perkHash": 902}],
        ),
        # Linked perks exist but carry no text.
        202: plug_definition("Hollow Link", "", perks=[{"perkHash": 903}]),
        # Links a sandbox perk the Manifest lookup cannot return.
        203: plug_definition("Unresolved Link", "", perks=[{"perkHash": 999}]),
        # No description and no links: verified absence.
        204: plug_definition("Bare Mod", ""),
        # A real definition with a blank display name and only stats (an archetype package).
        205: plug_definition(
            "",
            "",
            "intrinsics",
            investmentStats=[{"statTypeHash": 800, "value": 5, "isConditionallyActive": True}],
        ),
        206: plug_definition("Shader Thing", "Cosmetic.", "shader"),
        207: plug_definition("Empty Mod Socket", "No mod currently selected."),
    },
    PERK: {
        900: named("Sandbox A", "Short effect A."),
        901: named("Sandbox B", "Short effect B."),
        902: named("Both Layers Perk", "Short summary."),
        903: named("", "", isDisplayable=False),
    },
    STAT: {800: named("Resilience")},
    CATEGORY: {7001: named("ARMOR MODS"), 7002: named("ARMOR PERKS")},
}
PLUGS_BY_SOCKET = [200, 201, 202, 203, 204, 205, 206, 207, 999]


class RecordingResolver(FakeResolver):
    """Records whether enrichment lookups ask to stay off the network."""

    def __init__(self, definitions: dict[str, dict[int, dict[str, Any]]]) -> None:
        super().__init__(definitions)
        self.local_only_calls: dict[str, bool] = {}

    async def resolve_many(  # type: ignore[override]
        self, entity_type: str, entity_hashes: set[int], *, local_only: bool = False
    ) -> dict[int, dict[str, Any]]:
        self.local_only_calls[entity_type] = local_only
        return await super().resolve_many(entity_type, entity_hashes)


def profile(
    *, location: str = "equipped", sockets: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    item = {"itemHash": 100, "itemInstanceId": "i100"}
    entries = sockets or [{"plugHash": value} for value in PLUGS_BY_SOCKET]
    base: dict[str, Any] = {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "characterEquipment": {"data": {"c1": {"items": []}}},
        "profileInventory": {"data": {"items": []}},
        "itemComponents": {"sockets": {"data": {"i100": {"sockets": entries}}}},
    }
    if location == "equipped":
        base["characterEquipment"]["data"]["c1"]["items"] = [item]
    else:
        base["profileInventory"]["data"]["items"] = [{**item, "location": 2}]
    return base


def normalize(extra: dict[str, Any], resolver: FakeResolver | None = None) -> Any:
    return _normalize_profile(DEFINITIONS, extra, resolver=resolver)


def equipped_plugs(context: Any) -> dict[str, SocketedPlugSummary]:
    return {
        value.name: value for value in context.characters[0].equipped_gear[0].socketed_plug_details
    }


# --- Normalization: what is preserved ---


def test_sandbox_text_is_kept_apart_from_an_empty_item_description() -> None:
    plugs = equipped_plugs(normalize(profile()))
    blank = plugs["Blank Description Mod"]

    assert not blank.description
    assert [value.description for value in blank.sandbox_perks] == [
        "Short effect A.",
        "Short effect B.",
    ]
    assert blank.evidence == PlugEvidence(
        description="empty_in_manifest", sandbox_perks="present", stat_effects="present"
    )


def test_both_text_layers_are_preserved_separately() -> None:
    both = equipped_plugs(normalize(profile()))["Both Layers Perk"]

    assert both.description == "Long conditional text for the plug."
    assert both.sandbox_perks[0].description == "Short summary."
    assert both.sandbox_perks[0].name == "Both Layers Perk"


def test_duplicate_perk_links_are_collapsed_to_one_record_each() -> None:
    blank = equipped_plugs(normalize(profile()))["Blank Description Mod"]

    assert [value.perk_hash for value in blank.sandbox_perks] == [900, 901]


def test_socket_position_category_and_state_flags_come_from_the_socket_and_item() -> None:
    sockets = [{"plugHash": 200, "isEnabled": False, "isVisible": True}, {"plugHash": 201}]
    plugs = equipped_plugs(normalize(profile(sockets=sockets)))

    first, second = plugs["Blank Description Mod"], plugs["Both Layers Perk"]
    assert (first.socket_index, first.socket_category) == (0, "ARMOR MODS")
    assert first.is_enabled is False  # a real "off" is kept, not treated as missing
    assert first.is_visible is True
    assert (second.socket_index, second.socket_category) == (1, "ARMOR MODS")
    assert second.is_enabled is None and second.is_visible is None


def test_stat_deltas_and_energy_cost_are_named_from_definitions() -> None:
    blank = equipped_plugs(normalize(profile()))["Blank Description Mod"]

    assert blank.stat_effects == [PlugStatEffect(name="Resilience", value=3)]
    assert (blank.energy_cost, blank.energy_type) == (2, "Void")


def test_conditional_stat_deltas_stay_marked_conditional() -> None:
    unnamed = equipped_plugs(normalize(profile()))["Unnamed plug"]

    assert unnamed.stat_effects == [PlugStatEffect(name="Resilience", value=5, conditional=True)]


# --- Missing information is not verified absence ---


def test_each_evidence_state_is_distinct() -> None:
    plugs = equipped_plugs(normalize(profile()))

    assert plugs["Bare Mod"].evidence == PlugEvidence(
        description="empty_in_manifest", sandbox_perks="none_declared", stat_effects="none_declared"
    )
    assert plugs["Hollow Link"].evidence.sandbox_perks == "declared_without_text"  # type: ignore[union-attr]
    assert plugs["Hollow Link"].sandbox_perks == []
    assert plugs["Unresolved Link"].evidence.sandbox_perks == "unresolved"  # type: ignore[union-attr]
    assert effect_status(plugs["Bare Mod"]) == "none_in_manifest"
    assert effect_status(plugs["Unresolved Link"]) == "not_retrieved"
    assert effect_status(plugs["Blank Description Mod"]) == "present"


def test_a_plug_whose_definition_is_missing_is_unavailable_not_empty() -> None:
    unknown = equipped_plugs(normalize(profile()))["Unknown plug"]

    assert unknown.evidence == PlugEvidence(
        description="definition_unavailable",
        sandbox_perks="definition_unavailable",
        stat_effects="definition_unavailable",
    )
    assert effect_status(unknown) == "not_retrieved"


def test_a_blank_name_plug_is_visible_but_never_an_empty_name() -> None:
    context = normalize(profile())
    gear = context.characters[0].equipped_gear[0]
    unnamed = equipped_plugs(context)["Unnamed plug"]

    assert unnamed.unnamed is True
    assert "" not in gear.socketed_plugs and "Unnamed plug" not in gear.socketed_plugs
    assert unnamed.stat_effects


def test_cosmetics_and_empty_socket_placeholders_collect_no_evidence() -> None:
    plugs = equipped_plugs(normalize(profile()))

    assert plugs["Shader Thing"].evidence is None
    assert plugs["Empty Mod Socket"].evidence is None
    assert plugs["Shader Thing"].sandbox_perks == []


def test_only_equipped_items_collect_plug_evidence() -> None:
    context = normalize(profile(location="vault"))
    vault_item = context.inventory.items[0]

    assert vault_item.socketed_plug_details
    assert all(value.evidence is None for value in vault_item.socketed_plug_details)
    assert all(value.sandbox_perks == [] for value in vault_item.socketed_plug_details)


def test_a_resolver_without_sandbox_tables_degrades_to_unresolved() -> None:
    thin = {key: value for key, value in DEFINITIONS.items() if key != PERK}
    context = _normalize_profile(thin, profile())
    blank = equipped_plugs(context)["Blank Description Mod"]

    assert blank.sandbox_perks == []
    assert blank.evidence.sandbox_perks == "unresolved"  # type: ignore[union-attr]
    assert effect_status(blank) == "not_retrieved"


# --- Lookups: batched, local, no new network requests ---


def test_enrichment_lookups_ask_the_resolver_to_stay_local() -> None:
    resolver = RecordingResolver(DEFINITIONS)
    normalize(profile(), resolver)

    assert resolver.local_only_calls[PERK] is True
    assert resolver.local_only_calls[CATEGORY] is True
    assert resolver.local_only_calls.get(ITEM) is False  # existing lookups are unchanged


def test_local_only_resolution_never_falls_back_to_bungie() -> None:
    fetched: list[tuple[str, set[int]]] = []

    class Resolver(DefinitionResolver):
        async def _get_manifest(self) -> dict[str, Any]:
            return {"version": "test"}

        async def _ensure_database(self) -> tuple[Any, str]:
            return "db", "test"

        def _query_database(self, path: Any, entity_type: str, hashes: set[int]) -> Any:  # type: ignore[override]
            return {value: named("Known") for value in hashes if value < 10}

        async def _fetch_fallback(self, entity_type: str, hashes: set[int]) -> Any:
            fetched.append((entity_type, set(hashes)))
            return {}

    client = SimpleNamespace(
        settings=SimpleNamespace(
            manifest_definition_cache_size=16, manifest_metadata_ttl_seconds=60
        )
    )
    resolver = Resolver(cast(Any, client))

    local = asyncio.run(resolver.resolve_many(PERK, {1, 2, 99}, local_only=True))
    assert set(local) == {1, 2} and fetched == []

    asyncio.run(resolver.resolve_many(PERK, {98}))
    assert fetched == [(PERK, {98})]  # the default path keeps its existing fallback


def test_equipped_lookups_are_bounded_to_equipped_plug_links() -> None:
    resolver = RecordingResolver(DEFINITIONS)
    context = normalize(profile(location="vault"), resolver)

    assert context.inventory.items  # the vault item is normalized
    # Nothing is equipped, so no enrichment lookup is made at all.
    assert PERK not in resolver.local_only_calls
    assert all(
        value.sandbox_perks == [] for value in context.inventory.items[0].socketed_plug_details
    )


# --- Serialization and cache compatibility ---


def test_serialization_omits_empty_evidence_fields_but_keeps_real_false() -> None:
    bare = SocketedPlugSummary(item_hash=1, name="Plain")
    off = SocketedPlugSummary(item_hash=2, name="Off", is_enabled=False, unnamed=False)

    assert set(bare.model_dump()) == {
        "item_hash",
        "name",
        "description",
        "item_type",
        "plug_type",
        "category_identifier",
    }
    assert off.model_dump()["is_enabled"] is False
    assert "unnamed" not in off.model_dump()


def test_evidence_round_trips_through_the_json_cache_shape() -> None:
    plug = equipped_plugs(normalize(profile()))["Blank Description Mod"]

    restored = SocketedPlugSummary.model_validate(plug.model_dump(mode="json"))

    assert restored == plug


def test_a_plug_cached_before_this_evidence_existed_still_loads() -> None:
    old = {"item_hash": 5, "name": "Old", "description": "d", "item_type": "Mod"}

    restored = SocketedPlugSummary.model_validate(old)

    assert restored.evidence is None and restored.sandbox_perks == []
    assert effect_status(restored) is None  # "not collected", never "verified absent"


def test_inventory_item_payloads_do_not_grow() -> None:
    context = normalize(profile(location="vault"))
    keys = set(json.loads(context.inventory.items[0].model_dump_json())["socketed_plug_details"][0])

    assert keys <= {
        "item_hash",
        "name",
        "description",
        "item_type",
        "plug_type",
        "category_identifier",
    }


# --- Tool exposure and bounds ---


def context_with(plugs: list[SocketedPlugSummary], item_name: str = "Test Gauntlets") -> Any:
    context = build_context()
    gear = context.characters[0].equipped_gear
    target = next(value for value in gear if value.item_type == "Armor")
    target.name = item_name
    target.socketed_plug_details = plugs
    target.socketed_plugs = [value.name for value in plugs]
    return context


def mechanics_plug(
    name: str, text: str, category: str = "enhancements.v2_arms", **kwargs: Any
) -> Any:
    return SocketedPlugSummary(
        item_hash=abs(hash(name)) % 10**6,
        name=name,
        description="",
        item_type="Mod",
        category_identifier=category,
        sandbox_perks=[SandboxPerkSummary(perk_hash=1, description=text)] if text else [],
        evidence=PlugEvidence(
            description="empty_in_manifest",
            sandbox_perks="present" if text else "none_declared",
            stat_effects="none_declared",
        ),
        **kwargs,
    )


def analyze(context: Any) -> dict[str, Any]:
    return GuardianToolService(context).execute(
        "analyze_current_build",
        {
            "character_id": None,
            "character_class": None,
            "goal": None,
            "activity": None,
            "locked_items": None,
            "preserve_exotics": False,
        },
    )


def test_analysis_exposes_sandbox_effects_stats_cost_and_socket_for_equipped_plugs() -> None:
    plug = mechanics_plug(
        "Effect Mod",
        "Does the thing.",
        stat_effects=[PlugStatEffect(name="Resilience", value=2)],
        energy_cost=3,
        energy_type="Any",
        socket_index=1,
        socket_category="ARMOR MODS",
        is_enabled=False,
    )
    result = analyze(context_with([plug]))
    found = next(
        value
        for value in result["equipped_armor"][0]["socketed_plugs"]
        if value["name"] == "Effect Mod"
    )

    assert found["sandbox_effects"] == ["Does the thing."]
    assert found["stats"] == {"Resilience": 2}
    assert (found["energy_cost"], found["energy_type"]) == (3, "Any")
    assert found["socket"] == "ARMOR MODS #1"
    assert found["enabled"] is False and "visible" not in found
    assert found["effect_status"] == "present"
    assert found["description"] is None  # the item-description layer is reported as it is


def test_effect_status_distinguishes_absent_from_not_retrieved() -> None:
    absent = mechanics_plug("Absent", "")
    missing = mechanics_plug("Missing", "")
    missing.evidence = PlugEvidence(
        description="definition_unavailable",
        sandbox_perks="definition_unavailable",
        stat_effects="definition_unavailable",
    )
    result = analyze(context_with([absent, missing]))
    by_name = {value["name"]: value for value in result["equipped_armor"][0]["socketed_plugs"]}

    assert by_name["Absent"]["effect_status"] == "none_in_manifest"
    assert by_name["Missing"]["effect_status"] == "not_retrieved"


def test_a_sandbox_perk_that_repeats_the_description_or_a_stat_is_not_duplicated() -> None:
    plug = mechanics_plug("Echo", "Same words.")
    plug.description = "Same words."
    plug.sandbox_perks.append(SandboxPerkSummary(perk_hash=2, description="-10 Grenade ▼"))
    result = analyze(context_with([plug]))
    found = result["equipped_armor"][0]["socketed_plugs"][0]

    assert "sandbox_effects" not in found


def test_a_plug_repeated_in_the_analysis_carries_its_mechanics_once() -> None:
    result = analyze(context_with([mechanics_plug("Repeated", "Effect text.")]))
    carriers = [
        value
        for value in _all_dicts(result)
        if value.get("name") == "Repeated" and "sandbox_effects" in value
    ]

    assert len(carriers) == 1


def _all_dicts(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for child in node.values():
            found.extend(_all_dicts(child))
    elif isinstance(node, list):
        for child in node:
            found.extend(_all_dicts(child))
    return found


def test_mechanics_in_a_response_are_bounded_and_the_overflow_is_reported() -> None:
    plugs = [
        {
            "name": f"Mod {index}",
            "role": "mod",
            "category": "enhancements.v2_arms",
            "description": None,
            "effect_status": "present",
            "sandbox_effects": ["Long effect text. " * 10],
        }
        for index in range(12)
    ]
    result = bound_plug_mechanics({"plugs": plugs}, max_chars=900)
    kept = [value for value in result["plugs"] if "sandbox_effects" in value]

    assert 0 < len(kept) < 12
    assert sum(len(json.dumps({"sandbox_effects": v["sandbox_effects"]})) for v in kept) <= 900
    assert result["mechanics_budget"] == {
        "plugs_with_mechanics": len(kept),
        "plugs_omitted": 12 - len(kept),
        "max_chars": 900,
    }
    # Plugs past the budget keep their status so the gap stays visible.
    assert all(value["effect_status"] == "present" for value in result["plugs"])


def test_bound_plug_mechanics_leaves_results_without_mechanics_untouched() -> None:
    plain = {"a": [{"name": "x", "role": "mod", "category": "c", "description": "d"}]}

    assert bound_plug_mechanics(json.loads(json.dumps(plain))) == plain


def test_build_details_lists_equipped_plug_effects_in_priority_order() -> None:
    context = context_with(
        [
            mechanics_plug("Arms Mod", "Mod effect."),
            mechanics_plug("Trait Perk", "Perk effect.", "intrinsics", socket_index=0),
        ]
    )
    result = GuardianToolService(context).execute(
        "get_build_details", {"character_id": None, "character_class": None}
    )
    effects = result["equipped_plug_effects"]

    names = [value["name"] for value in effects["plugs"]]
    assert {"Arms Mod", "Trait Perk"} <= set(names)
    assert effects["omitted"] == 0
    assert all("on" in value and "role" in value for value in effects["plugs"])
    assert next(value for value in effects["plugs"] if value["name"] == "Arms Mod")[
        "sandbox_effects"
    ] == ["Mod effect."]


def test_build_details_skips_cosmetics_and_unevidenced_plugs_and_caps_entries() -> None:
    cosmetic = mechanics_plug("Skin", "Pretty.", "armor_skins_empty")
    old_cache = SocketedPlugSummary(item_hash=9, name="Old", item_type="Mod")
    plugs = [mechanics_plug(f"Mod {index}", "e", "enhancements.v2_arms") for index in range(40)]
    context = context_with([cosmetic, old_cache, *plugs])
    gear = context.characters[0]
    effects = equipped_plug_effects(gear, max_entries=5)

    assert len(effects["plugs"]) == 5
    assert effects["omitted"] == 35
    assert "Skin" not in [value["name"] for value in effects["plugs"]]
    assert "Old" not in [value["name"] for value in effects["plugs"]]


def test_alternatives_search_output_is_unchanged_by_plug_evidence() -> None:
    context = build_context()
    result = GuardianToolService(context).execute(
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

    assert "sandbox_effects" not in json.dumps(result)


# --- The M0 evidence ledger recognizes the new evidence ---


def test_ledger_reads_sandbox_effects_so_a_claim_on_them_is_supported() -> None:
    plug = mechanics_plug(
        "Armor Charge Mod", "When grenade energy is fully expended, an Armor Charge is consumed."
    )
    result = GuardianToolService(context_with([plug])).execute(
        "get_build_details", {"character_id": None, "character_class": None}
    )
    ledger = MechanicsEvidenceLedger()
    ledger.record("get_build_details", "guardian", result)

    report = detect_mechanics_claims(
        "Armor Charge Mod consumes an Armor Charge when grenade energy is expended.", ledger
    )

    assert [value.status for value in report.claims] == ["supported"]
    assert "armor charge mod" in ledger.equipped_plugs


# --- Coverage ---


def test_effect_coverage_improves_for_plugs_whose_text_lives_in_sandbox_perks() -> None:
    plugs = list(equipped_plugs(normalize(profile())).values())
    build_affecting = [value for value in plugs if value.evidence is not None and not value.unnamed]

    before = sum(bool(value.description) for value in build_affecting)
    after = sum(effect_status(value) == "present" for value in build_affecting)

    assert after > before
    assert after == 2  # Blank Description Mod and Both Layers Perk; the rest verifiably have none


@pytest.mark.parametrize("state", ["none_in_manifest", "not_retrieved"])
def test_unevidenced_effects_stay_visible_in_build_details(state: str) -> None:
    plug = mechanics_plug("Gap", "")
    if state == "not_retrieved":
        plug.evidence = PlugEvidence(
            description="definition_unavailable",
            sandbox_perks="unresolved",
            stat_effects="none_declared",
        )
    effects = equipped_plug_effects(context_with([plug]).characters[0])

    assert effects["plugs"][0]["effect_status"] == state


def test_a_normalized_character_round_trips_through_the_refresh_cache_format() -> None:
    from app.models import CharacterSummary

    context = normalize(profile())
    character = context.characters[0]

    restored = CharacterSummary.model_validate(character.model_dump(mode="json"))

    assert restored == character
    assert restored.equipped_gear[0].socketed_plug_details[0].sandbox_perks
