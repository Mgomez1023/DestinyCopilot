"""M2: canonical equipment identity, equipment slots, and vaulted gear.

Storage (where a copy sits) and equipment slot (what the definition says) are separate. The
synthetic definitions mirror real Manifest values verified in the audit: Bungie reports a
vaulted copy in bucket 138197802 ("General", category 1, a vault location), while the item's
own definition names its slot bucket (category 3, "Equippable").
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.build_analysis import FindBuildAlternativesRequest
from app.config import Settings
from app.destiny_knowledge import DestinyKnowledgeService, ManifestKnowledgeProvider
from app.equipment_identity import equipment_slot, is_engram_variant
from app.guardian_refresh import CACHE_SCHEMA_VERSION
from app.guardian_tools import GuardianToolService
from app.models import ItemSummary
from tests.test_build_analysis import build_context, item
from tests.test_guardian import _normalize_profile

ITEM = "DestinyInventoryItemDefinition"
BUCKET = "DestinyInventoryBucketDefinition"
VAULT_BUCKET = 138197802
HELMET_BUCKET = 3448274439
GAUNTLET_BUCKET = 3551918588
ENERGY_BUCKET = 2465295065
INVISIBLE_BUCKET = 2422292810


def named(name: str, description: str = "", **extra: Any) -> dict[str, Any]:
    return {"displayProperties": {"name": name, "description": description}, **extra}


BUCKETS: dict[int, dict[str, Any]] = {
    VAULT_BUCKET: named("General", category=1, location=2),
    HELMET_BUCKET: named("Helmet", category=3, location=1),
    GAUNTLET_BUCKET: named("Gauntlets", category=3, location=1),
    ENERGY_BUCKET: named("Energy Weapons", category=3, location=1),
    INVISIBLE_BUCKET: named("", category=0, location=1),
}


def gear(
    name: str,
    *,
    item_type: int = 2,
    subtype: str = "Helmet",
    class_type: int = 0,
    bucket: int = HELMET_BUCKET,
    equippable: bool | None = True,
    tier: str = "Legendary",
    **extra: Any,
) -> dict[str, Any]:
    definition = named(
        name,
        itemType=item_type,
        itemTypeDisplayName=subtype,
        classType=class_type,
        inventory={"tierTypeName": tier, "bucketTypeHash": bucket},
        **extra,
    )
    if equippable is not None:
        definition["equippable"] = equippable
    return definition


DEFINITIONS: dict[str, dict[int, dict[str, Any]]] = {
    "DestinyClassDefinition": {10: named("Titan")},
    BUCKET: dict(BUCKETS),
    ITEM: {
        100: gear("Vaulted Helm", class_type=0),
        101: gear("Hunter Helm", class_type=1),
        102: gear("Warlock Helm", class_type=2),
        103: gear(
            "Vaulted Hand Cannon",
            item_type=3,
            subtype="Hand Cannon",
            class_type=3,
            bucket=ENERGY_BUCKET,
        ),  # fmt: skip
        104: named(
            "Enhancement Core",
            itemType=9,
            itemTypeDisplayName="Material",
            inventory={"tierTypeName": "Common", "bucketTypeHash": VAULT_BUCKET},
            equippable=False,
        ),
        # Same name, different slot and class: two real, distinct definitions.
        105: gear("Twin Piece", subtype="Gauntlets", class_type=0, bucket=GAUNTLET_BUCKET),
        106: gear("Twin Piece", class_type=1),
        # The engram preview dummy of a real item: same name, not equippable.
        107: gear("Engram Item", item_type=0, bucket=INVISIBLE_BUCKET, equippable=False),
        # Definition present but without an `equippable` flag.
        108: gear("Flagless Helm", equippable=None),
        # Definition present but its bucket is not in the Manifest tables.
        109: gear("Orphan Bucket Helm", bucket=424242),
        110: gear("Equipped Helm", class_type=0, tier="Exotic"),
    },
}


def vault_row(item_hash: int, instance_id: str) -> dict[str, Any]:
    # Bungie shape: a vaulted copy reports the vault bucket, not the gear slot.
    return {
        "itemHash": item_hash,
        "itemInstanceId": instance_id,
        "bucketHash": VAULT_BUCKET,
        "location": 2,
        "quantity": 1,
    }


def profile(vault: list[int], equipped: int | None = None) -> dict[str, Any]:
    rows = [vault_row(value, f"v{value}") for value in vault]
    equipment = (
        [{"itemHash": equipped, "itemInstanceId": f"e{equipped}", "bucketHash": HELMET_BUCKET}]
        if equipped
        else []
    )
    return {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "characterEquipment": {"data": {"c1": {"items": equipment}}},
        "profileInventory": {"data": {"items": rows}},
        "itemComponents": {"sockets": {"data": {}}},
    }


def normalized(vault: list[int], equipped: int | None = None) -> Any:
    return _normalize_profile(DEFINITIONS, profile(vault, equipped))


def by_name(context: Any) -> dict[str, ItemSummary]:
    items = [*context.inventory.items, *context.characters[0].equipped_gear]
    return {value.name: value for value in items}


# --- The slot rule ---


def test_slot_comes_from_the_definitions_equippable_bucket() -> None:
    assert equipment_slot(DEFINITIONS[ITEM][100], BUCKETS[HELMET_BUCKET]) == ("Helmet", "known")
    assert equipment_slot(DEFINITIONS[ITEM][103], BUCKETS[ENERGY_BUCKET]) == (
        "Energy Weapons",
        "known",
    )


def test_a_non_equippable_definition_has_no_slot_even_in_the_vault_bucket() -> None:
    assert equipment_slot(DEFINITIONS[ITEM][104], BUCKETS[VAULT_BUCKET]) == (
        None,
        "not_equippable",
    )


def test_an_equippable_flagged_item_in_a_non_slot_bucket_has_no_slot() -> None:
    odd = gear("Odd", bucket=VAULT_BUCKET)

    assert equipment_slot(odd, BUCKETS[VAULT_BUCKET]) == (None, "not_equippable")


def test_the_engram_dummy_is_not_equippable_and_is_recognized() -> None:
    dummy = DEFINITIONS[ITEM][107]

    assert equipment_slot(dummy, BUCKETS[INVISIBLE_BUCKET]) == (None, "not_equippable")
    assert is_engram_variant(dummy) is True
    assert is_engram_variant({**dummy, "traitIds": ["item.engram"]}) is True
    assert is_engram_variant(DEFINITIONS[ITEM][100]) is False


def test_missing_information_is_unresolved_not_guessed() -> None:
    assert equipment_slot({}, None) == (None, "unresolved")
    assert equipment_slot(DEFINITIONS[ITEM][100], None) == (None, "unresolved")
    assert equipment_slot(DEFINITIONS[ITEM][108], BUCKETS[HELMET_BUCKET]) == (None, "unresolved")
    assert equipment_slot({"equippable": True}, BUCKETS[HELMET_BUCKET]) == (None, "unresolved")


# --- Vaulted gear keeps its slot ---


def test_a_vaulted_helmet_is_stored_in_general_but_is_a_helmet() -> None:
    helm = by_name(normalized([100]))["Vaulted Helm"]

    assert helm.location == "vault"
    assert helm.bucket_name == "General"  # storage, unchanged
    assert helm.equipment_slot == "Helmet"
    assert helm.gear_slot() == "Helmet"
    assert helm.slot_status() == "known"


def test_a_vaulted_weapon_keeps_its_weapon_slot() -> None:
    cannon = by_name(normalized([103]))["Vaulted Hand Cannon"]

    assert (cannon.bucket_name, cannon.equipment_slot) == ("General", "Energy Weapons")


def test_non_equippable_inventory_gets_no_slot_and_stays_out_of_gear() -> None:
    core = by_name(normalized([104]))["Enhancement Core"]

    assert core.bucket_name == "General"
    assert core.equipment_slot is None
    assert core.gear_slot() is None
    assert core.item_type == "Material" or core.item_type != "Armor"


def test_an_equipped_item_reports_its_slot_and_its_equipped_location() -> None:
    helm = by_name(normalized([], equipped=110))["Equipped Helm"]

    assert helm.location == "equipped" and helm.is_equipped
    assert helm.equipment_slot == "Helmet"
    assert helm.bucket_name == "Helmet"


@pytest.mark.parametrize(
    ("item_hash", "restriction"),
    [(100, "Titan"), (101, "Hunter"), (102, "Warlock"), (103, "Any")],
)
def test_class_restrictions_come_from_the_definition(item_hash: int, restriction: str) -> None:
    context = normalized([item_hash])

    assert context.inventory.items[0].class_restriction == restriction


# --- Variants, missing definitions ---


def test_same_name_definitions_keep_their_own_slot_and_class() -> None:
    context = normalized([105, 106])
    twins = {value.item_hash: value for value in context.inventory.items}

    assert (twins[105].equipment_slot, twins[105].class_restriction) == ("Gauntlets", "Titan")
    assert (twins[106].equipment_slot, twins[106].class_restriction) == ("Helmet", "Hunter")


def test_the_exact_hash_decides_identity_not_the_name() -> None:
    context = normalized([105])

    assert context.inventory.items[0].item_hash == 105
    assert context.inventory.items[0].equipment_slot == "Gauntlets"


def test_an_engram_dummy_in_the_inventory_is_never_treated_as_gear() -> None:
    dummy = by_name(normalized([107]))["Engram Item"]

    assert dummy.item_type != "Armor" and dummy.item_type != "Weapon"
    assert dummy.gear_slot() is None


def test_unresolved_definitions_report_unresolved_and_no_slot() -> None:
    names = by_name(normalized([108, 109, 999]))

    assert names["Flagless Helm"].equipment_slot is None
    assert names["Flagless Helm"].slot_status() == "unresolved"
    assert names["Orphan Bucket Helm"].slot_status() == "unresolved"
    # Hash 999 has no definition at all: an unknown item, never a guessed slot.
    unknown = next(value for value in names.values() if value.name == "Unknown")
    assert unknown.gear_slot() is None


# --- Build and inventory tools ---


def titan_context_with(*extra: ItemSummary) -> Any:
    context = build_context()
    context.inventory.items = [*context.inventory.items, *extra]
    return context


def alternatives(context: Any, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
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
        "limit": 8,
        **overrides,
    }
    FindBuildAlternativesRequest.model_validate(arguments)
    return GuardianToolService(context).execute("find_build_alternatives", arguments)


def vaulted(name: str, item_type: str, slot: str, **kwargs: Any) -> ItemSummary:
    return item(name, item_type, bucket=slot, location="vault", **kwargs)


def test_a_vaulted_helmet_is_found_by_an_armor_slot_search() -> None:
    context = titan_context_with(vaulted("Vaulted Helm", "Armor", "Helmet", subtype="Helmet"))
    found = alternatives(context, slot="Helmet")["candidates"]

    assert [(value["name"], value["slot"], value["location"]) for value in found] == [
        ("Vaulted Helm", "Helmet", "vault")
    ]


def test_searching_for_the_storage_bucket_name_is_not_a_slot_search() -> None:
    context = titan_context_with(vaulted("Vaulted Helm", "Armor", "Helmet"))

    assert alternatives(context, slot="General")["candidates"] == []


def test_a_vaulted_weapon_is_found_by_its_weapon_slot() -> None:
    context = titan_context_with(
        vaulted("Vaulted Cannon", "Weapon", "Energy Weapons", subtype="Hand Cannon")
    )

    names = [value["name"] for value in alternatives(context, slot="Energy")["candidates"]]
    assert "Vaulted Cannon" in names


def test_vaulted_armor_of_another_class_is_still_excluded() -> None:
    context = titan_context_with(
        vaulted("Titan Helm", "Armor", "Helmet", class_restriction="Titan"),
        vaulted("Hunter Helm", "Armor", "Helmet", class_restriction="Hunter"),
    )
    result = alternatives(context, slot="Helmet")

    assert [value["name"] for value in result["candidates"]] == ["Titan Helm"]
    assert result["excluded_other_class_count"] == 1


def test_non_equippable_inventory_never_appears_as_an_alternative() -> None:
    material = ItemSummary(
        item_hash=7,
        name="Enhancement Core",
        item_type="Material",
        bucket_name="General",
        location="vault",
    )
    context = titan_context_with(material)

    assert all(value["name"] != "Enhancement Core" for value in alternatives(context)["candidates"])


def test_inventory_search_filters_by_equipment_slot_for_vaulted_items() -> None:
    context = titan_context_with(vaulted("Vaulted Helm", "Armor", "Helmet", subtype="Helmet"))
    result = GuardianToolService(context).execute(
        "search_inventory",
        {
            "character_id": None,
            "character_class": None,
            "query": None,
            "item_type": None,
            "subtype": None,
            "bucket": "Helmet",
            "equipped_only": False,
            "limit": 10,
        },
    )

    assert "Vaulted Helm" in [value["name"] for value in result["items"]]
    vaulted_item = next(value for value in result["items"] if value["name"] == "Vaulted Helm")
    assert vaulted_item["bucket"] == "Helmet"
    assert vaulted_item["location"] == "vault"
    assert "slot_status" not in vaulted_item  # verified slots add no noise


def test_an_item_cached_before_slots_existed_is_not_matched_and_is_disclosed() -> None:
    legacy = ItemSummary(
        item_hash=8,
        name="Legacy Vault Helm",
        item_type="Armor",
        item_subtype="Helmet",
        class_restriction="Titan",
        bucket_name="General",
        tier="Legendary",
        location="vault",
    )
    assert legacy.equipment_slot_state == "not_collected" and legacy.gear_slot() is None
    context = titan_context_with(legacy)

    result = alternatives(context, slot="Helmet")

    assert all(value["name"] != "Legacy Vault Helm" for value in result["candidates"])
    assert result["items_without_verified_slot"] >= 1
    assert any("no verified equipment slot" in text for text in result["limitations"])


def test_a_legacy_character_copy_falls_back_to_its_slot_bucket_but_a_vault_copy_does_not() -> None:
    character_copy = ItemSummary(
        item_hash=9, name="C", item_type="Armor", bucket_name="Helmet", location="character"
    )
    vault_copy = ItemSummary(
        item_hash=10, name="V", item_type="Armor", bucket_name="General", location="vault"
    )

    assert character_copy.gear_slot() == "Helmet"
    assert character_copy.slot_status() == "legacy_location_bucket"
    assert vault_copy.gear_slot() is None
    assert vault_copy.slot_status() == "unverified"


def test_unverified_gear_is_reported_with_a_status_in_tool_output() -> None:
    legacy = ItemSummary(
        item_hash=8,
        name="Legacy Vault Helm",
        item_type="Armor",
        bucket_name="General",
        location="vault",
    )
    result = GuardianToolService(titan_context_with(legacy)).execute(
        "search_inventory",
        {
            "character_id": None,
            "character_class": None,
            "query": "Legacy",
            "item_type": None,
            "subtype": None,
            "bucket": None,
            "equipped_only": False,
            "limit": 5,
        },
    )

    assert result["items"][0]["slot_status"] == "unverified"
    assert result["items"][0]["bucket"] is None


def test_equipped_state_and_ownership_are_unchanged_by_slot_identity() -> None:
    context = build_context()
    result = GuardianToolService(context).execute(
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
    armor = result["equipped_armor"][0]

    assert armor["equipped"] is True and armor["location"] == "equipped"
    assert armor["slot"] == "Chest Armor"
    assert "slot_status" not in armor


def test_another_characters_equipped_gear_is_not_offered_after_a_character_switch() -> None:
    context = build_context()
    hunter_helm = item(
        "Hunter Equipped Helm",
        "Armor",
        bucket="Helmet",
        location="equipped",
        character_id="hunter-private-id",
        equipped=True,
        class_restriction="Hunter",
    )
    other = context.characters[0].model_copy(
        update={
            "character_id": "hunter-private-id",
            "class_name": "Hunter",
            "equipped_gear": [hunter_helm],
            "subclass": None,
        }
    )
    context.characters.append(other)
    context.inventory.items.append(vaulted("Shared Titan Helm", "Armor", "Helmet"))

    titan = alternatives(context, slot="Helmet")
    names = [value["name"] for value in titan["candidates"]]

    assert "Hunter Equipped Helm" not in names
    assert "Shared Titan Helm" in names


# --- Knowledge: variants stay distinct ---


class KnowledgeResolver:
    def __init__(self) -> None:
        self.tables = {
            ITEM: {
                "200": {**gear("Variant Plate", class_type=0), "hash": 200},
                "201": {
                    **gear("Variant Plate", class_type=0, collectibleHash=5),
                    "hash": 201,
                },
                "202": {
                    **gear("Variant Plate", item_type=0, bucket=INVISIBLE_BUCKET, equippable=False),
                    "hash": 202,
                },
            },
            BUCKET: {str(key): {**value, "hash": key} for key, value in BUCKETS.items()},
        }

    async def manifest_version(self) -> str:
        return "m2-tests"

    async def load_table(self, entity_type: str) -> dict[str, Any]:
        return self.tables.get(entity_type, {})

    async def iter_definitions(self, entity_type: str):
        for value in self.tables.get(entity_type, {}).values():
            yield value

    def release_table(self, _entity_type: str) -> None:
        pass

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        table = self.tables.get(entity_type, {})
        return {value: table[str(value)] for value in entity_hashes if str(value) in table}


def details(tmp_path: Any, query: str) -> dict[str, Any]:
    provider = ManifestKnowledgeProvider(KnowledgeResolver(), Settings(manifest_cache_dir=tmp_path))  # type: ignore[arg-type]
    service = DestinyKnowledgeService([provider])
    return asyncio.run(service.execute("get_item_details", {"item_name_or_hash": query}))


def test_a_name_lookup_prefers_the_real_item_over_the_engram_dummy(tmp_path: Any) -> None:
    result = details(tmp_path, "Variant Plate")

    assert result["item"]["hash"] == 201  # equippable armor with a collectible
    assert result["item"]["equipment_slot"] == "Helmet"
    assert result["item"]["slot_status"] == "known"


def test_same_name_variants_are_reported_not_merged(tmp_path: Any) -> None:
    result = details(tmp_path, "Variant Plate")
    variants = {value["hash"]: value for value in result["same_name_variants"]}

    assert set(variants) == {200, 201, 202}
    assert variants[201]["selected"] is True and variants[201]["has_collectible"] is True
    assert variants[202]["engram_dummy"] is True and variants[202]["equippable"] is False
    assert variants[202]["slot_status"] == "not_equippable"
    assert "equipment_slot" in result["variants_differ_in"]


def test_an_exact_hash_returns_that_definition_even_when_it_is_the_dummy(tmp_path: Any) -> None:
    result = details(tmp_path, "202")

    assert result["item"]["hash"] == 202
    assert result["item"]["slot_status"] == "not_equippable"
    assert result["item"]["equipment_slot"] is None
    assert result["resolution"]["match_kind"] == "hash"


def test_an_unambiguous_item_reports_no_variant_block(tmp_path: Any) -> None:
    resolver = KnowledgeResolver()
    del resolver.tables[ITEM]["200"], resolver.tables[ITEM]["202"]
    provider = ManifestKnowledgeProvider(resolver, Settings(manifest_cache_dir=tmp_path))  # type: ignore[arg-type]
    result = asyncio.run(
        DestinyKnowledgeService([provider]).execute(
            "get_item_details", {"item_name_or_hash": "Variant Plate"}
        )
    )

    assert "same_name_variants" not in result
    assert result["item"]["equipment_slot"] == "Helmet"


# --- Serialization and cache ---


def test_slot_fields_are_compact_in_json_and_round_trip() -> None:
    helm = by_name(normalized([100]))["Vaulted Helm"]
    payload = helm.model_dump(mode="json")

    assert payload["equipment_slot"] == "Helmet"
    assert "equipment_slot_state" not in payload  # "known" is implied by a set slot
    assert ItemSummary.model_validate(payload) == helm


def test_unresolved_and_not_equippable_states_survive_serialization() -> None:
    names = by_name(normalized([108, 104]))

    flagless = ItemSummary.model_validate(names["Flagless Helm"].model_dump(mode="json"))
    assert flagless.equipment_slot_state == "unresolved"
    assert "equipment_slot" not in names["Flagless Helm"].model_dump()


def test_an_item_payload_cached_before_slots_loads_as_not_collected() -> None:
    old = {
        "item_hash": 1,
        "name": "Old",
        "item_type": "Armor",
        "bucket_name": "General",
        "location": "vault",
    }

    restored = ItemSummary.model_validate(old)

    assert restored.equipment_slot is None
    assert restored.equipment_slot_state == "not_collected"
    assert restored.gear_slot() is None


def test_non_gear_inventory_payloads_do_not_grow() -> None:
    keys = set(json.loads(by_name(normalized([104]))["Enhancement Core"].model_dump_json()))

    assert "equipment_slot" not in keys and "equipment_slot_state" not in keys


def test_the_guardian_cache_schema_covers_m1_m2_and_m3b_fields() -> None:
    assert CACHE_SCHEMA_VERSION >= 5


# --- The real Manifest (skipped when it is not available locally) ---


def _manifest_path() -> Path | None:
    try:
        from evals.runtime import DEFAULT_MANIFEST
    except Exception:  # noqa: BLE001
        return None
    return DEFAULT_MANIFEST if DEFAULT_MANIFEST and Path(DEFAULT_MANIFEST).is_file() else None


REAL = _manifest_path()


def real(table: str, entity_hash: int) -> dict[str, Any] | None:
    assert REAL is not None
    with sqlite3.connect(str(REAL)) as connection:
        signed = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
        row = connection.execute(f'SELECT json FROM "{table}" WHERE id = ?', (signed,)).fetchone()
    return json.loads(row[0]) if row else None


pytestmark_real = pytest.mark.skipif(REAL is None, reason="local Manifest not available")


@pytestmark_real
def test_real_vault_bucket_is_a_vault_location_not_an_equipment_slot() -> None:
    general = real(BUCKET, VAULT_BUCKET)

    assert general is not None
    assert general["displayProperties"]["name"] == "General"
    assert general["category"] == 1 and general["location"] == 2


@pytestmark_real
@pytest.mark.parametrize(
    ("item_hash", "slot", "restriction"),
    [
        (2907129557, "Energy Weapons", 3),  # Sunshot
        (2002759681, "Gauntlets", 0),  # Synthoceps, the collectible variant
        (3960926756, "Helmet", 1),  # Celestial Nighthawk
        (2553961985, "Helmet", 2),  # Verity's Brow
        (943320520, "Helmet", 0),  # Helm of Saint-14
    ],
)
def test_real_definitions_yield_their_slot_and_class(
    item_hash: int, slot: str, restriction: int
) -> None:
    definition = real(ITEM, item_hash)
    assert definition is not None
    bucket = real(BUCKET, definition["inventory"]["bucketTypeHash"])

    assert equipment_slot(definition, bucket) == (slot, "known")
    assert definition["classType"] == restriction


@pytestmark_real
def test_real_engram_dummy_shares_the_name_but_has_no_slot() -> None:
    dummy = real(ITEM, 3966240957)  # the Synthoceps engram dummy
    assert dummy is not None
    real_item = real(ITEM, 2002759681)
    assert real_item is not None
    bucket = real(BUCKET, dummy["inventory"]["bucketTypeHash"])

    assert dummy["displayProperties"]["name"] == real_item["displayProperties"]["name"]
    assert equipment_slot(dummy, bucket) == (None, "not_equippable")
    assert is_engram_variant(dummy) and not is_engram_variant(real_item)


@pytestmark_real
def test_real_vault_row_normalizes_to_a_slot_through_the_real_normalizer() -> None:
    from app.bungie.guardian import GuardianNormalizer
    from evals.runtime import LocalManifestResolver

    helm_hash = 943320520  # Helm of Saint-14
    raw = {
        "characters": {
            "data": {"c1": {"classHash": 3655393761, "light": 1, "minutesPlayedTotal": "0"}}
        },
        "characterEquipment": {"data": {"c1": {"items": []}}},
        "profileInventory": {"data": {"items": [vault_row(helm_hash, "v-real")]}},
        "itemComponents": {"sockets": {"data": {}}},
    }
    context = asyncio.run(
        GuardianNormalizer(LocalManifestResolver(REAL)).normalize(  # type: ignore[arg-type]
            {},
            {"membershipId": "1", "membershipType": 3, "displayName": "Guardian"},
            {"profile": {"data": {"minutesPlayedTotal": "0"}}, **raw},
        )
    )
    helm = context.inventory.items[0]

    assert (helm.bucket_name, helm.equipment_slot) == ("General", "Helmet")
    assert helm.class_restriction == "Titan"
    assert helm.location == "vault"
