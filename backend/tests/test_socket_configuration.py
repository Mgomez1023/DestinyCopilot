"""M3A: what the Manifest declares about an item's sockets.

Pure classification tests use synthetic definitions in the real Manifest shapes (verified in
the audit and again in the real-Manifest tests at the end). A randomized roll pool is a list of
possible rolls, never a menu; a fake plug set never implies a selectable option.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app import socket_configuration
from app.config import Settings
from app.destiny_knowledge import DestinyKnowledgeService, ManifestKnowledgeProvider
from app.socket_configuration import (
    MAX_POOL_FOR_NAME_COUNT,
    MAX_SOCKETS_IN_OUTPUT,
    build_specs,
    classify_socket,
    describe_sockets,
    name_hashes_needed,
    referenced_definition_hashes,
)

ITEM = "DestinyInventoryItemDefinition"
PLUG_SET = "DestinyPlugSetDefinition"
SOCKET_TYPE = "DestinySocketTypeDefinition"
CATEGORY = "DestinySocketCategoryDefinition"
BUCKET = "DestinyInventoryBucketDefinition"

PERKS, MODS, COSMETICS = 7001, 7002, 7003


def named(name: str, **extra: Any) -> dict[str, Any]:
    return {"displayProperties": {"name": name, "description": ""}, **extra}


def option(plug_hash: int, *, can_roll: bool = True) -> dict[str, Any]:
    return {"plugItemHash": plug_hash, "currentlyCanRoll": can_roll}


def plug_set(*options: dict[str, Any], fake: bool = False) -> dict[str, Any]:
    return {"isFakePlugSet": fake, "reusablePlugItems": list(options)}


PLUG_NAMES = {
    10: "Alpha Barrel",
    11: "Beta Barrel",
    12: "Gamma Barrel",
    13: "Gamma Barrel",  # same name, different hash (a base and an enhanced variant)
    20: "Solo Trait",
    30: "Mod One",
    31: "Mod Two",
    40: "Skin One",
}
SOCKET_TYPES = {
    1: {
        "plugWhitelist": [
            {"categoryIdentifier": "barrels"},
            {"categoryIdentifier": "crafting.recipes.empty_socket"},
        ]
    },
    2: {"plugWhitelist": [{"categoryIdentifier": "frames"}]},
    3: {"plugWhitelist": [{"categoryIdentifier": "v400.weapon.mod_guns"}]},
    4: {"plugWhitelist": [{"categoryIdentifier": "shader"}]},
}
CATEGORIES = {
    PERKS: named("WEAPON PERKS"),
    MODS: named("WEAPON MODS"),
    COSMETICS: named("WEAPON COSMETICS"),
}
PLUG_SETS = {
    100: plug_set(option(20), fake=True),  # fake, one option: fixed
    101: plug_set(option(10), option(11), option(12), option(13)),  # real, four options
    102: plug_set(option(10, can_roll=False), option(11), option(12), option(10)),  # duplicates
    103: plug_set(option(10), option(11), fake=True),  # fake, several: display only
    104: plug_set(option(30)),  # real, one option
    105: plug_set(option(30), option(31)),  # real mod pool
}


def socket(**values: Any) -> dict[str, Any]:
    return {"socketTypeHash": 2, "plugSources": 2, **values}


def item(
    *sockets: dict[str, Any], categories: dict[int, list[int]] | None = None, **extra: Any
) -> dict[str, Any]:
    layout = categories or {PERKS: list(range(len(sockets)))}
    return {
        "equippable": True,
        "sockets": {
            "socketEntries": list(sockets),
            "socketCategories": [
                {"socketCategoryHash": key, "socketIndexes": value} for key, value in layout.items()
            ],
        },
        **extra,
    }


def names() -> dict[int, dict[str, Any]]:
    return {key: named(value) for key, value in PLUG_NAMES.items()}


def classify(
    entry: dict[str, Any], plug_sets: dict[int, dict[str, Any]] | None = None
) -> str | None:
    specs = build_specs(item(entry), plug_sets if plug_sets is not None else PLUG_SETS)
    return classify_socket(specs[0])


def describe(
    definition: dict[str, Any], plug_sets: dict[int, dict[str, Any]] | None = None
) -> dict[str, Any]:
    specs = build_specs(definition, plug_sets if plug_sets is not None else PLUG_SETS)
    return describe_sockets(definition, specs, SOCKET_TYPES, CATEGORIES, names())


# --- Classification rules ---


def test_a_fake_plug_set_with_one_option_is_fixed() -> None:
    assert (
        classify(socket(reusablePlugSetHash=100, singleInitialItemHash=20)) == "fixed_by_manifest"
    )


def test_a_real_plug_set_with_one_option_is_fixed_when_nothing_else_supplies_options() -> None:
    assert (
        classify(socket(reusablePlugSetHash=104, singleInitialItemHash=30)) == "fixed_by_manifest"
    )


def test_an_inventory_sourced_socket_is_never_called_fixed() -> None:
    entry = socket(reusablePlugSetHash=104, singleInitialItemHash=30, plugSources=3)

    assert classify(entry) == "inventory_sourced"


def test_a_randomized_plug_set_is_a_roll_pool_not_a_choice() -> None:
    assert (
        classify(socket(randomizedPlugSetHash=101, singleInitialItemHash=10))
        == "randomized_roll_pool"
    )


def test_a_real_multi_option_set_is_a_reusable_pool() -> None:
    assert (
        classify(socket(reusablePlugSetHash=105, singleInitialItemHash=30)) == "reusable_plug_pool"
    )


def test_a_fake_plug_set_with_several_options_is_display_only() -> None:
    assert (
        classify(socket(reusablePlugSetHash=103, singleInitialItemHash=10))
        == "display_only_options"
    )


def test_inline_reusable_items_form_a_pool_and_count_distinct_hashes() -> None:
    entry = socket(reusablePlugItems=[option(10), option(11), option(10)])
    specs = build_specs(item(entry), PLUG_SETS)

    assert specs[0].pools[0].kind == "inline" and specs[0].pools[0].options == 2
    assert classify_socket(specs[0]) == "reusable_plug_pool"


def test_a_single_inline_item_is_fixed() -> None:
    assert (
        classify(socket(reusablePlugItems=[option(10)], singleInitialItemHash=10))
        == "fixed_by_manifest"
    )


def test_a_missing_plug_set_is_insufficient_information_not_a_guess() -> None:
    assert (
        classify(socket(reusablePlugSetHash=999, singleInitialItemHash=10))
        == "insufficient_information"
    )
    assert classify(socket(randomizedPlugSetHash=999)) == "insufficient_information"


def test_one_unreadable_pool_makes_the_socket_insufficient_even_if_another_resolved() -> None:
    entry = socket(randomizedPlugSetHash=101, reusablePlugSetHash=999)

    assert classify(entry) == "insufficient_information"


def test_a_typed_socket_with_nothing_declared_is_insufficient_information() -> None:
    assert classify(socket()) == "insufficient_information"


def test_an_empty_socket_slot_is_unused() -> None:
    assert classify({"socketTypeHash": 0, "plugSources": 0}) is None


def test_multiple_possible_rolls_do_not_make_a_socket_player_selectable() -> None:
    result = describe(item(socket(randomizedPlugSetHash=101, singleInitialItemHash=10)))
    entry = result["sockets"][0]

    assert entry["classification"] == "randomized_roll_pool"
    assert not any("select" in key for key in entry)
    assert "not choices" in result["note"]


# --- Pools: counts, duplicates, samples ---


def test_pool_options_are_distinct_hashes_and_distinct_names_are_reported_apart() -> None:
    entry = describe(item(socket(randomizedPlugSetHash=101, singleInitialItemHash=10)))["sockets"][
        0
    ]
    pool = entry["pool"]

    assert pool["options"] == 4  # four hashes
    assert pool["distinct_names"] == 3  # two share a name
    assert pool["set_hash"] == 101
    assert entry["initial_plug"] == "Alpha Barrel"


def test_repeated_hashes_within_a_set_count_once_and_can_roll_is_counted() -> None:
    pool = describe(item(socket(randomizedPlugSetHash=102)))["sockets"][0]["pool"]

    assert pool["options"] == 3  # the repeated hash collapses
    assert pool["rollable"] == 2  # one distinct option cannot roll


def test_a_randomized_pool_alongside_a_reusable_pool_reports_both_without_merging() -> None:
    entry = describe(item(socket(randomizedPlugSetHash=101, reusablePlugSetHash=105)))["sockets"][0]

    assert entry["pool"]["kind"] == "randomized"
    assert entry["other_pools"] == [{"kind": "reusable", "options": 2}]


def test_perk_pools_carry_a_small_sample_and_mod_pools_only_counts() -> None:
    result = describe(
        item(
            socket(randomizedPlugSetHash=101),
            socket(socketTypeHash=3, reusablePlugSetHash=105, plugSources=3),
            categories={PERKS: [0], MODS: [1]},
        )
    )
    perk, mod = result["sockets"]

    assert perk["pool"]["sample"] == ["Alpha Barrel", "Beta Barrel", "Gamma Barrel"]
    assert mod == {
        "index": 1,
        "category": "WEAPON MODS",
        "classification": "reusable_plug_pool",
        "accepts": ["v400.weapon.mod_guns"],
        "options": 2,
    }


def test_a_very_large_pool_is_counted_without_resolving_every_name() -> None:
    big = plug_set(*(option(1000 + value) for value in range(700)))
    specs = build_specs(item(socket(randomizedPlugSetHash=900)), {900: big})

    needed = name_hashes_needed(specs, CATEGORIES)
    pool = describe_sockets(
        item(socket(randomizedPlugSetHash=900)), specs, SOCKET_TYPES, CATEGORIES, {}
    )
    summary = pool["sockets"][0]["pool"]

    assert len(needed) <= socket_configuration.MAX_SAMPLE_NAMES
    assert summary["options"] == 700 and "distinct_names" not in summary
    assert MAX_POOL_FOR_NAME_COUNT < 700


def test_every_referenced_definition_is_listed_for_one_batched_lookup() -> None:
    definition = item(
        socket(randomizedPlugSetHash=101, socketTypeHash=1),
        socket(reusablePlugSetHash=105, socketTypeHash=3),
        categories={PERKS: [0], MODS: [1]},
    )

    assert referenced_definition_hashes(definition) == {
        "plug_sets": {101, 105},
        "socket_types": {1, 3},
        "categories": {PERKS, MODS},
    }


# --- Output bounds and omissions ---


def test_cosmetic_unused_and_uncategorized_sockets_are_omitted_but_counted() -> None:
    result = describe(
        item(
            socket(randomizedPlugSetHash=101),
            socket(socketTypeHash=4, reusablePlugSetHash=105),  # a shader socket
            {"socketTypeHash": 0, "plugSources": 0},  # unused
            socket(reusablePlugSetHash=104, singleInitialItemHash=30),  # no category
            categories={PERKS: [0], COSMETICS: [1]},
        )
    )

    assert [value["index"] for value in result["sockets"]] == [0]
    assert result["omitted"] == {"unused": 1, "cosmetic": 1, "uncategorized": 1}
    assert sum(result["classification_counts"].values()) == 3


def test_the_socket_list_is_capped_and_the_overflow_is_reported() -> None:
    sockets = [socket(randomizedPlugSetHash=101) for _ in range(MAX_SOCKETS_IN_OUTPUT + 4)]
    result = describe(item(*sockets))

    assert len(result["sockets"]) == MAX_SOCKETS_IN_OUTPUT
    assert result["omitted"]["over_limit"] == 4


def test_a_partial_status_names_the_unreadable_plug_set() -> None:
    result = describe(item(socket(reusablePlugSetHash=999, singleInitialItemHash=10)))

    assert result["status"] == "partial"
    assert result["sockets"][0]["unresolved_plug_sets"] == [999]


def test_the_module_names_no_items() -> None:
    source = Path(socket_configuration.__file__).read_text(encoding="utf-8").casefold()

    for name in ("sunshot", "austringer", "synthoceps", "dead man", "rampage"):
        assert name not in source


# --- Through the knowledge provider ---


class Resolver:
    """A resolver double that records lookups and whether they asked to stay local."""

    def __init__(self, with_local_only: bool = True) -> None:
        item_tables = {
            "300": {
                **named(
                    "Test Rifle",
                    itemType=3,
                    itemTypeDisplayName="Pulse Rifle",
                    classType=3,
                    equippable=True,
                    inventory={"tierTypeName": "Legendary", "bucketTypeHash": 50},
                    collectibleHash=1,
                ),
                "hash": 300,
                "sockets": item(
                    socket(randomizedPlugSetHash=101, singleInitialItemHash=10, socketTypeHash=1),
                    socket(reusablePlugSetHash=100, singleInitialItemHash=20, socketTypeHash=2),
                    categories={PERKS: [0, 1]},
                )["sockets"],
            },
            "301": {
                **named(
                    "Test Rifle",
                    itemType=0,
                    itemTypeDisplayName="Pulse Rifle",
                    classType=3,
                    equippable=False,
                    inventory={"tierTypeName": "Legendary", "bucketTypeHash": 51},
                ),
                "hash": 301,
                "sockets": {"socketEntries": [{"socketTypeHash": 0}]},
            },
            "302": {
                **named(
                    "Test Mod",
                    itemType=19,
                    itemTypeDisplayName="Mod",
                    equippable=False,
                    inventory={"tierTypeName": "Common", "bucketTypeHash": 52},
                ),
                "hash": 302,
            },
            "303": {
                **named(
                    "Bare Gun",
                    itemType=3,
                    itemTypeDisplayName="Pulse Rifle",
                    classType=3,
                    equippable=True,
                    inventory={"tierTypeName": "Common", "bucketTypeHash": 50},
                ),
                "hash": 303,
            },
        }
        for plug_hash, plug_name in PLUG_NAMES.items():
            item_tables[str(plug_hash)] = {**named(plug_name), "hash": plug_hash}
        self.tables = {
            ITEM: item_tables,
            PLUG_SET: {str(key): value for key, value in PLUG_SETS.items()},
            SOCKET_TYPE: {str(key): value for key, value in SOCKET_TYPES.items()},
            CATEGORY: {str(key): value for key, value in CATEGORIES.items()},
            BUCKET: {
                "50": named("Kinetic Weapons", category=3),
                "51": named("", category=0),
                "52": named("", category=0),
            },
        }
        self.calls: list[tuple[str, int, bool | None]] = []
        if not with_local_only:
            self.resolve_many = self._plain  # type: ignore[method-assign]

    async def manifest_version(self) -> str:
        return "m3a-tests"

    async def load_table(self, entity_type: str) -> dict[str, Any]:
        return self.tables.get(entity_type, {})

    async def iter_definitions(self, entity_type: str):
        for value in self.tables.get(entity_type, {}).values():
            yield value

    def release_table(self, _entity_type: str) -> None:
        pass

    async def _plain(self, entity_type: str, entity_hashes: set[int]) -> dict[int, dict[str, Any]]:
        self.calls.append((entity_type, len(entity_hashes), None))
        table = self.tables.get(entity_type, {})
        return {value: table[str(value)] for value in entity_hashes if str(value) in table}

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int], *, local_only: bool = False
    ) -> dict[int, dict[str, Any]]:
        self.calls.append((entity_type, len(entity_hashes), local_only))
        table = self.tables.get(entity_type, {})
        return {value: table[str(value)] for value in entity_hashes if str(value) in table}


def details(
    tmp_path: Any, query: str, resolver: Resolver | None = None
) -> tuple[dict[str, Any], Resolver]:
    resolver = resolver or Resolver()
    provider = ManifestKnowledgeProvider(resolver, Settings(manifest_cache_dir=tmp_path))  # type: ignore[arg-type]
    result = asyncio.run(
        DestinyKnowledgeService([provider]).execute(
            "get_item_details", {"item_name_or_hash": query}
        )
    )
    return result, resolver


def test_item_details_gain_socket_configuration_and_keep_existing_fields(tmp_path: Any) -> None:
    result, _ = details(tmp_path, "300")

    assert {"item", "resolution", "limitations", "source", "found"} <= set(result)
    assert result["item"]["hash"] == 300 and result["item"]["equipment_slot"] == "Kinetic Weapons"
    config = result["socket_configuration"]
    assert config["status"] == "verified"
    kinds = {value["index"]: value["classification"] for value in config["sockets"]}
    assert kinds == {0: "randomized_roll_pool", 1: "fixed_by_manifest"}


def test_socket_lookups_are_batched_and_stay_local(tmp_path: Any) -> None:
    _, resolver = details(tmp_path, "300")
    config_calls = [call for call in resolver.calls if call[0] in {PLUG_SET, SOCKET_TYPE, CATEGORY}]

    assert {call[0] for call in config_calls} == {PLUG_SET, SOCKET_TYPE, CATEGORY}
    assert len(config_calls) == 3  # one batched lookup per definition type
    assert all(call[2] is True for call in config_calls)


def test_a_resolver_without_the_local_only_option_still_works(tmp_path: Any) -> None:
    result, _ = details(tmp_path, "300", Resolver(with_local_only=False))

    assert result["socket_configuration"]["status"] == "verified"


def test_non_equippable_definitions_report_not_applicable(tmp_path: Any) -> None:
    dummy, _ = details(tmp_path, "301")
    mod, _ = details(tmp_path, "302")

    assert dummy["socket_configuration"] == {
        "status": "not_applicable",
        "reason": "not_equippable",
        "basis": "manifest_definition",
    }
    assert mod["socket_configuration"]["status"] == "not_applicable"


def test_equippable_items_without_sockets_say_so(tmp_path: Any) -> None:
    result, _ = details(tmp_path, "303")

    assert result["socket_configuration"]["status"] == "no_sockets_declared"


def test_a_failing_socket_lookup_never_breaks_item_details(tmp_path: Any) -> None:
    resolver = Resolver()
    original = resolver.resolve_many

    async def flaky(entity_type: str, hashes: set[int], **kwargs: Any) -> Any:
        if entity_type == PLUG_SET:
            raise RuntimeError("database unavailable")
        return await original(entity_type, hashes, **kwargs)

    resolver.resolve_many = flaky  # type: ignore[method-assign]
    result, _ = details(tmp_path, "300", resolver)

    assert result["found"] is True
    assert result["socket_configuration"] == {
        "status": "unresolved",
        "basis": "manifest_definition",
    }


def test_variants_report_when_their_socket_counts_differ(tmp_path: Any) -> None:
    result, _ = details(tmp_path, "300")
    variants = {value["hash"]: value for value in result["same_name_variants"]}

    assert variants[300]["selected"] is True
    assert variants[301]["socket_entries"] == 1  # the dummy has a different socket layout


# --- The real Manifest (skipped when it is not available locally) ---


def _manifest_path() -> Path | None:
    try:
        from evals.runtime import DEFAULT_MANIFEST
    except Exception:  # noqa: BLE001
        return None
    return DEFAULT_MANIFEST if DEFAULT_MANIFEST and Path(DEFAULT_MANIFEST).is_file() else None


REAL = _manifest_path()
needs_manifest = pytest.mark.skipif(REAL is None, reason="local Manifest not available")


def real_definition(table: str, entity_hash: int) -> dict[str, Any]:
    assert REAL is not None
    signed = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
    with sqlite3.connect(str(REAL)) as connection:
        row = connection.execute(f'SELECT json FROM "{table}" WHERE id = ?', (signed,)).fetchone()
    assert row is not None, (table, entity_hash)
    return json.loads(row[0])


def real_config(entity_hash: int) -> dict[str, Any]:
    from evals.runtime import LocalManifestResolver

    assert REAL is not None
    provider = ManifestKnowledgeProvider(LocalManifestResolver(REAL), Settings())  # type: ignore[arg-type]
    definition = real_definition(ITEM, entity_hash)
    return asyncio.run(provider._socket_configuration(definition))


def sockets_by_index(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {value["index"]: value for value in config["sockets"]}


@needs_manifest
def test_real_sunshot_barrel_magazine_and_grip_are_fixed_single_option_fake_sets() -> None:
    sunshot = real_definition(ITEM, 2907129557)
    config = sockets_by_index(real_config(2907129557))
    expected = {1: ("barrels", 850), 2: ("magazines", 210), 4: ("grips", 448)}

    for index, (accepts, set_hash) in expected.items():
        entry = config[index]
        assert entry["classification"] == "fixed_by_manifest", entry
        assert accepts in entry["accepts"] and entry["set_hash"] == set_hash
        entry_definition = sunshot["sockets"]["socketEntries"][index]
        assert entry_definition["reusablePlugSetHash"] == set_hash
        assert (
            "randomizedPlugSetHash" not in entry_definition
            or not entry_definition["randomizedPlugSetHash"]
        )
        real_set = real_definition(PLUG_SET, set_hash)
        assert real_set["isFakePlugSet"] is True and len(real_set["reusablePlugItems"]) == 1


@needs_manifest
def test_real_sunshot_has_no_socket_the_manifest_alone_calls_changeable() -> None:
    config = real_config(2907129557)
    changeable = {"randomized_roll_pool", "reusable_plug_pool"}

    perk_sockets = [value for value in config["sockets"] if value["category"] != "WEAPON MODS"]
    assert all(value["classification"] not in changeable for value in perk_sockets)
    assert config["classification_counts"].get("randomized_roll_pool", 0) == 0


@needs_manifest
def test_real_austringer_barrel_is_a_randomized_roll_pool_not_a_choice() -> None:
    barrel = sockets_by_index(real_config(3055790362))[1]
    definition = real_definition(ITEM, 3055790362)["sockets"]["socketEntries"][1]

    assert barrel["classification"] == "randomized_roll_pool"
    assert definition["randomizedPlugSetHash"] == barrel["pool"]["set_hash"] == 2044988252
    assert barrel["pool"]["options"] == 22
    assert barrel["pool"]["distinct_names"] == 11  # many hashes share names
    assert barrel["initial_plug"] == "Corkscrew Rifling"  # a default, not a guaranteed roll
    assert "selectable" not in json.dumps(barrel)


@needs_manifest
def test_real_exotic_with_randomized_perks_differs_from_a_fixed_exotic() -> None:
    random_exotic = real_config(2188764214)  # an Exotic whose perk columns are randomized
    fixed_exotic = real_config(2907129557)

    assert random_exotic["classification_counts"].get("randomized_roll_pool", 0) >= 4
    assert fixed_exotic["classification_counts"].get("randomized_roll_pool", 0) == 0
    assert (
        real_definition(ITEM, 2188764214)["inventory"]["tierTypeName"]
        == real_definition(ITEM, 2907129557)["inventory"]["tierTypeName"]
        == "Exotic"
    )


@needs_manifest
def test_real_same_name_definitions_keep_their_own_socket_behavior() -> None:
    first, second = real_config(2188764214), real_config(3654674561)

    assert first["status"] == second["status"] == "verified"
    assert first["classification_counts"] != second["classification_counts"]


@needs_manifest
def test_real_display_only_sets_are_never_called_selectable() -> None:
    definition = real_definition(ITEM, 2907129557)
    wanted = referenced_definition_hashes(definition)["plug_sets"]
    plug_sets = {value: real_definition(PLUG_SET, value) for value in wanted}
    display_only = [
        spec
        for spec in build_specs(definition, plug_sets)
        if classify_socket(spec) == "display_only_options"
    ]

    assert display_only  # Sunshot's tracker and ornament sockets list several display options
    for spec in display_only:
        assert all(pool.fake is True and pool.options >= 2 for pool in spec.pools)
    # Those display-only sockets are cosmetic (tracker, ornament): omitted, never "selectable".
    config = real_config(2907129557)
    assert config["omitted"]["cosmetic"] >= len(display_only)
    assert "display_only_options" not in config["classification_counts"]


@needs_manifest
def test_real_weapon_with_several_plug_hashes_per_name_reports_both_counts() -> None:
    definition = real_definition(ITEM, 2150012406)
    origin = definition["sockets"]["socketEntries"][8]
    hashes = {value["plugItemHash"] for value in origin["reusablePlugItems"]}
    names = {real_definition(ITEM, value)["displayProperties"]["name"] for value in sorted(hashes)}
    pool = sockets_by_index(real_config(2150012406))[8]["pool"]

    assert not origin.get("reusablePlugSetHash") and len(hashes) == 4 and len(names) == 2
    assert pool["kind"] == "inline"
    assert (pool["options"], pool["distinct_names"]) == (4, 2)


@needs_manifest
def test_real_engram_dummy_has_no_configuration_and_exact_hash_is_honored(tmp_path: Any) -> None:
    from evals.runtime import LocalManifestResolver

    assert REAL is not None
    provider = ManifestKnowledgeProvider(
        LocalManifestResolver(REAL), Settings(manifest_cache_dir=tmp_path)
    )  # type: ignore[arg-type]
    result = asyncio.run(
        DestinyKnowledgeService([provider]).execute(
            "get_item_details", {"item_name_or_hash": "3966240957"}
        )
    )

    assert result["item"]["hash"] == 3966240957
    assert result["socket_configuration"]["status"] == "not_applicable"


@needs_manifest
def test_real_synthoceps_variants_with_different_socket_layouts_are_visible(tmp_path: Any) -> None:
    from evals.runtime import LocalManifestResolver

    assert REAL is not None
    provider = ManifestKnowledgeProvider(
        LocalManifestResolver(REAL), Settings(manifest_cache_dir=tmp_path)
    )  # type: ignore[arg-type]
    result = asyncio.run(
        DestinyKnowledgeService([provider]).execute(
            "get_item_details", {"item_name_or_hash": "2002759681"}
        )
    )
    layouts = {
        value["hash"]: value.get("socket_entries", "same") for value in result["same_name_variants"]
    }

    assert layouts[1035680666] == 12 and layouts[2002759681] == "same"
