"""Exact-match resolution, quest rewards, sub-areas and index schema 5 for the Manifest
provider. Fixture shapes mirror the real Manifest (hashes taken from it where noted)."""

import asyncio
import json
from typing import Any

import pytest

from app.config import Settings
from app.core_knowledge import normalize_text
from app.destiny_knowledge import KNOWLEDGE_INDEX_SCHEMA_VERSION, ManifestKnowledgeProvider
from tests.test_destiny_knowledge import CountingIndexResolver, FakeResolver, display

MOON_PLACE = 3325508439
EARTH_PLACE = 3747705955
MOON = 290444260  # The Moon destination that declares bubbles.
MOON_WITHOUT_BUBBLES = 2087192301  # A second, bubble-less "The Moon" definition.
MOON_DUPLICATE = 677774031  # A third "The Moon" that declares the same bubbles.
EDZ = 697502628
COSMODROME = 3990611421
ARC_SCOURGE = 1405737718
LEGENDARY_PAYOUT = 2738452221
RETURNERS_GIFTS = 4134078229


def bubble(bubble_hash: int, name: str) -> dict[str, Any]:
    return {
        "hash": bubble_hash,
        "displayProperties": {"description": "", "name": name, "hasIcon": False},
    }


def destination(destination_hash: int, name: str, place: int, bubbles: list[Any]) -> dict:
    return {
        **display(name),
        "hash": destination_hash,
        "placeHash": place,
        "bubbles": bubbles,
    }


def resolution_tables() -> dict[str, dict[str, Any]]:
    tables = FakeResolver().tables
    items = tables["DestinyInventoryItemDefinition"]
    items[str(ARC_SCOURGE)] = {
        **display("Arc Scourge", "Defeat combatants with Arc damage."),
        "hash": ARC_SCOURGE,
        "itemType": 26,
        "itemTypeDisplayName": "Duality Order",
        "inventory": {"tierTypeName": "Legendary"},
        "value": {
            "itemValue": [
                {"itemHash": LEGENDARY_PAYOUT, "quantity": 1},
                {"itemHash": RETURNERS_GIFTS, "quantity": 1},
                {"itemHash": 0, "quantity": 0},
                {"itemHash": 0, "quantity": 0},
            ],
            "valueDescription": "",
        },
    }
    items[str(LEGENDARY_PAYOUT)] = {**display("Legendary Payout"), "hash": LEGENDARY_PAYOUT}
    items[str(RETURNERS_GIFTS)] = {
        **display("Returners Gifts Unlocked"),
        "hash": RETURNERS_GIFTS,
    }
    reward_hashes = [501, 502, 503, 504, 505, 509]  # 509 has no definition.
    for index, reward_hash in enumerate(reward_hashes[:5]):
        items[str(reward_hash)] = {**display(f"Reward {index + 1}"), "hash": reward_hash}
    items["500"] = {
        **display("Generous Bounty", "A bounty declaring many rewards."),
        "hash": 500,
        "itemType": 26,
        "itemTypeDisplayName": "Weekly Bounty",
        "value": {"itemValue": [{"itemHash": value, "quantity": 1} for value in reward_hashes]},
    }
    # The real Manifest also has a non-activity item literally named "Altars of Sorrow".
    items["510"] = {
        **display("Altars of Sorrow"),
        "hash": 510,
        "itemType": 20,
        "itemTypeDisplayName": "Material",
    }
    items["520"] = {
        **display("Moonfang", "An emblem."),
        "hash": 520,
        "itemType": 14,
        "itemTypeDisplayName": "Emblem",
    }
    items["521"] = {
        **display("Moonfang-X7 Rifle", "A rifle."),
        "hash": 521,
        "itemType": 3,
        "itemTypeDisplayName": "Auto Rifle",
        "collectibleHash": 2,
    }
    tables["DestinyActivityDefinition"]["700"] = {
        **display("Haunted Altars of Sorrow", "Defeat the Nightmares on the Moon."),
        "hash": 700,
        "activityTypeHash": 11,
        "destinationHash": MOON,
        "placeHash": MOON_PLACE,
        "matchmaking": {"isMatchmade": True, "minParty": 1, "maxParty": 3},
    }
    tables["DestinyActivityDefinition"]["701"] = {
        **display("Altar of Reflection"),
        "hash": 701,
        "destinationHash": MOON,
    }
    tables["DestinyPlaceDefinition"].update(
        {
            str(MOON_PLACE): {**display("The Moon"), "hash": MOON_PLACE},
            str(EARTH_PLACE): {**display("Earth"), "hash": EARTH_PLACE},
        }
    )
    moon_bubbles = [
        bubble(3326367698, "Archer's Line"),
        bubble(513390123, ""),  # Unnamed bubbles exist in the real Manifest.
        bubble(417490937, "Sorrow's Harbor"),
        bubble(3259346071, "The Pit"),
    ]
    tables["DestinyDestinationDefinition"].update(
        {
            str(MOON): destination(MOON, "The Moon", MOON_PLACE, moon_bubbles),
            str(MOON_DUPLICATE): destination(MOON_DUPLICATE, "The Moon", MOON_PLACE, moon_bubbles),
            str(MOON_WITHOUT_BUBBLES): destination(
                MOON_WITHOUT_BUBBLES, "The Moon", MOON_PLACE, []
            ),
            str(EDZ): destination(
                EDZ,
                "European Dead Zone",
                EARTH_PLACE,
                [bubble(3091570009, "Trostland"), bubble(698761453, "The Pit")],
            ),
            str(COSMODROME): destination(
                COSMODROME, "Cosmodrome", EARTH_PLACE, [bubble(1432145396, "Skywatch")]
            ),
        }
    )
    return tables


class ResolutionResolver(CountingIndexResolver):
    def __init__(self) -> None:
        super().__init__()
        self.tables = resolution_tables()


def provider(tmp_path: Any, resolver: FakeResolver | None = None) -> ManifestKnowledgeProvider:
    settings = Settings(manifest_cache_dir=tmp_path)
    return ManifestKnowledgeProvider(  # type: ignore[arg-type]
        resolver or ResolutionResolver(), settings
    )


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# --- Exact-match metadata ---------------------------------------------------------------


def test_exact_activity_name_reports_exact_resolution(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_activity_details("Haunted Altars of Sorrow"))

    assert result["activity"]["hash"] == 700
    assert result["resolution"] == {
        "query": "Haunted Altars of Sorrow",
        "matched_name": "Haunted Altars of Sorrow",
        "matched_hash": 700,
        "exact": True,
        "match_kind": "exact_name",
    }
    assert not any("not an exact" in value for value in result["limitations"])


def test_altars_of_sorrow_does_not_silently_become_haunted_altars(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_activity_details("Altars of Sorrow"))

    # Fuzzy behavior is preserved: the closest activity is still returned ...
    assert result["found"] is True
    assert result["activity"]["name"] == "Haunted Altars of Sorrow"
    # ... but it is explicitly marked as not being the requested entity.
    resolution = result["resolution"]
    assert resolution["exact"] is False
    assert resolution["match_kind"] == "closest_name"
    assert resolution["query"] == "Altars of Sorrow"
    assert resolution["matched_name"] == "Haunted Altars of Sorrow"
    assert "Altar of Reflection" in resolution["other_close_matches"]
    assert result["limitations"][0].startswith(
        "'Altars of Sorrow' is not an exact Manifest activity name."
    )
    assert "may be a different activity" in result["limitations"][0]
    # The existing activity payload and limitations are unchanged.
    assert result["activity"]["matchmaking"]["enabled"] is True
    assert any("possible static metadata" in value for value in result["limitations"])


def test_activity_exact_lookup_ignores_same_named_items(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_activity_details("Altars of Sorrow"))
    item = run(provider(tmp_path).get_item_details("Altars of Sorrow"))

    assert result["resolution"]["exact"] is False
    assert item["item"]["hash"] == 510
    assert item["resolution"]["exact"] is True


def test_hash_query_reports_hash_resolution(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_activity_details("700"))

    assert result["activity"]["name"] == "Haunted Altars of Sorrow"
    assert result["resolution"]["match_kind"] == "hash"
    assert result["resolution"]["exact"] is True


def test_exact_item_with_same_name_entries_prefers_gear_and_counts_them(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_item_details("wish ender"))

    assert result["item"]["hash"] == 1
    assert result["resolution"]["exact"] is True
    assert result["resolution"]["same_name_count"] == 2
    assert result["resolution"]["matched_name"] == "Wish-Ender"


def test_fuzzy_item_match_is_marked_not_exact(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_item_details("Wish Endr"))

    assert result["item"]["name"] == "Wish-Ender"
    assert result["resolution"]["exact"] is False
    assert "may be a different item" in result["limitations"][0]
    assert any("random rolls" in value for value in result["limitations"])


def test_exact_name_now_always_wins_over_a_preferred_fuzzy_gear_match(tmp_path: Any) -> None:
    # Before CP2 a fuzzy weapon (+110 preference) could outrank an exact-named emblem
    # (+100). An exact name is now always the resolved entity and is reported as exact.
    result = run(provider(tmp_path).get_item_details("Moonfang"))

    assert result["item"]["hash"] == 520
    assert result["resolution"]["exact"] is True


def test_find_item_source_carries_resolution(tmp_path: Any) -> None:
    exact = run(provider(tmp_path).find_item_source("Wish-Ender"))
    fuzzy = run(provider(tmp_path).find_item_source("Unknown Relik"))

    assert exact["resolution"]["exact"] is True
    assert fuzzy["item"]["name"] == "Unknown Relic"
    assert fuzzy["resolution"]["exact"] is False
    # The missing-source limitation keeps its leading position.
    assert "no reliable acquisition source" in fuzzy["limitations"][0]
    assert any("not an exact Manifest item name" in value for value in fuzzy["limitations"])


def test_not_found_shape_is_unchanged(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_activity_details("Zzzzqqq"))

    assert result == {
        "found": False,
        "query": "Zzzzqqq",
        "entity_type": "activity",
        "source": "bungie_manifest",
        "limitations": ["No matching visible English Manifest activity was found."],
    }


def test_exact_lookup_avoids_the_fuzzy_scan(tmp_path: Any) -> None:
    manifest = provider(tmp_path)
    run(manifest.warm_index())
    calls: list[tuple[Any, ...]] = []
    original = manifest._score_entries

    def recording(*args: Any) -> Any:
        calls.append(args)
        return original(*args)

    manifest._score_entries = recording  # type: ignore[method-assign]

    run(manifest.get_activity_details("Haunted Altars of Sorrow"))
    run(manifest.get_quest_details("Arc Scourge"))
    run(manifest.get_item_details("700"))
    assert calls == []

    run(manifest.get_activity_details("Altars of Sorrow"))
    assert len(calls) == 1


def test_exact_entries_uses_knowledge_engine_normalization(tmp_path: Any) -> None:
    manifest = provider(tmp_path)
    assert manifest.exact_entries("Wish-Ender", "DestinyInventoryItemDefinition") == []
    run(manifest.warm_index())

    for query in ("Wish-Ender", "WISH ENDER", "wish–ender"):
        assert normalize_text(query) == "wish ender"
        hashes = {
            value.entity_hash
            for value in manifest.exact_entries(query, "DestinyInventoryItemDefinition")
        }
        assert hashes == {1, 8}
    assert manifest.exact_entries("Wish", "DestinyInventoryItemDefinition") == []
    assert manifest.exact_entries("Wish-Ender", "DestinyActivityDefinition") == []


# --- Quest / bounty type label and rewards ----------------------------------------------


def test_arc_scourge_quest_details_expose_type_label_and_reward_names(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_quest_details("Arc Scourge"))

    quest = result["quest"]
    assert result["resolution"]["exact"] is True
    assert quest["item_type_name"] == "Duality Order"
    assert quest["reward_names"] == ["Legendary Payout", "Returners Gifts Unlocked"]
    assert quest["reward_total"] == 2
    assert quest["rewards_unresolved"] == 0
    assert any("do not describe quantities" in value for value in result["limitations"])


def test_quest_reward_names_are_capped_and_unresolved_rewards_counted(tmp_path: Any) -> None:
    quest = run(provider(tmp_path).get_quest_details("Generous Bounty"))["quest"]

    assert quest["reward_names"] == ["Reward 1", "Reward 2", "Reward 3", "Reward 4"]
    assert quest["reward_total"] == 6
    assert quest["rewards_unresolved"] == 1


def test_quest_without_declared_rewards_reports_none(tmp_path: Any) -> None:
    quest = run(provider(tmp_path).get_quest_details("Hunter's Remembrance"))["quest"]

    assert quest["item_type_name"] == "Quest"
    assert quest["reward_names"] == []
    assert quest["reward_total"] == 0
    assert quest["rewards_unresolved"] == 0


def test_fuzzy_quest_match_is_marked_not_exact(tmp_path: Any) -> None:
    result = run(provider(tmp_path).get_quest_details("Arc Scourg"))

    assert result["quest"]["name"] == "Arc Scourge"
    assert result["resolution"]["exact"] is False
    assert "may be a different quest" in result["limitations"][0]


# --- Destination sub-areas --------------------------------------------------------------


def test_sorrows_harbor_resolves_to_the_moon_from_destination_bubbles(tmp_path: Any) -> None:
    result = run(provider(tmp_path).resolve_sub_area("Sorrow's Harbor"))

    assert result["found"] is True
    assert result["ambiguous"] is False
    assert result["sub_area"] == "Sorrow's Harbor"
    assert result["sub_area_hashes"] == [417490937]
    # Two "The Moon" definitions declare the bubble; the bubble-less one does not.
    assert result["parents"] == [
        {
            "destination": "The Moon",
            "destination_hashes": [MOON, MOON_DUPLICATE],
            "place": "The Moon",
        }
    ]
    assert result["derivation"] == "DestinyDestinationDefinition.bubbles"
    assert result["limitations"] == []


@pytest.mark.parametrize("query", ["sorrows harbor", "SORROW’S HARBOR", " Sorrow's  Harbor "])
def test_sub_area_lookup_is_normalization_tolerant(tmp_path: Any, query: str) -> None:
    result = run(provider(tmp_path).resolve_sub_area(query))

    assert result["parents"][0]["destination"] == "The Moon"


@pytest.mark.parametrize(
    ("query", "destination_name", "place"),
    [
        ("Skywatch", "Cosmodrome", "Earth"),
        ("Trostland", "European Dead Zone", "Earth"),
        ("Archer's Line", "The Moon", "The Moon"),
    ],
)
def test_sub_area_lookup_is_generic_across_destinations(
    tmp_path: Any, query: str, destination_name: str, place: str
) -> None:
    result = run(provider(tmp_path).resolve_sub_area(query))

    assert result["found"] is True and result["ambiguous"] is False
    assert [value["destination"] for value in result["parents"]] == [destination_name]
    assert result["parents"][0]["place"] == place


def test_sub_area_declared_by_several_destinations_is_ambiguous(tmp_path: Any) -> None:
    result = run(provider(tmp_path).resolve_sub_area("The Pit"))

    assert result["found"] is True
    assert result["ambiguous"] is True
    assert {value["destination"] for value in result["parents"]} == {
        "The Moon",
        "European Dead Zone",
    }
    assert "does not identify one location" in result["limitations"][0]


@pytest.mark.parametrize("query", ["Sorrow", "Harbor", "", "Altars of Sorrow", "The Moon"])
def test_sub_area_lookup_never_partially_matches(tmp_path: Any, query: str) -> None:
    result = run(provider(tmp_path).resolve_sub_area(query))

    assert result["found"] is False
    assert result["parents"] == []
    assert result["sub_area"] is None


def test_unnamed_bubbles_are_not_indexed(tmp_path: Any) -> None:
    manifest = provider(tmp_path)
    run(manifest.warm_index())
    moon = manifest._hash_lookup[("DestinyDestinationDefinition", MOON)]

    assert 513390123 not in {value["hash"] for value in moon.sub_areas}
    assert moon.place_hash == MOON_PLACE
    assert "" not in manifest._sub_areas


# --- Index schema 5, persistence and warmup ---------------------------------------------


def _index_files(tmp_path: Any) -> list[Any]:
    return sorted(tmp_path.glob("destiny-knowledge-index-*.json"))


def test_index_is_written_with_schema_5_and_sub_areas(tmp_path: Any) -> None:
    run(provider(tmp_path).warm_index())

    (path,) = _index_files(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert KNOWLEDGE_INDEX_SCHEMA_VERSION == 5
    assert payload["schema_version"] == 5
    moon = next(
        value
        for value in payload["entries"]
        if value["definition_type"] == "DestinyDestinationDefinition"
        and value["entity_hash"] == MOON
    )
    assert {value["name"] for value in moon["sub_areas"]} >= {"Sorrow's Harbor"}


def test_schema_4_index_is_ignored_and_rebuilt(tmp_path: Any) -> None:
    run(provider(tmp_path).warm_index())
    (path,) = _index_files(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 4
    for value in payload["entries"]:
        value.pop("sub_areas"), value.pop("place_hash")
    path.write_text(json.dumps(payload), encoding="utf-8")

    resolver = ResolutionResolver()
    state = run(provider(tmp_path, resolver).warm_index())

    assert state["state"] == "ready"
    assert "DestinyDestinationDefinition" in resolver.scans
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 5


def test_reloaded_index_keeps_sub_areas_and_exact_names_without_rescanning(
    tmp_path: Any,
) -> None:
    run(provider(tmp_path).warm_index())
    resolver = ResolutionResolver()
    reloaded = provider(tmp_path, resolver)

    state = run(reloaded.warm_index())
    harbor = run(reloaded.resolve_sub_area("Sorrow's Harbor"))
    activity = run(reloaded.get_activity_details("Haunted Altars of Sorrow"))

    assert state["state"] == "ready"
    assert resolver.scans == []
    assert harbor["parents"][0]["destination"] == "The Moon"
    assert activity["resolution"]["match_kind"] == "exact_name"


def test_sub_area_lookup_before_warmup_builds_the_index_lazily(tmp_path: Any) -> None:
    resolver = ResolutionResolver()
    manifest = provider(tmp_path, resolver)

    async def concurrent() -> tuple[dict[str, Any], dict[str, Any]]:
        warmup = asyncio.create_task(manifest.warm_index())
        await asyncio.sleep(0)
        result = await manifest.resolve_sub_area("Skywatch")
        return await warmup, result

    state, result = run(concurrent())

    assert state["state"] == "ready"
    assert result["parents"][0]["destination"] == "Cosmodrome"
    assert resolver.scans.count("DestinyDestinationDefinition") == 1


def test_index_preparation_builds_exact_and_sub_area_maps_off_the_event_loop(
    tmp_path: Any,
) -> None:
    import threading

    manifest = provider(tmp_path)
    threads: list[int] = []
    original = manifest._prepare_index

    def recording(entries: Any) -> Any:
        threads.append(threading.get_ident())
        return original(entries)

    manifest._prepare_index = recording  # type: ignore[method-assign]

    async def warm() -> int:
        await manifest.warm_index()
        return threading.get_ident()

    loop_thread = run(warm())

    assert threads and all(value != loop_thread for value in threads)
    assert manifest._exact_names and manifest._sub_areas


# --- Search semantics are unchanged -----------------------------------------------------


def test_search_ranking_is_unchanged_and_flags_exact_names(tmp_path: Any) -> None:
    manifest = provider(tmp_path)
    result = run(manifest.search_entities("Altars of Sorrow", None, 10))
    direct = manifest._score_entries("Altars of Sorrow", None, 10)

    assert [value["hash"] for value in result["results"]] == [
        entry.entity_hash for entry, _ in direct
    ]
    flags = {value["name"]: value["exact_name_match"] for value in result["results"]}
    assert flags["Altars of Sorrow"] is True
    assert flags["Haunted Altars of Sorrow"] is False
