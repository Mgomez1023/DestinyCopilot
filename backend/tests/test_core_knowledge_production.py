"""Production core knowledge corpus: provenance, ownership, aliases, anchors, and the
deterministic engine results future golden cases depend on."""

import copy
import json
import os
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.core_knowledge import (
    MAX_BUNDLE_CHARS,
    CoreKnowledgeError,
    DestinyKnowledgeEngine,
    corpus_audit,
    load_core_knowledge,
    normalize_text,
    overdue_reviews,
    parse_core_knowledge,
)
from app.core_knowledge_anchors import (
    SnapshotDefinitions,
    SqliteManifestDefinitions,
    build_snapshot,
    structural_claim_issues,
    validate_anchors,
)

DATA = Path(__file__).parents[1] / "app" / "data" / "core_knowledge.json"
SNAPSHOT = Path(__file__).parent / "fixtures" / "core_knowledge_anchor_snapshot.json"

# Every single-token key (canonical name, alias, plural, shorthand, ambiguity alias) can
# fire on ordinary words, so each one is reviewed here. Adding a key fails this test until
# it is consciously added with a reason.
REVIEWED_SINGLE_TOKEN_KEYS = {
    # Ambiguity, never a single entity.
    "altar": "ambiguity; 'altar' alone names several activities and places",
    "altars": "ambiguity; never resolves to one activity",
    # Low collision risk in Destiny chat.
    "bounties": "Destiny term",
    "bounty": "Destiny term",
    "crucible": "Destiny mode name; ordinary English use is rare in chat",
    "dungeon": "Destiny activity type",
    "dungeons": "Destiny activity type",
    "fireteam": "Destiny term",
    "fireteams": "Destiny term",
    "gambit": "mode name; English 'gambit' is rare in Destiny chat",
    "glaive": "weapon archetype",
    "glaives": "weapon archetype",
    "handcannon": "registered misspelling of Hand Cannon",
    "handcannons": "plural of the registered misspelling",
    "matchmaking": "Destiny/gaming term",
    "nightfall": "Destiny activity type",
    "nightfalls": "Destiny activity type",
    "sidearm": "weapon archetype",
    "sidearms": "weapon archetype",
    "triumph": "Destiny term; English use is uncommon",
    "triumphs": "Destiny term",
    # Higher frequency; attach often, accepted because the cards are small.
    "bow": "weapon archetype; English verb 'bow' is rare in Destiny chat",
    "bows": "weapon archetype",
    "exotic": "HIGH FREQUENCY: attaches the Exotic equip-limits rule card (~460 chars)",
    "exotics": "HIGH FREQUENCY: attaches the Exotic equip-limits rule card",
    "hunter": "class name; also ordinary English",
    "hunters": "class name; also ordinary English",
    "patrol": "activity type; also ordinary English",
    "patrols": "activity type; also ordinary English",
    "quest": "HIGH FREQUENCY: Destiny term with a small card",
    "quests": "HIGH FREQUENCY: Destiny term with a small card",
    "raid": "HIGH FREQUENCY activity type",
    "raids": "HIGH FREQUENCY activity type",
    "shotgun": "weapon archetype",
    "shotguns": "weapon archetype",
    "sniper": "common community name for Sniper Rifle",
    "snipers": "plural",
    "strike": "activity type; also ordinary English",
    "strikes": "activity type; also ordinary English",
    "sword": "weapon archetype",
    "swords": "weapon archetype",
    "titan": "class name",
    "titans": "class name",
    "trials": "community name for Trials of Osiris; also ordinary English",
    "vendor": "Destiny term; also ordinary English",
    "vendors": "Destiny term; also ordinary English",
    "warlock": "class name",
    "warlocks": "class name",
    # Registered shorthand.
    "gl": "shorthand: Grenade Launcher",
    "hc": "shorthand: Hand Cannon",
    "lfr": "shorthand: Linear Fusion Rifle",
    "smg": "shorthand: Submachine Gun",
}

EXPECTED_COMPOSITION = {
    "activity": 4,
    "activity_type": 8,
    "armor_concept": 1,
    "class": 3,
    "concept": 7,
    "game_mode": 4,
    "progression": 3,
    "rule": 1,
    "weapon_archetype": 17,
}

MANIFEST_OWNED = {
    "class_restriction",
    "item_slot",
    "item_tier",
    "damage_type",
    "item_type_label",
    "quest_reward_names",
    "ammo_type",
    "matchmaking",
}


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus() -> Any:
    return load_core_knowledge(DATA)


@pytest.fixture(scope="module")
def engine(corpus: Any) -> DestinyKnowledgeEngine:
    return DestinyKnowledgeEngine(corpus)


# --- Loading, composition and provenance ------------------------------------------------


def test_production_corpus_loads_and_composition_is_reviewed(corpus: Any) -> None:
    audit = corpus_audit(corpus)

    # Changing the corpus size or mix must be a deliberate edit of this expectation.
    assert audit["by_kind"] == EXPECTED_COMPOSITION
    assert audit["entities"] == 48
    assert corpus.manifest_version


def test_every_fact_has_non_curated_provenance(corpus: Any) -> None:
    for entity in corpus.entities:
        for fact in entity.facts:
            kinds = {corpus.sources[value].kind for value in fact.src}
            assert kinds - {"curated"}, f"{entity.id}.{fact.p} rests on curation alone"


def test_every_summary_cites_non_curated_sources(corpus: Any) -> None:
    for entity in corpus.entities:
        assert entity.summary_src, f"{entity.id} summary has no source"
        kinds = {corpus.sources[value].kind for value in entity.summary_src}
        assert kinds - {"curated"}, f"{entity.id} summary rests on curation alone"


def test_no_model_memory_sources_and_no_orphan_sources(corpus: Any) -> None:
    cited = {value for entity in corpus.entities for fact in entity.facts for value in fact.src}
    cited |= {value for entity in corpus.entities for value in entity.summary_src}

    assert not [key for key, value in corpus.sources.items() if value.kind == "curated"]
    assert set(corpus.sources) == cited
    for source_id, source in corpus.sources.items():
        assert source.verified_at is not None, source_id
        if source.kind == "official_bungie":
            assert source.ref.startswith("https://bungie-net.github.io/"), source_id


def test_no_production_text_is_fixture_or_instruction_text(payload: dict[str, Any]) -> None:
    text = json.dumps(payload).casefold()

    assert "fixture" not in text
    for phrase in ("you must", "always answer", "never answer", "ignore previous"):
        assert phrase not in text


def test_freshness_and_review_metadata_are_valid(corpus: Any) -> None:
    today = date.today()
    for entity in corpus.entities:
        assert entity.last_verified <= today, entity.id
        if entity.freshness != "stable" or any(
            corpus.predicates[fact.p].freshness != "stable" for fact in entity.facts
        ):
            assert entity.review_after is not None, entity.id
    # An overdue review fails CI (forcing re-verification) but never the runtime loader.
    assert overdue_reviews(corpus, today) == []


def test_overdue_review_is_reported_but_never_blocks_loading(payload: dict[str, Any]) -> None:
    corpus = parse_core_knowledge(payload)

    assert overdue_reviews(corpus, date(2099, 1, 1))
    assert "weapon_archetype.hand_cannon" in overdue_reviews(corpus, date(2099, 1, 1))


def test_non_stable_knowledge_without_review_after_is_rejected(payload: dict[str, Any]) -> None:
    broken = copy.deepcopy(payload)
    hand_cannon = next(
        value for value in broken["entities"] if value["id"] == "weapon_archetype.hand_cannon"
    )
    del hand_cannon["review_after"]

    with pytest.raises(CoreKnowledgeError, match="non-stable knowledge needs review_after"):
        parse_core_knowledge(broken)


# --- Predicate ownership ----------------------------------------------------------------


def test_manifest_owned_predicates_are_registered_and_never_curated(corpus: Any) -> None:
    for name in MANIFEST_OWNED:
        predicate = corpus.predicates[name]
        assert predicate.owner == "manifest" and predicate.curatable is False, name
    used = {fact.p for entity in corpus.entities for fact in entity.facts}
    assert not used & MANIFEST_OWNED


@pytest.mark.parametrize(
    ("predicate", "value"),
    [
        ("class_restriction", "Hunter"),
        ("damage_type", "Solar"),
        ("item_type_label", "Duality Order"),
        ("quest_reward_names", "Legendary Payout"),
        ("matchmaking", "matchmade"),
    ],
)
def test_manual_curation_of_manifest_owned_facts_is_rejected(
    payload: dict[str, Any], predicate: str, value: str
) -> None:
    broken = copy.deepcopy(payload)
    broken["entities"][0]["facts"].append(
        {"p": predicate, "o": value, "src": ["manifest.class_definitions"]}
    )

    with pytest.raises(CoreKnowledgeError, match="owned by the Manifest"):
        parse_core_knowledge(broken)


# --- Aliases, shorthand and ambiguity ---------------------------------------------------


def test_all_aliases_are_collision_free(payload: dict[str, Any]) -> None:
    # The loader rejects collisions; re-parsing the committed file proves there are none.
    assert parse_core_knowledge(payload).entities


def test_single_token_keys_are_explicitly_reviewed(corpus: Any) -> None:
    keys = set(corpus_audit(corpus)["single_token_keys"])

    assert keys == set(REVIEWED_SINGLE_TOKEN_KEYS)


def test_shorthand_registry_is_small_and_deliberate(corpus: Any) -> None:
    assert {key: value.entity for key, value in corpus.shorthands.items()} == {
        "hc": "weapon_archetype.hand_cannon",
        "smg": "weapon_archetype.submachine_gun",
        "lfr": "weapon_archetype.linear_fusion_rifle",
        "gl": "weapon_archetype.grenade_launcher",
    }


def test_unregistered_short_alias_is_rejected(payload: dict[str, Any]) -> None:
    broken = copy.deepcopy(payload)
    auto = next(value for value in broken["entities"] if value["id"].endswith("auto_rifle"))
    auto["aliases"] = ["ar"]

    with pytest.raises(CoreKnowledgeError, match="must be registered as explicit shorthand"):
        parse_core_knowledge(broken)


def test_ambiguity_candidates_exist(corpus: Any) -> None:
    ids = {entity.id for entity in corpus.entities}
    for group in corpus.ambiguity_groups:
        assert set(group.candidates) <= ids


# --- Manifest anchors -------------------------------------------------------------------


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_all_manifest_anchors_validate_against_the_committed_snapshot(
    corpus: Any, snapshot: dict[str, Any]
) -> None:
    assert snapshot["manifest_version"] == corpus.manifest_version
    assert validate_anchors(corpus, SnapshotDefinitions(snapshot)) == []


def test_snapshot_is_exactly_what_the_corpus_needs(corpus: Any, snapshot: dict[str, Any]) -> None:
    # Rebuilding from the snapshot itself must be lossless; a stale snapshot would differ.
    assert build_snapshot(corpus, SnapshotDefinitions(snapshot)) == snapshot


def test_every_entity_is_anchored_or_cites_verified_manifest_text(corpus: Any) -> None:
    for entity in corpus.entities:
        text_sources = [
            value for value in entity.summary_src if corpus.sources[value].kind == "manifest_text"
        ]
        assert entity.manifest_anchors or text_sources, entity.id
        assert all(anchor.hash for anchor in entity.manifest_anchors), entity.id


def _drifted(snapshot: dict[str, Any], definition_type: str, entity_hash: int) -> Any:
    changed = copy.deepcopy(snapshot)
    return changed, changed["definitions"][definition_type][str(entity_hash)]


@pytest.mark.parametrize(
    ("definition_type", "entity_hash", "mutate", "message"),
    [
        (
            "DestinyActivityModeDefinition",
            1164760504,
            lambda value: value.update(activityModeCategory=1),
            "activityModeCategory is 1, expected 2",
        ),
        (
            "DestinyDestinationDefinition",
            290444260,
            lambda value: value.update(bubbles=[]),
            "no longer declares sub-area 'Sorrow's Harbor'",
        ),
        (
            "DestinyItemCategoryDefinition",
            6,
            lambda value: value["displayProperties"].update(name="Hand Gun"),
            "is named 'Hand Gun', expected 'Hand Cannon'",
        ),
        (
            "DestinyInventoryItemDefinition",
            3960926756,
            lambda value: value["equippingBlock"].update(uniqueLabel=None),
            "equippingBlock.uniqueLabel is None",
        ),
        (
            "DestinyRecordDefinition",
            448445933,
            lambda value: value["displayProperties"].update(description="Exotics are fun."),
            "quoted text is no longer present",
        ),
        (
            "DestinyInventoryItemDefinition",
            2531049971,
            lambda value: value.update(redacted=True),
            "is redacted",
        ),
    ],
)
def test_anchor_drift_is_reported(
    corpus: Any,
    snapshot: dict[str, Any],
    definition_type: str,
    entity_hash: int,
    mutate: Any,
    message: str,
) -> None:
    changed, definition = _drifted(snapshot, definition_type, entity_hash)
    mutate(definition)

    issues = validate_anchors(corpus, SnapshotDefinitions(changed))

    assert issues and all(issue.severity == "error" for issue in issues)
    assert any(message in issue.message for issue in issues), [str(v) for v in issues]


def test_missing_definition_and_broken_ancestry_are_reported(
    corpus: Any, snapshot: dict[str, Any]
) -> None:
    changed = copy.deepcopy(snapshot)
    del changed["definitions"]["DestinyActivityDefinition"]["4140407301"]
    altars = changed["definitions"]["DestinyPresentationNodeDefinition"]["1325869705"]
    altars["parentNodeHashes"] = []

    messages = [issue.message for issue in validate_anchors(corpus, SnapshotDefinitions(changed))]

    assert any("4140407301 no longer exists" in value for value in messages)
    assert any("lack 'The Moon'" in value for value in messages)


def _item(**values: Any) -> dict[str, Any]:
    return {"equippable": True, "itemType": 3, "inventory": {"tierTypeName": "Legendary"}, **values}


def test_structural_recount_flags_contradicting_items(corpus: Any) -> None:
    items = [
        _item(hash=1, inventory={"tierTypeName": "Exotic"}, equippingBlock={}),
        _item(
            hash=2,
            itemType=2,
            classType=3,
            inventory={"tierTypeName": "Exotic"},
            equippingBlock={"uniqueLabel": "exotic_armor"},
        ),
        _item(hash=3, equippingBlock={"uniqueLabel": "exotic_weapon"}),
        *[
            _item(
                hash=10 + n,
                collectibleHash=5,
                itemCategoryHashes=[6],
                equippingBlock={"ammoType": 2},
            )
            for n in range(3)
        ],
    ]

    messages = " | ".join(str(value) for value in structural_claim_issues(corpus, items))

    assert "item 1: uniqueLabel None contradicts" in messages
    assert "item 2: Exotic armor is not class-restricted" in messages
    assert "item 3: uniqueLabel 'exotic_weapon' contradicts" in messages
    assert "weapon_archetype.hand_cannon" in messages and "covers 0.0%" in messages


LOCAL_MANIFEST = os.environ.get("CORE_KNOWLEDGE_MANIFEST")


@pytest.mark.skipif(not LOCAL_MANIFEST, reason="set CORE_KNOWLEDGE_MANIFEST to a local SQLite")
def test_anchors_and_structure_validate_against_a_local_manifest(corpus: Any) -> None:
    definitions = SqliteManifestDefinitions(Path(str(LOCAL_MANIFEST)))

    assert validate_anchors(corpus, definitions) == []
    items = definitions.iter_table("DestinyInventoryItemDefinition")
    assert structural_claim_issues(corpus, items) == []


# --- Deterministic results for future golden cases --------------------------------------


def _card(engine: DestinyKnowledgeEngine, message: str, name: str) -> dict[str, Any]:
    bundle = engine.bundle(message)
    assert bundle is not None, message
    cards = bundle.payload.get("entities", []) + bundle.payload.get("rules", [])
    return next(value for value in cards if value["name"] == name)


def _facts(card: dict[str, Any]) -> dict[str, Any]:
    return {predicate: value for predicate, value, _ in card.get("facts", [])}


@pytest.mark.parametrize(
    ("message", "name", "expected"),
    [
        ("What is Crucible?", "Crucible", {"game_mode": "pvp"}),
        ("Is Crucible PvP or PvE?", "Crucible", {"game_mode": "pvp"}),
        (
            "What is a Nightfall?",
            "Nightfall",
            {"kind_of": "Strike", "game_mode": "pve", "fireteam_size": 3},
        ),
        ("What is a dungeon?", "Dungeon", {"game_mode": "pve", "fireteam_size": 3}),
        ("What is a raid?", "Raid", {"game_mode": "pve", "fireteam_size": 6}),
        ("What is a Lost Sector?", "Lost Sector", {"game_mode": "pve", "fireteam_size": 3}),
        ("What is a Hand Cannon?", "Hand Cannon", {"usual_ammo_type": "primary"}),
        (
            "Can I equip two Exotic armor pieces?",
            "Exotic armor",
            {"max_equipped_exotic_armor": 1},
        ),
        (
            "Can a Titan equip Hunter Exotic armor?",
            "Exotic armor",
            {"each_piece_restricted_to_one_class": True},
        ),
        ("can I run two exotics", "Exotic equip limits", {"max_equipped_exotic_weapons": 1}),
    ],
)
def test_golden_knowledge_resolves(
    engine: DestinyKnowledgeEngine, message: str, name: str, expected: dict[str, Any]
) -> None:
    facts = _facts(_card(engine, message, name))

    assert {key: facts.get(key) for key in expected} == expected


def test_titan_hunter_exotic_armor_question_carries_both_classes(
    engine: DestinyKnowledgeEngine,
) -> None:
    bundle = engine.bundle("Can a Titan equip Hunter Exotic armor?")

    assert bundle is not None
    assert [value["name"] for value in bundle.payload["entities"]] == [
        "Titan",
        "Hunter",
        "Exotic armor",
    ]


def test_exotic_weapon_and_armor_together(engine: DestinyKnowledgeEngine) -> None:
    message = "Can I equip an Exotic weapon and Exotic armor together?"

    weapon = _facts(_card(engine, message, "Exotic weapon"))
    armor = _facts(_card(engine, message, "Exotic armor"))

    assert weapon["exotic_weapon_and_armor_together"] is True
    assert armor["exotic_weapon_and_armor_together"] is True
    assert weapon["max_equipped_exotic_weapons"] == 1


def test_altars_is_explicitly_ambiguous_and_never_chooses(engine: DestinyKnowledgeEngine) -> None:
    for text in ('What does "Altars" refer to?', "is the altar worth it", "ALTARS"):
        bundle = engine.bundle(text)
        assert bundle is not None
        assert "entities" not in bundle.payload
        assert [value["name"] for value in bundle.payload["ambiguous"][0]["candidates"]] == [
            "Altars of Sorrow",
            "Haunted Altars of Sorrow",
            "Altars of Summoning",
            "Altar of Reflection",
        ]
    assert engine.resolve("altars").status == "ambiguous"
    assert engine.resolve("altars").entity_id is None


def test_where_is_altars_of_sorrow(engine: DestinyKnowledgeEngine) -> None:
    card = _card(engine, "Where is Altars of Sorrow?", "Altars of Sorrow")

    assert _facts(card) == {"located_on": "The Moon", "located_in": "Sorrow's Harbor"}
    assert card["not_same_as"] == ["Haunted Altars of Sorrow"]


def test_altars_of_sorrow_matchmaking_is_unknown_not_guessed(
    engine: DestinyKnowledgeEngine,
) -> None:
    card = _card(engine, "Is Altars of Sorrow matchmade?", "Altars of Sorrow")

    assert "matchmaking" in card["unknown"]
    assert "matchmaking" not in _facts(card)
    assert card["not_same_as"] == ["Haunted Altars of Sorrow"]


def test_haunted_altars_is_a_separate_seasonal_entity(engine: DestinyKnowledgeEngine) -> None:
    resolution = engine.resolve("Haunted Altars of Sorrow")
    card = engine.lookup("Haunted Altars of Sorrow")["entity"]

    assert resolution.entity_id == "activity.haunted_altars_of_sorrow"
    assert card["status"] == "seasonal"
    assert card["not_same_as"] == ["Altars of Sorrow"]


def test_hc_resolves_and_unregistered_shorthand_stays_unknown(
    engine: DestinyKnowledgeEngine,
) -> None:
    hc = engine.resolve("HC")

    assert (hc.status, hc.entity_id, hc.via) == (
        "entity",
        "weapon_archetype.hand_cannon",
        "shorthand",
    )
    for token in ("ar", "sr", "mg", "rl", "lmg", "shotty", "hcs", "pvp", "pve"):
        assert engine.resolve(token).status == "unknown", token
    assert engine.bundle("best ar for pvp?") is None


def test_registered_shorthand_resolves(engine: DestinyKnowledgeEngine) -> None:
    assert engine.resolve("smg").entity_id == "weapon_archetype.submachine_gun"
    assert engine.resolve("LFR").entity_id == "weapon_archetype.linear_fusion_rifle"
    assert engine.resolve("gl").entity_id == "weapon_archetype.grenade_launcher"


def test_mixed_ammo_archetypes_mark_ammo_unknown(engine: DestinyKnowledgeEngine) -> None:
    for name in (
        "Grenade Launcher",
        "Fusion Rifle",
        "Linear Fusion Rifle",
        "Sidearm",
        "Bow",
        "Trace Rifle",
    ):
        card = engine.lookup(name)["entity"]
        assert "usual_ammo_type" not in _facts(card), name
        assert "usual_ammo_type" in card["unknown"], name


def test_unknown_terms_stay_unknown(engine: DestinyKnowledgeEngine) -> None:
    for text in ("Shatterdive", "Prismatic", "Void", "Arc", "Solar", "Pale Heart"):
        assert engine.resolve(text).status == "unknown", text


# --- Bundle determinism and timing ------------------------------------------------------


def test_production_bundles_are_deterministic_and_bounded(corpus: Any) -> None:
    message = (
        "Titan or Hunter for the raid, dungeon and Nightfall this week? I run an HC, a sniper, "
        "a sword and two exotics, Exotic armor too, before Trials of Osiris and Altars of "
        "Sorrow. Also Gambit, Iron Banner, Lost Sectors and the Crucible."
    )

    first = DestinyKnowledgeEngine(corpus).bundle(message, ("Strike", "Patrol"))
    second = DestinyKnowledgeEngine(corpus).bundle(message, ("Strike", "Patrol"))

    assert first is not None and second is not None
    assert first.text == second.text
    assert first.char_count <= MAX_BUNDLE_CHARS
    assert first.truncated is True
    assert first.payload["omitted"]["entities"] > 0
    assert [value["name"] for value in first.payload["entities"]][:2] == ["Titan", "Hunter"]


def test_production_bundle_timing_stays_inside_the_existing_ceiling(
    engine: DestinyKnowledgeEngine,
) -> None:
    names = [entity.name for entity in engine._entities.values()]  # noqa: SLF001
    message = " ".join(["Can I run", *names, "with HC and exotics?"] * 3)[:2000]
    candidates = tuple(names[:12])
    durations = []
    for _ in range(25):
        started = time.perf_counter()
        engine.bundle(message, candidates)
        durations.append((time.perf_counter() - started) * 1000)

    assert statistics.median(durations) < 15


def test_normalization_matches_manifest_names(engine: DestinyKnowledgeEngine) -> None:
    assert normalize_text("Savathûn's Throne World") == "savathuns throne world"
    card = engine.lookup("altar of reflection")["entity"]
    assert _facts(card)["located_on"] == "Savathûn's Throne World"
