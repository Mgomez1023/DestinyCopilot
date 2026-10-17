import builtins
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from app.core_knowledge import (
    MAX_BUNDLE_AMBIGUITIES,
    MAX_BUNDLE_CHARS,
    MAX_BUNDLE_ENTITIES,
    MAX_BUNDLE_FACTS_PER_ENTITY,
    MAX_BUNDLE_RULES,
    CoreKnowledgeError,
    DestinyKnowledgeEngine,
    load_core_knowledge,
    normalize_text,
    parse_core_knowledge,
)

FIXTURE = Path(__file__).parent / "fixtures" / "core_knowledge_fixture.json"


def fixture_payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def engine() -> DestinyKnowledgeEngine:
    return DestinyKnowledgeEngine.from_file(FIXTURE)


def entity(entity_id: str, name: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": entity_id,
        "name": name,
        "kind": "concept",
        "summary": f"Synthetic {name}.",
        "last_verified": "2026-10-17",
        **extra,
    }


def with_entities(*entities: dict[str, Any], **top: Any) -> dict[str, Any]:
    payload = fixture_payload()
    payload["entities"].extend(entities)
    payload.update(top)
    return payload


def rejected(payload: dict[str, Any]) -> list[str]:
    with pytest.raises(CoreKnowledgeError) as error:
        parse_core_knowledge(payload)
    return error.value.errors


def find(payload: dict[str, Any], entity_id: str) -> dict[str, Any]:
    return next(value for value in payload["entities"] if value["id"] == entity_id)


# --- Loader and validation ---------------------------------------------------------------


def test_fixture_corpus_loads_and_validates() -> None:
    corpus = load_core_knowledge(FIXTURE)

    assert corpus.schema_version == 1
    assert corpus.corpus_version == "fixture-2026-10-17"
    assert "class_restriction" in corpus.predicates


def test_unknown_predicate_is_rejected() -> None:
    payload = with_entities(
        entity("concept.bad", "Bad", facts=[{"p": "made_up", "o": "x", "src": "fixture.curated"}])
    )

    assert any("unknown predicate 'made_up'" in value for value in rejected(payload))


def test_curated_fact_on_manifest_owned_predicate_is_rejected() -> None:
    payload = with_entities(
        entity(
            "concept.synthoceps",
            "Synthoceps",
            facts=[{"p": "class_restriction", "o": "Titan", "src": "fixture.curated"}],
        )
    )

    assert any("owned by the Manifest" in value for value in rejected(payload))


def test_manifest_preferred_fact_requires_an_absence_reason() -> None:
    payload = fixture_payload()
    del find(payload, "activity.altars_of_sorrow")["facts"][0]["manifest_absent_reason"]

    assert any("needs a manifest_absent_reason" in value for value in rejected(payload))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: find(payload, "activity.altars_of_sorrow")["related"].append(
                {"rel": "related_to", "id": "activity.missing"}
            ),
            "related entity 'activity.missing' does not exist",
        ),
        (
            lambda payload: find(payload, "mode.crucible")["facts"][0].update(
                {"src": ["fixture.missing"]}
            ),
            "cites unknown source 'fixture.missing'",
        ),
        (
            lambda payload: payload["shorthands"].update({"gl": {"entity": "weapon_archetype.gl"}}),
            "targets unknown entity 'weapon_archetype.gl'",
        ),
        (
            lambda payload: payload["ambiguity_groups"][0]["candidates"].append("activity.ghost"),
            "candidate 'activity.ghost' does not exist",
        ),
    ],
)
def test_dangling_references_are_rejected(mutate: Any, message: str) -> None:
    payload = fixture_payload()
    mutate(payload)

    assert any(message in value for value in rejected(payload))


def test_entity_reference_values_must_point_to_known_entities() -> None:
    payload = fixture_payload()
    payload["predicates"]["counter_to"] = {
        "value": "entity",
        "owner": "curated",
        "description": "Synthetic entity-valued predicate.",
    }
    find(payload, "mode.crucible")["facts"].append(
        {"p": "counter_to", "o": "Gambit", "o_ref": "mode.gambit", "src": "fixture.curated"}
    )

    assert any("references unknown entity 'mode.gambit'" in value for value in rejected(payload))


def test_duplicate_entity_ids_are_rejected() -> None:
    payload = with_entities(entity("mode.crucible", "Crucible Copy"))

    assert any("duplicate entity id 'mode.crucible'" in value for value in rejected(payload))


@pytest.mark.parametrize(
    ("extra", "collision"),
    [
        # Two entities claiming the same alias.
        ([entity("concept.copy", "Copy", aliases=["altars of sorrow"])], "altars of sorrow"),
        # A canonical name that normalizes to another entity's name.
        ([entity("concept.punct", "ALTARS-of-Sorrow!")], "altars of sorrow"),
        # A generated plural that collides with another entity's name.
        ([entity("concept.cannons", "Hand Cannons")], "hand cannons"),
        # An entity alias colliding with an ambiguity group alias.
        ([entity("concept.altar", "Altar Thing", aliases=["altars"])], "altars"),
    ],
)
def test_alias_collisions_are_rejected(extra: list[dict[str, Any]], collision: str) -> None:
    payload = with_entities(*extra)

    assert any(f"alias collision: '{collision}'" in value for value in rejected(payload))


def test_shorthand_cannot_collide_with_an_entity_name() -> None:
    payload = with_entities(entity("concept.smg", "SMG"))
    payload["shorthands"]["smg"] = {"entity": "weapon_archetype.hand_cannon"}

    assert any("alias collision: 'smg'" in value for value in rejected(payload))


def test_short_aliases_must_be_registered_as_explicit_shorthand() -> None:
    payload = fixture_payload()
    find(payload, "weapon_archetype.hand_cannon")["aliases"] = ["hc"]

    assert any(
        "short alias 'hc' must be registered as explicit shorthand" in value
        for value in rejected(payload)
    )


@pytest.mark.parametrize("token", ["h c", "HC", "x", "toolongshorthand"])
def test_shorthand_tokens_must_be_single_lowercase_tokens(token: str) -> None:
    payload = fixture_payload()
    payload["shorthands"] = {token: {"entity": "weapon_archetype.hand_cannon"}}

    assert any("must be one lowercase alphanumeric token" in value for value in rejected(payload))


@pytest.mark.parametrize(
    ("predicate", "value", "message"),
    [
        ("fireteam_size_max", "3", "requires an integer value"),
        ("fireteam_size_max", True, "requires an integer value"),
        ("exotic_weapon_and_armor_together", 1, "requires a boolean value"),
        ("located_on", 7, "requires a non-empty text value"),
        ("game_mode", "co-op", "'co-op' is not an allowed value for predicate 'game_mode'"),
    ],
)
def test_fact_values_must_match_the_predicate_type(
    predicate: str, value: Any, message: str
) -> None:
    payload = with_entities(
        entity(
            "concept.typed",
            "Typed",
            facts=[
                {
                    "p": predicate,
                    "o": value,
                    "src": "fixture.curated",
                    "manifest_absent_reason": "Synthetic.",
                }
            ],
        )
    )

    assert any(message in error for error in rejected(payload))


def test_class_values_are_restricted_for_curated_class_predicates() -> None:
    payload = fixture_payload()
    payload["predicates"]["intended_class"] = {
        "value": "class",
        "owner": "curated",
        "description": "Synthetic curated class predicate.",
    }
    find(payload, "mode.crucible")["facts"].append(
        {"p": "intended_class", "o": "Paladin", "src": "fixture.curated"}
    )

    assert any("'Paladin' is not a valid class" in value for value in rejected(payload))


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (["activity.altars_of_sorrow", "activity.altars_of_sorrow"], "candidates must be unique"),
        (["activity.altars_of_sorrow"], "at least 2"),
    ],
)
def test_invalid_ambiguity_candidates_are_rejected(candidates: list[str], message: str) -> None:
    payload = fixture_payload()
    payload["ambiguity_groups"][0]["candidates"] = candidates

    assert any(message in value for value in rejected(payload))


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"value": "enum", "owner": "curated", "description": "x"}, "must declare their allowed"),
        (
            {"value": "int", "owner": "curated", "values": ["1"], "description": "x"},
            "Only enum predicates",
        ),
        ({"value": "class", "owner": "manifest", "description": "x"}, "cannot be curatable"),
    ],
)
def test_inconsistent_predicate_definitions_are_rejected(
    definition: dict[str, Any], message: str
) -> None:
    payload = fixture_payload()
    payload["predicates"]["synthetic"] = definition

    assert any(message in value for value in rejected(payload))


def test_validation_reports_every_problem_at_once() -> None:
    payload = with_entities(
        entity("mode.crucible", "Second Crucible"),
        entity("concept.bad", "Bad", facts=[{"p": "made_up", "o": "x", "src": "nope"}]),
    )

    errors = rejected(payload)

    assert any("duplicate entity id" in value for value in errors)
    assert any("unknown predicate" in value for value in errors)


def test_unreadable_corpus_is_a_clear_error(tmp_path: Path) -> None:
    broken = tmp_path / "corpus.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(CoreKnowledgeError, match="cannot read corpus"):
        load_core_knowledge(broken)


# --- Normalization and resolution --------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("Sorrow’s Harbor", "sorrows harbor"),
        ("Hunter's Remembrance", "hunters remembrance"),
        ("ALTARS—of   SORROW!!", "altars of sorrow"),
        ("Xûr", "xur"),
        ("Wish-Ender", "wish ender"),
    ],
)
def test_normalization_is_nfkd_casefolded_and_punctuation_insensitive(
    raw: str, normalized: str
) -> None:
    assert normalize_text(raw) == normalized


@pytest.mark.parametrize(
    ("query", "status", "via", "entity_id"),
    [
        ("Altars of Sorrow", "entity", "canonical", "activity.altars_of_sorrow"),
        ("  altárs OF sorrow? ", "entity", "canonical", "activity.altars_of_sorrow"),
        ("exotic armor", "entity", "alias", "rule.exotic_equip_limit"),
        ("HC", "entity", "shorthand", "weapon_archetype.hand_cannon"),
        ("hand cannons", "entity", "plural", "weapon_archetype.hand_cannon"),
        ("dungeons", "entity", "plural", "activity_type.dungeon"),
        ("Crucibles", "unknown", None, None),
        ("shotty", "unknown", None, None),
        ("Sorrow's Harbor", "unknown", None, None),
        ("", "unknown", None, None),
    ],
)
def test_resolution_is_exact_and_explicit(
    query: str, status: str, via: str | None, entity_id: str | None
) -> None:
    resolution = engine().resolve(query)

    assert resolution.status == status
    assert resolution.via == via
    assert resolution.entity_id == entity_id


def test_ambiguous_alias_never_picks_a_candidate() -> None:
    resolution = engine().resolve("Altars")

    assert resolution.status == "ambiguous"
    assert resolution.entity_id is None
    assert resolution.candidate_ids == (
        "activity.altars_of_sorrow",
        "activity.altars_of_summoning",
        "activity.altar_of_reflection",
    )


def test_longest_span_wins_over_a_shorter_ambiguous_alias() -> None:
    mentions = engine().mentions("Is Altars of Sorrow better than other altars?")

    assert [(value.text, value.resolution.status) for value in mentions] == [
        ("altars of sorrow", "entity"),
        ("altars", "ambiguous"),
    ]
    assert (mentions[0].start_token, mentions[0].end_token) == (1, 4)


@pytest.mark.parametrize("text", ["hcs are fun", "ahc", "high caliber", "hand-cannoneer"])
def test_matching_is_whole_token_only(text: str) -> None:
    assert engine().mentions(text) == []


def test_spans_are_bounded_to_six_tokens() -> None:
    six = "one two three four five six"
    seven = "one two three four five six seven"
    custom = DestinyKnowledgeEngine(
        parse_core_knowledge(
            with_entities(entity("concept.six", six), entity("concept.seven", seven))
        )
    )

    assert [value.text for value in custom.mentions(f"say {six} now")] == [six]
    # The 7-token name is never considered as a span; its 6-token prefix is the longest
    # span available, so only the 6-token entity can match.
    seven_mentions = custom.mentions(f"say {seven} now")
    assert [value.resolution.entity_id for value in seven_mentions] == ["concept.six"]
    # Exact whole-phrase lookup is not span-bounded.
    assert custom.resolve(seven).entity_id == "concept.seven"


def test_unknown_entities_produce_no_mentions_and_no_fuzzy_fallback() -> None:
    assert engine().mentions("Vault of Glass and King's Fall then Garden of Salvation") == []


def test_misspellings_do_not_resolve_without_an_explicit_misspelling_alias() -> None:
    mentions = engine().mentions("Where is Altars of Sorow and the dungoen?")

    # "Sorow" is not registered, so only the ambiguous "altars" span matches.
    assert [(value.text, value.resolution.status) for value in mentions] == [
        ("altars", "ambiguous")
    ]


def test_lookup_returns_entity_ambiguity_or_unknown_cards() -> None:
    knowledge = engine()

    entity_card = knowledge.lookup("Altars of Sorrow")
    ambiguous = knowledge.lookup("altar")
    unknown = knowledge.lookup("Vault of Glass")

    assert entity_card["status"] == "entity"
    assert entity_card["entity"]["facts"][0] == [
        "located_on",
        "The Moon",
        "manifest_structure+curated",
    ]
    assert entity_card["entity"]["not_same_as"] == ["Haunted Altars of Sorrow"]
    assert ambiguous["status"] == "ambiguous"
    assert [value["name"] for value in ambiguous["candidates"]] == [
        "Altars of Sorrow",
        "Altars of Summoning",
        "Altar of Reflection",
    ]
    assert unknown == {"status": "unknown", "query": "Vault of Glass"}


def test_facts_for_reports_source_kinds_in_order_without_duplicates() -> None:
    facts = engine().facts_for("activity.altars_of_sorrow")

    assert facts[0].predicate == "located_on"
    assert facts[0].source_kinds == ("manifest_structure", "curated")
    assert facts[2].value == "open_world"


# --- Grounded facts bundle ---------------------------------------------------------------


def test_bundle_is_none_when_nothing_matches() -> None:
    assert engine().bundle("I want something chill for an hour.") is None
    assert engine().bundle("", ()) is None


def test_bundle_contains_matched_entities_rules_and_shorthand_provenance() -> None:
    bundle = engine().bundle("Can I run two exotics with a HC in a dungeon?")

    assert bundle is not None
    payload = bundle.payload
    assert [value["name"] for value in payload["entities"]] == ["Hand Cannon", "Dungeon"]
    assert payload["entities"][0]["matched"] == {"text": "hc", "via": "shorthand"}
    assert [value["name"] for value in payload["rules"]] == ["Exotic equip limit"]
    assert bundle.text == json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert bundle.char_count == len(bundle.text) <= MAX_BUNDLE_CHARS
    assert bundle.truncated is False


def test_ambiguity_is_explicit_in_the_bundle() -> None:
    bundle = engine().bundle("Is Altars worth doing?")

    assert bundle is not None
    assert bundle.payload["ambiguous"] == [
        {
            "matched_text": "altars",
            "candidates": [
                {"name": "Altars of Sorrow", "kind": "activity"},
                {"name": "Altars of Summoning", "kind": "activity", "status": "legacy"},
                {"name": "Altar of Reflection", "kind": "activity", "status": "legacy"},
            ],
        }
    ]
    assert "entities" not in bundle.payload


def test_bundle_carries_evidence_only_and_no_curator_notes() -> None:
    bundle = engine().bundle("Altars of Sorrow or altars?", ("Dungeon",))

    assert bundle is not None
    text = bundle.text.casefold()
    for instruction in ("do not", "must", "should", "ask", "curator note"):
        assert instruction not in text
    allowed = {"corpus_version", "entities", "rules", "ambiguous", "omitted"}
    assert set(bundle.payload) <= allowed


def test_planning_candidate_entities_rank_after_message_entities() -> None:
    bundle = engine().bundle("What is a dungeon?", ("Altars of Sorrow: Nightmares", "Dungeon"))

    assert bundle is not None
    names = [value["name"] for value in bundle.payload["entities"]]
    # Dungeon is named by the user; it is not repeated or demoted by the candidate mention.
    assert names == ["Dungeon", "Altars of Sorrow"]
    assert "matched" not in bundle.payload["entities"][0]


def test_bundle_output_is_byte_identical_for_identical_input() -> None:
    message = "Is Altars of Sorrow a dungeon? Can I use exotics with hand cannons? altar"
    candidates = ("Altars of Sorrow: Nightmares", "Haunted Altars of Sorrow")
    first = engine().bundle(message, candidates)
    second = engine().bundle(message, candidates)
    repeated = DestinyKnowledgeEngine.from_file(FIXTURE).bundle(message, candidates)

    assert first is not None and second is not None and repeated is not None
    assert first.text == second.text == repeated.text


def _many_entities(count: int, *, kind: str = "concept", summary_size: int = 40) -> list[Any]:
    return [
        entity(
            f"{kind}.item_{index}",
            f"Synthetic {kind} {index}",
            kind=kind,
            summary=("s" * summary_size)[:300],
            facts=[
                {"p": "fireteam_size_max", "o": index, "src": "fixture.curated"} for _ in range(8)
            ],
        )
        for index in range(count)
    ]


def test_bundle_count_caps_and_fact_cap_are_enforced_with_explicit_omissions() -> None:
    concepts = _many_entities(7)
    rules = _many_entities(4, kind="rule")
    groups = [
        {
            "id": f"ambiguity.group_{index}",
            "aliases": [f"ambiguous word {index}"],
            "candidates": ["mode.crucible", "activity_type.dungeon"],
        }
        for index in range(3)
    ]
    payload = with_entities(*concepts, *rules)
    payload["ambiguity_groups"].extend(groups)
    custom = DestinyKnowledgeEngine(parse_core_knowledge(payload))
    message = " and ".join(
        [value["name"] for value in concepts + rules] + [f"ambiguous word {i}" for i in range(3)]
    )

    bundle = custom.bundle(message)

    assert bundle is not None
    data = bundle.payload
    assert len(data["entities"]) == MAX_BUNDLE_ENTITIES
    assert len(data["rules"]) <= MAX_BUNDLE_RULES
    assert len(data["ambiguous"]) == MAX_BUNDLE_AMBIGUITIES
    assert all(len(value["facts"]) == MAX_BUNDLE_FACTS_PER_ENTITY for value in data["entities"])
    assert all(value["facts_omitted"] == 2 for value in data["entities"])
    assert data["omitted"]["ambiguity_groups"] == 1
    assert data["omitted"]["entities"] >= 2
    assert bundle.truncated is True
    assert bundle.char_count <= MAX_BUNDLE_CHARS


def test_size_budget_drops_rules_then_candidates_then_message_entities() -> None:
    big = _many_entities(4, summary_size=300)
    big_rules = _many_entities(2, kind="rule", summary_size=300)
    custom = DestinyKnowledgeEngine(parse_core_knowledge(with_entities(*big, *big_rules)))
    message_names = [big[0]["name"], big[1]["name"], big_rules[0]["name"], big_rules[1]["name"]]
    candidate_names = (big[2]["name"], big[3]["name"])

    bundle = custom.bundle(" and ".join(message_names), candidate_names)

    assert bundle is not None
    assert bundle.char_count <= MAX_BUNDLE_CHARS
    kept = [value["name"] for value in bundle.payload["entities"]]
    # Both rules go first, then the last candidate entity; message entities survive.
    assert kept == [big[0]["name"], big[1]["name"], big[2]["name"]]
    assert "rules" not in bundle.payload
    assert bundle.payload["omitted"] == {"entities": 1, "rules": 2}


def test_ambiguity_groups_survive_the_size_budget() -> None:
    big = _many_entities(5, summary_size=300)
    custom = DestinyKnowledgeEngine(parse_core_knowledge(with_entities(*big)))

    bundle = custom.bundle(" and ".join(value["name"] for value in big) + " and altars")

    assert bundle is not None
    assert bundle.payload["ambiguous"][0]["matched_text"] == "altars"
    assert bundle.char_count <= MAX_BUNDLE_CHARS
    assert bundle.payload["omitted"]["entities"] >= 1


# --- Hot path guarantees -----------------------------------------------------------------


def test_engine_never_reads_disk_after_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    knowledge = engine()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("The engine must not touch the filesystem after initialization.")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)

    assert knowledge.resolve("HC").status == "entity"
    assert knowledge.mentions("Altars of Sorrow")
    assert knowledge.lookup("Dungeon")["status"] == "entity"
    assert knowledge.bundle("dungeon and exotics", ("Altars of Sorrow",)) is not None


def test_engine_is_unaffected_by_later_corpus_mutation() -> None:
    corpus = load_core_knowledge(FIXTURE)
    knowledge = DestinyKnowledgeEngine(corpus)
    before = knowledge.bundle("dungeon")

    dungeon = next(value for value in corpus.entities if value.id == "activity_type.dungeon")
    dungeon.summary = "Mutated after construction."
    dungeon.facts.clear()
    corpus.entities.clear()

    after = knowledge.bundle("dungeon")
    assert before is not None and after is not None
    assert after.text == before.text


def test_hot_path_timing_ceiling_on_a_large_synthetic_corpus() -> None:
    synthetic = [
        entity(
            f"concept.synthetic_{index}",
            f"Synthetic Concept {index}",
            aliases=[f"synthetic alias {index}", f"second synthetic alias {index}"],
            plural=True,
            facts=[{"p": "game_mode", "o": "pve", "src": "fixture.curated"}],
        )
        for index in range(600)
    ]
    started = time.perf_counter()
    custom = DestinyKnowledgeEngine(parse_core_knowledge(with_entities(*synthetic)))
    build_seconds = time.perf_counter() - started
    words = (
        "what should I do next with my titan in a dungeon or Altars of Sorrow using a HC and "
        "two exotics synthetic concept 42 synthetic alias 7 "
    )
    message = (words * 20)[:2000]
    candidates = tuple(f"Candidate objective {index}: Altars of Sorrow" for index in range(12))

    durations = []
    for _ in range(50):
        started = time.perf_counter()
        custom.bundle(message, candidates)
        durations.append(time.perf_counter() - started)

    median_ms = statistics.median(durations) * 1000
    print(f"engine build={build_seconds * 1000:.1f}ms bundle median={median_ms:.2f}ms")
    assert build_seconds < 2.0
    assert median_ms < 15.0
