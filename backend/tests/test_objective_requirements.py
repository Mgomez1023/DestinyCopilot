"""M4A: structured, conservative objective requirements from Manifest text.

Synthetic sentences use the real Manifest grammar found in the audit; none names a production
quest, bounty or activity. The real-Manifest tests at the end check the parser against actual
definitions and are skipped when the local Manifest is not available.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app import objective_requirements
from app.config import Settings
from app.destiny_knowledge import DestinyKnowledgeService, ManifestKnowledgeProvider
from app.guardian_refresh import CACHE_SCHEMA_VERSION
from app.guardian_tools import GuardianToolService
from app.models import (
    ObjectiveRequirement,
    ObjectiveSummary,
    QuestSummary,
    RequirementSource,
)
from app.objective_requirements import (
    compact_requirement,
    is_restrictive,
    item_description_requirements,
    objective_definition_requirement,
    parse_requirements,
    requirements_consistency,
)
from tests.test_build_analysis import build_context
from tests.test_guardian import _normalize_profile

SOURCE = RequirementSource(definition="DestinyInventoryItemDefinition", hash=1, field="f")
OBJECTIVE_SOURCE = RequirementSource(definition="DestinyObjectiveDefinition", hash=2, field="p")


def parse(text: str, kind: str = "description") -> list[ObjectiveRequirement]:
    source = OBJECTIVE_SOURCE if kind == "objective_label" else SOURCE
    return parse_requirements(text, kind=kind, source=source)  # type: ignore[arg-type]


def one(text: str, kind: str = "description") -> ObjectiveRequirement:
    found = parse(text, kind)
    assert len(found) == 1, found
    return found[0]


# --- Source: ability versus weapon ---


def test_ability_final_blows_are_ability_final_blows() -> None:
    requirement = one("Defeat combatants or Guardians with Arc ability final blows.")

    assert requirement.source_kind == "ability"
    assert requirement.element == "Arc"
    assert requirement.completion_action == "final_blow"
    assert requirement.target_kind == "combatants_or_guardians"
    assert requirement.parse_status == "verified"


def test_weapon_final_blows_are_weapon_final_blows() -> None:
    requirement = one("Defeat combatants with Void weapon final blows.")

    assert (requirement.source_kind, requirement.element) == ("weapon", "Void")
    assert requirement.completion_action == "final_blow"


@pytest.mark.parametrize("element", ["Arc", "Solar", "Void", "Stasis", "Strand"])
def test_every_element_is_recognized_on_both_sources(element: str) -> None:
    ability = one(f"Defeat combatants with {element} ability final blows.")
    weapon = one(f"Defeat combatants with {element} weapon final blows.")

    assert (ability.source_kind, ability.element) == ("ability", element)
    assert (weapon.source_kind, weapon.element) == ("weapon", element)


@pytest.mark.parametrize(
    ("text", "modifier"),
    [
        ("Get grenade final blows.", "grenade"),
        ("Get melee final blows.", "melee"),
        ("Defeat combatants using your Super.", "super"),
    ],
)
def test_named_abilities_are_abilities_with_a_modifier(text: str, modifier: str) -> None:
    requirement = one(text)

    assert requirement.source_kind == "ability" and modifier in requirement.modifiers


def test_a_grenade_launcher_is_a_weapon_not_a_grenade_ability() -> None:
    requirement = one("Defeat combatants using Grenade Launchers.")

    assert requirement.source_kind == "weapon"
    assert "grenade" not in requirement.modifiers


def test_abilities_and_weapons_together_are_ambiguous_not_merged() -> None:
    requirement = one("Defeat combatants with Solar weapons or abilities.")

    assert requirement.parse_status == "ambiguous"
    assert requirement.source_kind == "unknown"


# --- Elements ---


def test_elemental_damage_does_not_name_a_weapon_or_an_ability() -> None:
    requirement = one("Defeat combatants with Arc damage.")

    assert requirement.element == "Arc"
    assert requirement.source_kind == "unknown"  # damage from any source; not assumed either way


def test_elemental_weapons_do_not_satisfy_an_elemental_ability_requirement() -> None:
    ability = one("Defeat combatants with Arc ability final blows.")
    weapon = one("Defeat combatants with Arc weapon final blows.")

    assert ability.source_kind != weapon.source_kind
    assert requirements_consistency([ability, weapon_from_objective(weapon)]) == "conflicting"


def weapon_from_objective(requirement: ObjectiveRequirement) -> ObjectiveRequirement:
    return requirement.model_copy(update={"source": OBJECTIVE_SOURCE})


def test_matching_and_nonmatching_elements_are_compared_not_assumed() -> None:
    arc = one("Defeat combatants with Arc ability final blows.")
    solar = one("Defeat combatants with Solar ability final blows.", "description")

    assert arc.element != solar.element
    assert requirements_consistency([arc, weapon_from_objective(solar)]) == "conflicting"
    assert requirements_consistency([arc, weapon_from_objective(arc)]) == "consistent"


@pytest.mark.parametrize(
    "text",
    [
        "Defeat combatants affected by Arc debuffs.",
        "Defeat combatants while using your Solar subclass.",
        "Defeat a strong adversary with a Strand subclass equipped.",
    ],
)
def test_an_element_that_does_not_describe_damage_is_not_an_element_requirement(text: str) -> None:
    assert all(value.element == "unknown" for value in parse(text))


def test_a_list_of_elements_is_ambiguous() -> None:
    for text in (
        "Get Solar, Arc, Void, Strand, or Stasis final blows.",
        "Defeat Vex with Arc, Solar, and Void weapons.",
    ):
        requirement = one(text)

        assert requirement.parse_status == "ambiguous"
        assert requirement.element == "unknown"


def test_icon_glyph_markup_is_ignored_but_the_raw_text_keeps_it() -> None:
    text = "Defeat combatants or Guardians with [Stasis]Stasis damage final blows."
    requirement = one(text)

    assert (requirement.element, requirement.completion_action) == ("Stasis", "final_blow")
    assert requirement.raw_requirement == text


# --- Actions: defeat, damage, final blows ---


def test_a_generic_defeat_is_not_a_final_blow_requirement() -> None:
    requirement = one("Defeat Guardians.")

    assert requirement.completion_action == "defeat"
    assert requirement.source_kind == "unknown" and requirement.element == "unknown"
    assert requirement.target_kind == "guardians"
    assert requirement.parse_status == "verified"
    assert not is_restrictive(requirement)


def test_final_blows_are_only_recorded_when_the_text_says_so() -> None:
    assert one("Defeat combatants with Solar damage.").completion_action == "defeat"
    assert one("Defeat combatants with Solar damage final blows.").completion_action == "final_blow"


def test_dealing_damage_is_a_damage_action() -> None:
    requirement = one("Deal Solar damage to combatants.")

    assert requirement.completion_action == "damage" and requirement.element == "Solar"


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("Defeat combatants.", "combatants"),
        ("Defeat opposing Guardians.", "guardians"),
        ("Defeat opponents in the Crucible.", "opponents"),
        ("Defeat enemies.", "enemies"),
        ("Defeat targets.", "targets"),
        ("Defeat a Hive Wizard.", "unknown"),
    ],
)
def test_target_kind_is_only_what_the_text_names(text: str, target: str) -> None:
    assert one(text).target_kind == target


def test_precision_is_a_modifier_not_a_source() -> None:
    requirement = one("Defeat combatants with precision final blows.")

    assert requirement.modifiers == ["precision"]
    assert requirement.source_kind == "unknown"


# --- Bonus progress, extra conditions, multiple clauses ---


def test_bonus_progress_clauses_are_kept_raw_and_never_interpreted() -> None:
    requirement = one(
        "Defeat Guardians with Arc damage. Dealing final blows with the Arc Super grants "
        "bonus progress."
    )

    assert requirement.bonus_conditions == [
        "Dealing final blows with the Arc Super grants bonus progress."
    ]
    # The bonus clause mentions final blows and a Super; the requirement does not.
    assert requirement.completion_action == "defeat"
    assert requirement.source_kind == "unknown" and requirement.modifiers == []


def test_a_bonus_only_text_has_no_requirement() -> None:
    assert parse("Personal final blows grant bonus progress.") == []


def test_places_and_team_conditions_are_preserved_as_raw_conditions() -> None:
    requirement = one("As a fireteam, defeat opponents with Solar weapon final blows in Trials.")

    assert requirement.parse_status == "verified"
    assert "As a fireteam" in requirement.conditions
    assert any("Trials" in value for value in requirement.conditions)


def test_an_unrecognized_restriction_makes_the_parse_partial_and_is_kept() -> None:
    requirement = one("Defeat a strong adversary while using your Solar subclass.")

    assert requirement.parse_status == "partial"
    assert any("subclass" in value for value in requirement.conditions)
    assert requirement.element == "unknown"


def test_a_named_weapon_type_is_a_partial_weapon_requirement() -> None:
    requirement = one("Defeat targets while using any Fusion Rifle.")

    assert requirement.source_kind == "weapon" and requirement.parse_status == "partial"
    assert any("Fusion Rifle" in value for value in requirement.conditions)


def test_a_second_objective_in_the_same_sentence_makes_it_partial() -> None:
    requirement = one("Complete Gambit matches and defeat targets as a team using Arc damage.")

    assert requirement.parse_status == "partial" and requirement.element == "Arc"


def test_several_requirement_clauses_in_one_sentence_are_ambiguous_and_unmerged() -> None:
    requirement = one(
        "Defeat targets with precision final blows, defeat targets with melee final blows."
    )

    assert requirement.parse_status == "ambiguous"
    assert requirement.source_kind == "unknown" and requirement.modifiers == []
    assert requirement.completion_action == "unknown"


def test_alternative_final_blow_clauses_are_ambiguous() -> None:
    requirement = one("Land Arc final blows or final blows with the Prismatic subclass equipped.")

    assert requirement.parse_status == "ambiguous"


def test_separate_requirement_sentences_are_reported_separately() -> None:
    found = parse("Defeat combatants with Arc damage. Defeat Guardians with Void damage.")

    assert [value.element for value in found] == ["Arc", "Void"]


# --- Text that is not a requirement ---


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Complete the mission to advance the story.",
        "This weapon tracks the number of enemies you defeated with it.",
        "Energy weapons deal elemental damage.",
        "Collect tracking devices.",
    ],
)
def test_text_without_a_requirement_yields_nothing(text: str) -> None:
    assert parse(text) == []


def test_missing_text_yields_nothing() -> None:
    assert parse_requirements(None, kind="description", source=SOURCE) == []


# --- Objective labels ---


@pytest.mark.parametrize(
    ("label", "source_kind", "action", "status"),
    [
        ("Grenade kills", "ability", "defeat", "verified"),
        ("Arc weapon final blows", "weapon", "final_blow", "verified"),
        ("Super", "ability", "unknown", "partial"),
    ],
)
def test_short_labels_parse_only_what_they_state(
    label: str, source_kind: str, action: str, status: str
) -> None:
    requirement = one(label, "objective_label")

    assert (requirement.source_kind, requirement.completion_action, requirement.parse_status) == (
        source_kind,
        action,
        status,
    )


def test_a_generic_label_is_not_restrictive() -> None:
    assert all(not is_restrictive(value) for value in parse("Targets defeated", "objective_label"))
    assert objective_definition_requirement(5, {"progressDescription": "Targets defeated"}) is None


def test_an_objective_whose_label_carries_nothing_falls_back_to_its_description() -> None:
    definition = {
        "displayProperties": {"name": "", "description": "Defeat combatants with Void abilities."},
        "progressDescription": "Progress",
    }
    requirement = objective_definition_requirement(9, definition)

    assert requirement is not None and requirement.source.field == "displayProperties.description"
    assert (requirement.source_kind, requirement.element) == ("ability", "Void")


def test_a_missing_objective_definition_yields_nothing() -> None:
    assert objective_definition_requirement(9, {}) is None


# --- Provenance and raw text ---


def test_every_requirement_carries_its_source_and_raw_sentence() -> None:
    text = "Defeat combatants or Guardians with Arc ability final blows. Extra text here."
    requirement = one(text)

    assert requirement.source == SOURCE
    assert (
        requirement.raw_requirement
        == "Defeat combatants or Guardians with Arc ability final blows."
    )


def test_item_description_requirements_cite_the_item_definition_and_field() -> None:
    definition = {"displayProperties": {"description": "Defeat Guardians with Solar damage."}}
    requirement = item_description_requirements(77, definition)[0]

    assert requirement.source == RequirementSource(
        definition="DestinyInventoryItemDefinition", hash=77, field="displayProperties.description"
    )


def test_the_serialized_form_drops_unknowns_but_keeps_provenance_and_status() -> None:
    data = one("Defeat Guardians.").model_dump(mode="json")

    assert set(data) == {
        "raw_requirement",
        "source",
        "parse_status",
        "completion_action",
        "target_kind",
    }
    assert ObjectiveRequirement.model_validate(data).source_kind == "unknown"


def test_compact_form_omits_text_the_result_already_carries() -> None:
    requirement = one(
        "Defeat combatants with Arc ability final blows. Greater progress is bonus progress."
    )
    full = compact_requirement(requirement)
    compact = compact_requirement(requirement, include_raw=False)

    assert full["raw_requirement"].startswith("Defeat combatants")
    assert "raw_requirement" not in compact and "bonus_conditions" not in compact
    assert compact["bonus_clauses"] == 1
    assert compact["source"] == "DestinyInventoryItemDefinition:1:f"


def test_the_module_names_no_items_activities_or_quests() -> None:
    source = Path(objective_requirements.__file__).read_text(encoding="utf-8").casefold()

    for name in ("scourge", "arc scourge", "trials of osiris", "gambit", "neomuna", "europa"):
        assert name not in source


# --- Consistency across sources ---


def objective_req(text: str) -> ObjectiveRequirement:
    return one(text, "objective_label").model_copy(update={"source": OBJECTIVE_SOURCE})


def test_defeat_versus_final_blows_differs_in_specificity_not_conflict() -> None:
    description = one("Defeat combatants with Kinetic damage.")

    assert requirements_consistency([description, objective_req("Kinetic final blows")]) == (
        "differs_in_specificity"
    )


def test_objectives_are_not_compared_with_each_other() -> None:
    description = one("Defeat Vex with Arc, Solar, and Void weapons.")  # ambiguous, excluded
    objectives = [objective_req(f"{value} weapon") for value in ("Arc", "Solar", "Void")]

    assert requirements_consistency([description, *objectives]) is None


def test_a_source_silent_on_a_field_never_disagrees() -> None:
    description = one("Defeat combatants with Solar weapon final blows.")

    assert requirements_consistency([description, objective_req("Weapon")]) == "consistent"


def test_there_is_nothing_to_compare_with_a_single_kind_of_source() -> None:
    assert requirements_consistency([one("Defeat Guardians.")]) is None
    assert requirements_consistency([]) is None


def test_ambiguous_requirements_do_not_create_conflicts() -> None:
    ambiguous = one("Get Solar, Arc, or Void final blows.")

    assert requirements_consistency([ambiguous, objective_req("Arc weapon")]) is None


# --- Normalization, tools and cache ---

ITEM = "DestinyInventoryItemDefinition"
OBJECTIVE = "DestinyObjectiveDefinition"


def named(name: str, description: str = "", **extra: Any) -> dict[str, Any]:
    return {"displayProperties": {"name": name, "description": description}, **extra}


DEFINITIONS: dict[str, dict[int, dict[str, Any]]] = {
    "DestinyClassDefinition": {10: named("Titan")},
    ITEM: {
        500: named(
            "Test Quest",
            "Complete the line.",
            itemType=15,
            itemTypeDisplayName="Quest",
        ),
        501: named(
            "Step One",
            "Defeat combatants with Arc ability final blows. Greater progress is bonus progress.",
            itemType=12,
            itemTypeDisplayName="Quest Step",
        ),
        510: named(
            "Test Bounty",
            "Defeat opponents with Solar weapon final blows.",
            itemType=26,
            itemTypeDisplayName="Bounty",
        ),
        511: named("Plain Bounty", "Defeat Guardians.", itemType=26, itemTypeDisplayName="Bounty"),
        512: named(
            "Story Step", "Complete the mission.", itemType=26, itemTypeDisplayName="Bounty"
        ),
    },
    OBJECTIVE: {
        900: {**named(""), "progressDescription": "Solar weapon", "completionValue": 5},
        901: {**named(""), "progressDescription": "Targets defeated", "completionValue": 5},
        902: {**named(""), "progressDescription": "Arc weapon final blows", "completionValue": 5},
    },
}


def profile_with_quest() -> dict[str, Any]:
    return {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "characterProgressions": {
            "data": {
                "c1": {
                    "quests": [
                        {
                            "questHash": 500,
                            "stepHash": 501,
                            "stepObjectives": [
                                {"objectiveHash": 902, "progress": 1, "completionValue": 5},
                                {"objectiveHash": 901, "progress": 0, "completionValue": 5},
                            ],
                        }
                    ]
                }
            }
        },
        "characterInventories": {
            "data": {
                "c1": {
                    "items": [
                        {"itemHash": 510, "itemInstanceId": "b1"},
                        {"itemHash": 511, "itemInstanceId": "b2"},
                        {"itemHash": 512, "itemInstanceId": "b3"},
                    ]
                }
            }
        },
        "itemComponents": {
            "objectives": {
                "data": {
                    "b1": {
                        "objectives": [{"objectiveHash": 900, "progress": 0, "completionValue": 5}]
                    },
                    "b2": {
                        "objectives": [{"objectiveHash": 901, "progress": 0, "completionValue": 5}]
                    },
                    "b3": {"objectives": []},
                }
            }
        },
    }


def normalized_quests(definitions: Any = None) -> dict[str, QuestSummary]:
    context = _normalize_profile(definitions or DEFINITIONS, profile_with_quest())
    return {value.name: value for value in context.characters[0].quests}


def test_quest_description_step_description_and_objective_text_are_kept_apart() -> None:
    quest = normalized_quests()["Test Quest"]
    step = next(value for value in quest.requirements if value.source.hash == 501)

    assert quest.requirements_parsed is True
    assert quest.description == "Complete the line."  # existing field unchanged
    assert [value.source.hash for value in quest.requirements] == [501]
    assert (step.source_kind, step.element, step.completion_action) == (
        "ability",
        "Arc",
        "final_blow",
    )
    assert step.bonus_conditions == ["Greater progress is bonus progress."]


def test_only_restrictive_objective_text_is_attached_to_objectives() -> None:
    objectives = {
        value.objective_hash: value for value in normalized_quests()["Test Quest"].objectives
    }

    assert objectives[902].requirement is not None
    assert objectives[902].requirement.source.definition == "DestinyObjectiveDefinition"
    assert objectives[902].requirement.source.hash == 902
    assert objectives[901].requirement is None  # "Targets defeated" restricts nothing


def test_consistency_compares_the_description_with_the_objective_text() -> None:
    quests = normalized_quests()

    assert quests["Test Bounty"].requirements_consistency == "consistent"
    assert quests["Plain Bounty"].requirements_consistency is None


def test_a_quest_step_that_conflicts_with_its_objective_is_reported_not_merged() -> None:
    quest = normalized_quests()["Test Quest"]

    # Step: Arc ability final blows. Objective: Arc weapon final blows.
    assert quest.requirements_consistency == "conflicting"
    assert quest.requirements[0].source_kind == "ability"
    assert quest.objectives[0].requirement.source_kind == "weapon"  # type: ignore[union-attr]


def test_bounties_without_requirements_are_parsed_and_empty() -> None:
    quest = normalized_quests()["Story Step"]

    assert quest.requirements == [] and quest.requirements_parsed is True


def test_generic_defeat_bounty_stays_generic() -> None:
    requirement = normalized_quests()["Plain Bounty"].requirements[0]

    assert requirement.completion_action == "defeat"
    assert requirement.source_kind == "unknown" and requirement.parse_status == "verified"


def test_missing_definitions_yield_no_requirements_and_no_error() -> None:
    thin = {
        **DEFINITIONS,
        ITEM: {key: value for key, value in DEFINITIONS[ITEM].items() if key != 501},
        OBJECTIVE: {},
    }
    quest = normalized_quests(thin)["Test Quest"]

    assert quest.requirements == [] and quest.requirements_consistency is None
    assert all(value.requirement is None for value in quest.objectives)


def test_serialization_is_compact_and_round_trips() -> None:
    quest = normalized_quests()["Test Quest"]
    payload = quest.model_dump(mode="json")

    assert QuestSummary.model_validate(payload) == quest
    plain = normalized_quests()["Story Step"].model_dump(mode="json")
    assert "requirements" not in plain  # empty lists are omitted
    assert plain["requirements_parsed"] is True


def test_a_quest_cached_before_requirements_loads_as_not_parsed() -> None:
    old = {"quest_hash": 1, "name": "Old", "character_id": "c1"}

    restored = QuestSummary.model_validate(old)

    assert restored.requirements == [] and restored.requirements_parsed is False
    assert "requirements_parsed" not in restored.model_dump(mode="json")


def test_an_objective_without_a_requirement_serializes_unchanged() -> None:
    summary = ObjectiveSummary(objective_hash=1, name="x")

    assert "requirement" not in summary.model_dump(mode="json")


def test_the_cache_schema_covers_requirements() -> None:
    assert CACHE_SCHEMA_VERSION >= 7


def quest_tool_result(definitions: Any = None) -> dict[str, Any]:
    context = _normalize_profile(definitions or DEFINITIONS, profile_with_quest())
    return GuardianToolService(context).execute(
        "get_active_quests", {"character_id": None, "character_class": None}
    )


def test_get_active_quests_exposes_requirements_compactly() -> None:
    quests = {value["name"]: value for value in quest_tool_result()["quests"]}
    step = quests["Test Quest"]["requirements"][0]
    bounty = quests["Test Bounty"]["requirements"][0]

    assert step["source"] == "DestinyInventoryItemDefinition:501:displayProperties.description"
    assert step["raw_requirement"].startswith(
        "Defeat combatants"
    )  # the step text is shown nowhere else
    assert quests["Test Quest"]["requirements_consistency"] == "conflicting"
    # The bounty's own description already holds the text, so it is not repeated.
    assert "raw_requirement" not in bounty and bounty["source_kind"] == "weapon"
    assert "requirements" not in quests["Story Step"]


def test_objectives_in_the_quest_tool_carry_their_requirement() -> None:
    quests = {value["name"]: value for value in quest_tool_result()["quests"]}
    objective = next(v for v in quests["Test Quest"]["objectives"] if v["objective_hash"] == 902)

    assert objective["requirement"]["source_kind"] == "weapon"
    assert objective["requirement"]["raw_requirement"] == "Arc weapon final blows"


def test_planning_candidates_and_other_tools_are_untouched_by_m4a() -> None:
    from app.session_planning import PlanningCandidate

    assert "requirements" not in PlanningCandidate.model_fields
    result = GuardianToolService(build_context()).execute(
        "get_build_details", {"character_id": None, "character_class": None}
    )
    assert "requirements" not in json.dumps(result)


# --- Knowledge tools ---


class KnowledgeResolver:
    def __init__(self) -> None:
        self.tables = {
            ITEM: {
                "510": {
                    **DEFINITIONS[ITEM][510],
                    "hash": 510,
                    "objectives": {"objectiveHashes": [900]},
                },
                "511": {
                    **DEFINITIONS[ITEM][511],
                    "hash": 511,
                    "objectives": {"objectiveHashes": [901]},
                },
                "500": {
                    **DEFINITIONS[ITEM][500],
                    "hash": 500,
                    "setData": {"itemList": [{"itemHash": 501}]},
                },
                "501": {
                    **DEFINITIONS[ITEM][501],
                    "hash": 501,
                    "objectives": {"objectiveHashes": [902]},
                },
            },
            OBJECTIVE: {
                str(key): {**value, "hash": key} for key, value in DEFINITIONS[OBJECTIVE].items()
            },
        }

    async def manifest_version(self) -> str:
        return "m4a-tests"

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


def item_details(tmp_path: Any, query: str) -> dict[str, Any]:
    provider = ManifestKnowledgeProvider(KnowledgeResolver(), Settings(manifest_cache_dir=tmp_path))  # type: ignore[arg-type]
    return asyncio.run(
        DestinyKnowledgeService([provider]).execute(
            "get_item_details", {"item_name_or_hash": query}
        )
    )


def test_item_details_report_description_and_objective_requirements(tmp_path: Any) -> None:
    result = item_details(tmp_path, "510")
    sources = [value["source"] for value in result["requirements"]]

    assert sources == [
        "DestinyInventoryItemDefinition:510:displayProperties.description",
        "DestinyObjectiveDefinition:900:progressDescription",
    ]
    assert result["requirements_consistency"] == "consistent"
    assert "raw_requirement" not in result["requirements"][0]  # item.description already has it
    assert result["item"]["description"].startswith("Defeat opponents")


def test_item_details_without_a_restrictive_requirement_add_nothing_but_generic(
    tmp_path: Any,
) -> None:
    result = item_details(tmp_path, "511")

    assert [value["completion_action"] for value in result["requirements"]] == ["defeat"]
    assert "requirements_consistency" not in result


# --- The real Manifest (skipped when it is not available locally) ---


def _manifest_path() -> Path | None:
    try:
        from evals.runtime import DEFAULT_MANIFEST
    except Exception:  # noqa: BLE001
        return None
    return DEFAULT_MANIFEST if DEFAULT_MANIFEST and Path(DEFAULT_MANIFEST).is_file() else None


REAL = _manifest_path()
needs_manifest = pytest.mark.skipif(REAL is None, reason="local Manifest not available")


def real(table: str, entity_hash: int) -> dict[str, Any]:
    assert REAL is not None
    signed = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
    with sqlite3.connect(str(REAL)) as connection:
        row = connection.execute(f'SELECT json FROM "{table}" WHERE id = ?', (signed,)).fetchone()
    assert row is not None, (table, entity_hash)
    return json.loads(row[0])


def real_requirements(item_hash: int) -> list[ObjectiveRequirement]:
    definition = real(ITEM, item_hash)
    found = item_description_requirements(item_hash, definition)
    for objective_hash in definition.get("objectives", {}).get("objectiveHashes", []):
        parsed = objective_definition_requirement(objective_hash, real(OBJECTIVE, objective_hash))
        if parsed:
            found.append(parsed)
    return found


@needs_manifest
def test_real_arc_scourge_is_an_arc_ability_final_blow_requirement() -> None:
    requirement = real_requirements(1405737718)[0]

    assert requirement.source == RequirementSource(
        definition=ITEM, hash=1405737718, field="displayProperties.description"
    )
    assert (requirement.source_kind, requirement.element) == ("ability", "Arc")
    assert requirement.completion_action == "final_blow"
    assert requirement.target_kind == "combatants_or_guardians"
    assert requirement.parse_status == "verified"
    assert requirement.bonus_conditions == [
        "Combatants affected by Arc debuffs grant greater progress."
    ]
    assert (
        requirement.raw_requirement
        == real(ITEM, 1405737718)["displayProperties"]["description"].split(". ")[0] + "."
    )


@needs_manifest
def test_real_sibling_definitions_generalize_across_elements() -> None:
    solar = real_requirements(1405737714)[0]  # the Solar sibling of the same family

    assert (solar.source_kind, solar.element, solar.completion_action) == (
        "ability",
        "Solar",
        "final_blow",
    )


@needs_manifest
def test_real_weapon_final_blow_with_an_element_and_a_consistent_objective_label() -> None:
    requirements = real_requirements(2943701307)  # "As a fireteam, ... Solar weapon final blows"

    description, label = requirements[0], requirements[1]
    assert (description.source_kind, description.element, description.completion_action) == (
        "weapon",
        "Solar",
        "final_blow",
    )
    assert label.source.definition == OBJECTIVE and label.element == "Solar"
    assert requirements_consistency(requirements) == "consistent"


@needs_manifest
def test_real_elemental_damage_bounty_keeps_bonus_progress_as_raw_text() -> None:
    requirement = real_requirements(2702385474)[0]

    assert (requirement.source_kind, requirement.element) == ("unknown", "Arc")
    assert requirement.completion_action == "defeat"
    assert requirement.bonus_conditions and "bonus progress" in requirement.bonus_conditions[0]


@needs_manifest
def test_real_generic_defeat_bounty_has_no_source_element_or_final_blow() -> None:
    requirement = real_requirements(2702385477)[0]

    assert requirement.raw_requirement == "Defeat Guardians."
    assert (requirement.source_kind, requirement.element) == ("unknown", "unknown")
    assert requirement.completion_action == "defeat" and not is_restrictive(requirement)


@needs_manifest
def test_real_element_lists_are_ambiguous() -> None:
    requirement = real_requirements(2205125593)[0]

    assert requirement.parse_status == "ambiguous" and requirement.element == "unknown"
    assert requirement.completion_action == "final_blow"


@needs_manifest
def test_real_corpus_invariants_hold_across_every_quest_bounty_and_order_description() -> None:
    assert REAL is not None
    parsed = 0
    with sqlite3.connect(str(REAL)) as connection:
        rows = connection.execute(
            "SELECT json FROM DestinyInventoryItemDefinition WHERE json LIKE '%objectiveHashes%'"
        ).fetchall()
    for (raw,) in rows:
        definition = json.loads(raw)
        if not definition.get("objectives", {}).get("objectiveHashes"):
            continue
        for requirement in item_description_requirements(definition["hash"], definition):
            parsed += 1
            raw_text = requirement.raw_requirement.casefold()
            assert (
                requirement.raw_requirement.strip()
                and requirement.source.hash == definition["hash"]
            )
            if requirement.completion_action == "final_blow":
                assert "final blow" in raw_text
            if requirement.parse_status == "verified" and requirement.source_kind == "ability":
                assert " weapon" not in raw_text
            if requirement.parse_status == "verified" and requirement.source_kind == "weapon":
                assert "abilit" not in raw_text and "super" not in raw_text
            if requirement.element != "unknown":
                assert requirement.element.casefold() in raw_text
            if requirement.parse_status == "verified" and requirement.completion_action == "defeat":
                assert "final blow" not in raw_text
    assert parsed > 500
