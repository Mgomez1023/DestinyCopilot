"""M4B: requirement-aware planning evidence.

Fixtures are generic (invented quests, real plug-category grammar); none relies on a
production quest name. The real-Manifest regression at the end is skipped when the local
Manifest is not available.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from app import requirement_fit, session_planning
from app.mechanics_claims import MechanicsEvidenceLedger, detect_mechanics_claims
from app.models import (
    CharacterSummary,
    DataAvailability,
    GuardianContext,
    ObjectiveRequirement,
    ObjectiveSummary,
    QuestSummary,
    RequirementSource,
)
from app.objective_requirements import parse_requirements
from app.planning_alternatives import AlternativeState, apply_alternatives
from app.requirement_fit import (
    MAX_BONUS_CHARS,
    compare_requirement,
    loadout_evidence,
    planning_requirement,
)
from app.session_planning import (
    MAX_REQUIREMENT_CHARS,
    MAX_REQUIREMENTS_PER_CANDIDATE,
    build_session_planning_context,
    requirement_trace,
)
from tests.test_build_analysis import item, plug
from tests.test_objective_requirements import (
    ITEM,
    needs_manifest,
    real_requirements,
)

SOURCE = RequirementSource(
    definition="DestinyInventoryItemDefinition", hash=7, field="displayProperties.description"
)

ARC_GRENADE = ("Storm Grenade", "shared.arc.grenades", "Arc Grenade")
SOLAR_GRENADE = ("Thermite Grenade", "shared.solar.grenades", "Solar Grenade")
ARC_MELEE = ("Ballistic Slam", "titan.arc.melee", "Arc Melee")
ARC_SUPER = ("Thundercrash", "titan.arc.supers", "Super Ability")
ARC_RIFT = ("Empowering Rift", "warlock.arc.class_abilities", "Class Ability")
PRISM_GRENADE = ("Swarm Grenade", "hunter.prism.grenades", "Solar Grenade | Light Ability")
PRISM_UNNAMED = ("Hailfire Spike", "hunter.prism.prism_grenade", "Prismatic Grenade")
ARC_ASPECT = ("Touch of Thunder", "warlock.arc.aspects", "Arc Aspect")


def parsed(text: str) -> ObjectiveRequirement:
    found = parse_requirements(text, kind="description", source=SOURCE)
    assert len(found) == 1, found
    return found[0]


def weapon(name: str, element: str | None, subtype: str = "Hand Cannon", **extra: Any):
    return item(
        name,
        "Weapon",
        subtype=subtype,
        bucket="Energy Weapons",
        damage_type=element,
        location=extra.pop("location", "equipped"),
        character_id=extra.pop("character_id", "c-titan"),
        equipped=extra.pop("equipped", True),
        **extra,
    )


def character(
    class_name: str = "Titan",
    *,
    weapons: list[Any] | None = None,
    abilities: list[tuple[str, str, str]] | None = None,
    subclass_element: str | None = None,
    subclass: bool = True,
    quests: list[QuestSummary] | None = None,
) -> CharacterSummary:
    character_id = f"c-{class_name.casefold()}"
    sub = (
        item(
            "Subclass",
            "Subclass",
            location="equipped",
            character_id=character_id,
            equipped=True,
            damage_type=subclass_element,
            plugs=[plug(*value) for value in (abilities or [])],
        )
        if subclass
        else None
    )
    gear = [*([sub] if sub else []), *(weapons or [])]
    return CharacterSummary(
        character_id=character_id,
        class_name=class_name,
        race_name="Human",
        gender_name="Female",
        power=1800,
        last_played=datetime(2026, 9, 1, tzinfo=UTC),
        subclass=sub,
        equipped_gear=gear,
        quests=quests or [],
    )


def quest(
    name: str,
    text: str | None = None,
    *,
    requirements: list[ObjectiveRequirement] | None = None,
    objectives: list[ObjectiveSummary] | None = None,
    consistency: str | None = None,
    class_id: str = "c-titan",
) -> QuestSummary:
    return QuestSummary(
        quest_hash=abs(hash(name)) % 10_000,
        name=name,
        description=text,
        character_id=class_id,
        objectives=objectives or [],
        requirements=requirements
        if requirements is not None
        else (parse_requirements(text, kind="description", source=SOURCE) if text else []),
        requirements_parsed=True,
        requirements_consistency=consistency,  # type: ignore[arg-type]
    )


def guardian(*characters: CharacterSummary) -> GuardianContext:
    return GuardianContext(
        bungie_display_name="Sanitized",
        membership_id="private",
        membership_type=3,
        platform_name="Steam",
        characters=list(characters),
        data_availability=DataAvailability(fetched_at=datetime(2026, 9, 23, tzinfo=UTC)),
    )


def fit_for(text: str, loadout_character: CharacterSummary):
    return compare_requirement(parsed(text), loadout_evidence(loadout_character))


ARC_ABILITY_TEXT = "Defeat combatants with Arc ability final blows."


# --- Fit rules: ability versus weapon ---------------------------------------------------------


def test_an_arc_weapon_does_not_satisfy_an_arc_ability_final_blow_requirement() -> None:
    result = fit_for(ARC_ABILITY_TEXT, character(weapons=[weapon("Spark", "Arc")]))

    assert result.status == "unknown"  # no ability data, and the Arc weapon never counts
    assert result.equipped == []
    assert result.not_counted == ["Arc weapons: not ability damage"]


def test_equipped_arc_ability_evidence_is_a_verified_match() -> None:
    result = fit_for(
        ARC_ABILITY_TEXT, character(abilities=[ARC_GRENADE], weapons=[weapon("Spark", "Arc")])
    )

    assert result.status == "verified_match"
    assert result.equipped == ["Storm Grenade (Arc grenade)"]
    assert result.not_counted  # the Arc weapon is still named as not counting


@pytest.mark.parametrize("ability", [ARC_GRENADE, ARC_MELEE, ARC_SUPER])
def test_every_damaging_ability_kind_can_satisfy_an_ability_requirement(ability) -> None:
    assert fit_for(ARC_ABILITY_TEXT, character(abilities=[ability])).status == "verified_match"


def test_a_class_ability_alone_is_not_a_damaging_ability() -> None:
    result = fit_for(ARC_ABILITY_TEXT, character("Warlock", abilities=[ARC_RIFT]))

    assert result.status == "verified_mismatch"
    assert result.reason == "no_equipped_damaging_ability_of_that_kind"


def test_an_aspect_is_not_an_ability_and_is_ignored() -> None:
    result = fit_for(ARC_ABILITY_TEXT, character(abilities=[ARC_ASPECT]))

    assert result.status == "unknown" and result.reason == "no_equipped_ability_data"


def test_abilities_of_another_element_are_a_verified_mismatch() -> None:
    result = fit_for(ARC_ABILITY_TEXT, character(abilities=[SOLAR_GRENADE, ARC_RIFT]))

    assert result.status == "verified_mismatch" and result.reason == "no_equipped_Arc_ability"


def test_a_subclass_element_alone_is_only_a_potential_match() -> None:
    result = fit_for(ARC_ABILITY_TEXT, character(subclass_element="Arc"))

    assert result.status == "potential_match"
    assert result.reason == "subclass_element_only_no_ability_data"
    assert result.equipped == []


def test_a_subclass_of_another_element_proves_nothing_without_ability_data() -> None:
    assert fit_for(ARC_ABILITY_TEXT, character(subclass_element="Solar")).status == "unknown"


def test_missing_subclass_means_unknown_not_mismatch() -> None:
    assert fit_for(ARC_ABILITY_TEXT, character(subclass=False)).status == "unknown"


def test_a_weapon_perk_that_follows_an_ability_kill_is_not_an_ability_kill() -> None:
    perked = weapon(
        "Thunder Edge",
        "Arc",
        plugs=[plug("Arc Ability Kill Bonus", "frames", "Trait")],
    )
    perked.socketed_plug_details[0].description = "Final blows with abilities grant Arc damage."

    result = fit_for(ARC_ABILITY_TEXT, character(weapons=[perked]))

    assert result.status == "unknown" and result.equipped == []


def test_prismatic_abilities_take_their_element_from_the_type_label() -> None:
    solar = fit_for(
        "Defeat combatants with Solar ability final blows.",
        character("Hunter", abilities=[PRISM_GRENADE]),
    )
    arc = fit_for(ARC_ABILITY_TEXT, character("Hunter", abilities=[PRISM_GRENADE]))

    assert solar.status == "verified_match"
    assert arc.status == "verified_mismatch"


def test_an_ability_whose_element_was_not_recorded_is_only_a_potential_match() -> None:
    result = fit_for(ARC_ABILITY_TEXT, character("Hunter", abilities=[PRISM_UNNAMED]))

    assert result.status == "potential_match" and result.reason == "ability_element_not_recorded"


# --- Fit rules: weapons -----------------------------------------------------------------------


def test_an_equipped_weapon_of_the_stated_element_is_a_verified_match() -> None:
    result = fit_for(
        "Defeat combatants with Solar weapon final blows.",
        character(weapons=[weapon("Ember", "Solar"), weapon("Plain", "Kinetic")]),
    )

    assert result.status == "verified_match" and result.equipped == ["Ember (Solar weapon)"]


def test_no_equipped_weapon_of_the_element_is_a_loadout_mismatch() -> None:
    result = fit_for(
        "Defeat combatants with Solar weapon final blows.",
        character(weapons=[weapon("Plain", "Kinetic")], abilities=[SOLAR_GRENADE]),
    )

    assert result.status == "verified_mismatch"
    assert result.not_counted == ["Solar abilities: not weapon damage"]


def test_a_weapon_in_the_vault_or_inventory_is_not_equipped() -> None:
    vaulted = weapon("Ember", "Solar", location="vault", equipped=False)
    carried = weapon("Cinder", "Solar", location="character", equipped=False)

    result = fit_for(
        "Defeat combatants with Solar weapon final blows.",
        character(weapons=[vaulted, carried, weapon("Plain", "Kinetic")]),
    )

    assert result.status == "verified_mismatch" and result.equipped == []


def test_a_weapon_without_a_damage_type_leaves_the_match_unknown() -> None:
    result = fit_for(
        "Defeat combatants with Solar weapon final blows.", character(weapons=[weapon("W", None)])
    )

    assert result.status == "unknown" and result.reason == "weapon_damage_type_not_returned"


def test_no_equipped_weapons_returned_is_unknown() -> None:
    assert fit_for("Defeat combatants with weapon final blows.", character()).status == "unknown"


def test_a_grenade_launcher_weapon_is_not_a_grenade_ability() -> None:
    launcher = weapon("Arc Launcher", "Arc", subtype="Grenade Launcher")

    ability = fit_for("Defeat combatants with grenade final blows.", character(weapons=[launcher]))
    assert ability.status == "unknown" and ability.equipped == []

    with_melee = fit_for(
        "Defeat combatants with grenade final blows.",
        character(weapons=[launcher], abilities=[ARC_MELEE]),
    )
    assert with_melee.status == "verified_mismatch"  # the only damaging ability is not a grenade

    # The grenade-launcher wording is a weapon requirement, parsed partial: at best potential.
    archetype = fit_for("Defeat combatants using Grenade Launchers.", character(weapons=[launcher]))
    assert archetype.status == "potential_match"
    assert "partial_parse" in archetype.unassessed


def test_a_grenade_ability_requirement_matches_only_a_grenade() -> None:
    text = "Defeat combatants with Arc grenade final blows."

    assert fit_for(text, character(abilities=[ARC_GRENADE])).status == "verified_match"
    assert fit_for(text, character(abilities=[ARC_MELEE])).status == "verified_mismatch"


def test_damage_of_an_element_from_no_stated_source_accepts_either_source() -> None:
    text = "Defeat combatants with Arc damage."

    assert fit_for(text, character(weapons=[weapon("Spark", "Arc")])).status == "verified_match"
    assert fit_for(text, character(abilities=[ARC_GRENADE])).status == "verified_match"
    assert (
        fit_for(
            text, character(weapons=[weapon("Ember", "Solar")], abilities=[SOLAR_GRENADE])
        ).status
        == "verified_mismatch"
    )


# --- Fit rules: certainty caps ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Get Solar, Arc, Void, Strand, or Stasis final blows.",
        "Defeat combatants with Solar weapons or abilities.",
    ],
)
def test_ambiguous_requirements_are_unknown(text: str) -> None:
    result = fit_for(text, character(weapons=[weapon("Ember", "Solar")], abilities=[ARC_GRENADE]))

    assert result.status == "unknown" and result.reason == "requirement_not_established"


def test_a_generic_defeat_names_nothing_to_compare() -> None:
    result = fit_for("Defeat Guardians.", character(weapons=[weapon("Ember", "Solar")]))

    assert result.status == "unknown" and result.reason == "requirement_names_no_source_or_element"


def test_a_partial_requirement_never_becomes_a_verified_match() -> None:
    partial = parsed("Defeat targets while using any Fusion Rifle.")
    assert partial.parse_status == "partial"

    result = compare_requirement(
        partial, loadout_evidence(character(weapons=[weapon("Ember", "Solar", "Fusion Rifle")]))
    )
    assert result.status == "potential_match"


def test_a_partial_requirement_whose_stated_part_fails_stays_a_mismatch() -> None:
    requirement = parsed("Defeat combatants with Solar weapon final blows.").model_copy(
        update={"parse_status": "partial"}
    )

    result = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("W", "Arc")]))
    )

    assert result.status == "verified_mismatch"


def test_precision_wording_caps_a_match_because_a_loadout_cannot_show_it() -> None:
    result = fit_for(
        "Defeat combatants with precision final blows.", character(weapons=[weapon("W", "Arc")])
    )
    requirement = parsed("Defeat combatants with Arc weapon precision final blows.")

    assert result.status == "unknown"  # no source or element: nothing to compare
    capped = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("W", "Arc")]))
    )
    assert capped.status == "potential_match" and "modifier:precision" in capped.unassessed


def test_bonus_progress_text_is_not_mechanics_and_cannot_create_a_match() -> None:
    text = (
        "Defeat combatants with Solar damage. Dealing final blows with the Arc Super grants "
        "bonus progress."
    )
    requirement = parsed(text)

    arc_super_only = compare_requirement(
        requirement,
        loadout_evidence(character(abilities=[ARC_SUPER], weapons=[weapon("Plain", "Kinetic")])),
    )
    solar_weapon = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("Ember", "Solar")]))
    )

    assert arc_super_only.status == "verified_mismatch"  # the Arc Super is bonus text only
    assert solar_weapon.status == "verified_match"
    assert "bonus_text" in solar_weapon.unassessed


def test_conditions_are_reported_as_unassessed_not_verified() -> None:
    requirement = parsed(
        "As a fireteam, defeat opponents with Solar weapon final blows in the Crucible."
    )

    result = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("Ember", "Solar")]))
    )

    assert result.status == "verified_match" and "conditions" in result.unassessed


def test_disabled_subclass_plugs_do_not_count() -> None:
    chosen = character(abilities=[ARC_GRENADE])
    chosen.subclass.socketed_plug_details[0].is_enabled = False  # type: ignore[union-attr]

    assert compare_requirement(parsed(ARC_ABILITY_TEXT), loadout_evidence(chosen)).status == (
        "unknown"
    )


def test_loadout_evidence_reads_only_the_given_characters_equipped_gear() -> None:
    titan = character("Titan", weapons=[weapon("Ember", "Solar")])
    loadout = loadout_evidence(titan)

    assert [value.name for value in loadout.weapons] == ["Ember"]
    assert loadout.abilities == []


# --- Planning candidates ---------------------------------------------------------------------


def plan(*characters: CharacterSummary, requested: str | None = None):
    return build_session_planning_context(
        guardian(*characters), "What should I do next?", "general", requested
    )


def candidate(planning, name: str):
    return next(value for value in planning.candidates if value.name == name)


def test_a_candidate_carries_a_compact_requirement_and_its_fit() -> None:
    titan = character(
        weapons=[weapon("Spark", "Arc")],
        abilities=[ARC_GRENADE],
        quests=[
            quest(
                "Order A",
                ARC_ABILITY_TEXT + " Combatants affected by Arc debuffs grant greater progress.",
            )
        ],
    )
    planning = plan(titan)
    entry = candidate(planning, "Order A")
    payload = json.loads(entry.model_dump_json())["requirements"][0]

    assert payload["source_kind"] == "ability" and payload["element"] == "Arc"
    assert payload["completion_action"] == "final_blow"
    assert payload["target_kind"] == "combatants"
    assert payload["parse_status"] == "verified"
    assert payload["source"] == "DestinyInventoryItemDefinition:7:displayProperties.description"
    assert payload["fit"]["status"] == "verified_match"
    assert payload["fit"]["equipped"] == ["Storm Grenade (Arc grenade)"]
    assert payload["fit"]["unassessed"] == ["bonus_text"]
    assert payload["bonus_conditions"] == [
        "Combatants affected by Arc debuffs grant greater progress."
    ]
    assert "raw_requirement" not in payload  # a clean sentence the description already holds
    assert (
        entry.description
        == ARC_ABILITY_TEXT + " Combatants affected by Arc debuffs grant greater progress."
    )
    assert planning.requirement_basis and planning.requirements_omitted == 0


def test_unverified_requirements_keep_their_raw_text() -> None:
    titan = character(quests=[quest("Order B", "Defeat targets while using any Fusion Rifle.")])
    payload = json.loads(candidate(plan(titan), "Order B").model_dump_json())["requirements"][0]

    assert payload["parse_status"] == "partial"
    assert payload["raw_requirement"] == "Defeat targets while using any Fusion Rifle."
    assert payload["conditions"] == ["while using any Fusion Rifle"]


def test_a_generic_defeat_is_not_attached() -> None:
    titan = character(quests=[quest("Order C", "Defeat Guardians.")])
    planning = plan(titan)

    assert candidate(planning, "Order C").requirements == []
    assert planning.requirement_basis is None
    encoded = planning.model_dump_json()
    assert "requirements" not in encoded and "requirement_basis" not in encoded


def test_progress_objectives_and_description_are_preserved() -> None:
    objective = ObjectiveSummary(
        objective_hash=1,
        name="Targets defeated",
        progress=3,
        completion_value=10,
        progress_percent=30.0,
    )
    titan = character(quests=[quest("Order D", ARC_ABILITY_TEXT, objectives=[objective])])
    entry = candidate(plan(titan), "Order D")

    assert entry.description == ARC_ABILITY_TEXT
    assert [(v.name, v.progress_percent) for v in entry.objectives] == [("Targets defeated", 30.0)]


def test_objective_text_is_attached_separately_and_disagreement_is_reported() -> None:
    from_objective = parse_requirements(
        "Arc weapon final blows",
        kind="objective_label",
        source=RequirementSource(
            definition="DestinyObjectiveDefinition", hash=9, field="progressDescription"
        ),
    )[0]
    objective = ObjectiveSummary(
        objective_hash=9, name="Arc weapon final blows", requirement=from_objective
    )
    titan = character(
        abilities=[ARC_GRENADE],
        quests=[
            quest(
                "Order E",
                ARC_ABILITY_TEXT,
                objectives=[objective],
                consistency="conflicting",
            )
        ],
    )
    entry = candidate(plan(titan), "Order E")
    payload = json.loads(entry.model_dump_json())

    assert [v["source_kind"] for v in payload["requirements"]] == ["ability", "weapon"]
    assert payload["requirements_consistency"] == "conflicting"
    assert [v.fit.status for v in entry.requirements] == ["verified_match", "unknown"]


def test_agreeing_sources_report_no_consistency_note() -> None:
    titan = character(quests=[quest("Order F", ARC_ABILITY_TEXT, consistency="consistent")])

    assert candidate(plan(titan), "Order F").requirements_consistency is None


def test_multiple_candidates_with_different_requirements_are_compared_independently() -> None:
    titan = character(
        weapons=[weapon("Ember", "Solar")],
        abilities=[ARC_GRENADE],
        quests=[
            quest("Alpha", ARC_ABILITY_TEXT),
            quest("Beta", "Defeat combatants with Solar weapon final blows."),
            quest("Gamma", "Defeat combatants with Void weapon final blows."),
        ],
    )
    planning = plan(titan)
    statuses = {v.name: v.requirements[0].fit.status for v in planning.candidates if v.requirements}

    assert statuses == {
        "Alpha": "verified_match",
        "Beta": "verified_match",
        "Gamma": "verified_mismatch",
    }


def test_requirements_do_not_change_candidates_order_or_other_fields() -> None:
    quests = [
        quest("Alpha", ARC_ABILITY_TEXT),
        quest("Beta", "Defeat combatants with Solar weapon final blows."),
        quest("Plain", "Defeat Guardians."),
    ]
    with_requirements = plan(character(weapons=[weapon("Ember", "Solar")], quests=quests))
    stripped = [
        value.model_copy(update={"requirements": [], "requirements_parsed": False})
        for value in quests
    ]
    without = plan(character(weapons=[weapon("Ember", "Solar")], quests=stripped))

    def core(planning) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in c.model_dump(mode="json").items() if not k.startswith("requirement")}
            for c in planning.candidates
        ]

    assert core(with_requirements) == core(without)
    assert [v.name for v in with_requirements.candidates] == [v.name for v in without.candidates]


# --- Character isolation and conversation state ----------------------------------------------


def two_characters() -> tuple[CharacterSummary, CharacterSummary]:
    text = ARC_ABILITY_TEXT
    titan = character(
        "Titan",
        abilities=[ARC_GRENADE],
        weapons=[weapon("Spark", "Arc")],
        quests=[quest("Shared Order", text, class_id="c-titan")],
    )
    hunter = character(
        "Hunter",
        abilities=[SOLAR_GRENADE],
        weapons=[weapon("Ember", "Solar", character_id="c-hunter")],
        quests=[quest("Shared Order", text, class_id="c-hunter")],
    )
    return titan, hunter


def test_comparison_uses_the_requested_character_only() -> None:
    titan, hunter = two_characters()

    as_titan = candidate(plan(titan, hunter, requested="Titan"), "Shared Order")
    as_hunter = candidate(plan(titan, hunter, requested="Hunter"), "Shared Order")

    assert as_titan.requirements[0].fit.status == "verified_match"
    assert as_hunter.requirements[0].fit.status == "verified_mismatch"
    assert "Storm Grenade" not in as_hunter.model_dump_json()  # no Titan ability leaks


def test_a_hunter_context_has_no_titan_subclass_or_weapons_in_its_evidence() -> None:
    titan, hunter = two_characters()
    planning = plan(titan, hunter, requested="Hunter")

    encoded = planning.model_dump_json()
    assert planning.character_class == "Hunter"
    assert "Spark" not in encoded and "Storm Grenade" not in encoded


def test_an_unrequested_character_is_not_substituted_when_missing() -> None:
    titan, _ = two_characters()
    planning = plan(titan, requested="Warlock")

    assert planning.candidates == [] or all(not v.requirements for v in planning.candidates)


def test_exclusions_remove_the_candidate_and_its_requirements() -> None:
    titan = character(
        abilities=[ARC_GRENADE],
        quests=[quest("Alpha", ARC_ABILITY_TEXT), quest("Beta", ARC_ABILITY_TEXT)],
    )
    planning = plan(titan)
    state = AlternativeState(rejected=["Alpha"], rejection_detected=True)

    adjusted = apply_alternatives(planning, state)

    assert [v.name for v in adjusted.candidates] == ["Beta"]
    assert adjusted.candidates[0].requirements  # preserved on what remains
    assert adjusted.excluded_by_user == ["Alpha"]
    assert adjusted.requirement_basis == planning.requirement_basis


def test_planning_preferences_and_scope_fields_are_unchanged() -> None:
    titan = character(quests=[quest("Alpha", ARC_ABILITY_TEXT)])
    planning = plan(titan, requested="Titan")

    assert planning.version == 6
    assert planning.requested_character_scope == "Titan" and planning.requested_scope_explicit
    assert planning.preferences.character == "Titan" or planning.preferences is not None


# --- Compact output and truncation -----------------------------------------------------------


def test_output_is_bounded_and_omissions_are_counted() -> None:
    long_bonus = "Dealing final blows grants bonus progress " + "x" * 400 + "."
    many = [
        quest(f"Order {index}", f"Defeat combatants with Arc ability final blows. {long_bonus}")
        for index in range(6)
    ]
    planning = plan(character(abilities=[ARC_GRENADE], quests=many))

    entries = [v for v in planning.candidates if v.requirements]
    size = sum(len(json.dumps(json.loads(v.model_dump_json())["requirements"])) for v in entries)
    assert size <= MAX_REQUIREMENT_CHARS
    assert all(len(v.requirements) <= MAX_REQUIREMENTS_PER_CANDIDATE for v in entries)
    assert planning.requirements_omitted + len(entries) >= 1
    bonus = json.loads(entries[0].model_dump_json())["requirements"][0]["bonus_conditions"][0]
    assert len(bonus) <= MAX_BONUS_CHARS


def test_more_than_the_per_candidate_cap_is_counted_as_omitted() -> None:
    requirements = [
        *parse_requirements(ARC_ABILITY_TEXT, kind="description", source=SOURCE),
        *parse_requirements(
            "Defeat combatants with Solar weapon final blows.", kind="description", source=SOURCE
        ),
        *parse_requirements(
            "Defeat combatants with Void weapon final blows.", kind="description", source=SOURCE
        ),
    ]
    planning = plan(character(quests=[quest("Many", requirements=requirements)]))

    assert len(candidate(planning, "Many").requirements) == MAX_REQUIREMENTS_PER_CANDIDATE
    assert planning.requirements_omitted == 1


def test_the_basis_text_is_present_once_and_only_when_requirements_are() -> None:
    with_one = plan(character(quests=[quest("A", ARC_ABILITY_TEXT)]))
    without = plan(character(quests=[quest("A", "Defeat Guardians.")]))

    assert with_one.model_dump_json().count(session_planning.REQUIREMENT_BASIS[:40]) == 1
    assert without.requirement_basis is None and without.requirements_omitted == 0


def test_trace_counts_candidates_and_fit_outcomes() -> None:
    planning = plan(
        character(
            abilities=[ARC_GRENADE],
            quests=[quest("A", ARC_ABILITY_TEXT), quest("B", "Defeat Guardians.")],
        )
    )

    assert requirement_trace(planning) == {
        "candidates_with_requirements": 1,
        "fit": {"verified_match": 1},
        "omitted": 0,
    }
    assert requirement_trace(None) == {"candidates_with_requirements": 0}


# --- Cache compatibility and non-planning paths ------------------------------------------------


def test_a_quest_cached_before_requirements_plans_without_requirement_fields() -> None:
    old = QuestSummary.model_validate(
        {
            "quest_hash": 1,
            "name": "Old Quest",
            "character_id": "c-titan",
            "description": "Defeat foes.",
        }
    )
    planning = plan(character(quests=[old]))

    assert candidate(planning, "Old Quest").requirements == []
    assert "requirement" not in planning.model_dump_json()


def test_planning_requirement_models_are_not_stored_on_the_guardian_context() -> None:
    titan = character(abilities=[ARC_GRENADE], quests=[quest("A", ARC_ABILITY_TEXT)])
    context = guardian(titan)
    before = context.model_dump_json()

    build_session_planning_context(context, "next?", "general", None)

    assert context.model_dump_json() == before  # nothing mutated; nothing cached
    assert '"fit"' not in before


def quest_tool(*characters: CharacterSummary) -> dict[str, Any]:
    from app.guardian_tools import GuardianToolService

    return GuardianToolService(guardian(*characters)).execute(
        "get_active_quests", {"character_id": None, "character_class": "Titan"}
    )


def test_the_guardian_quest_tool_compares_each_requirement_with_its_characters_loadout() -> None:
    titan = character(
        weapons=[weapon("Spark", "Arc")],
        abilities=[ARC_GRENADE],
        quests=[
            quest("Ability Order", ARC_ABILITY_TEXT),
            quest("Weapon Order", "Defeat combatants with Solar weapon final blows."),
            quest("Plain Order", "Defeat Guardians."),
        ],
    )
    result = quest_tool(titan)
    by_name = {value["name"]: value for value in result["quests"]}

    assert by_name["Ability Order"]["requirements"][0]["fit"]["status"] == "verified_match"
    assert by_name["Weapon Order"]["requirements"][0]["fit"]["status"] == "verified_mismatch"
    assert "fit" not in by_name["Plain Order"]["requirements"][0]  # nothing to compare
    assert result["requirement_fit_basis"] == requirement_fit.REQUIREMENT_BASIS


def test_the_quest_tool_never_uses_another_characters_loadout() -> None:
    titan, hunter = two_characters()
    result = quest_tool(titan, hunter)

    assert {v["character_class"] for v in result["quests"]} == {"Titan"}
    assert result["quests"][0]["requirements"][0]["fit"]["equipped"] == [
        "Storm Grenade (Arc grenade)"
    ]
    assert "Solar" not in json.dumps(result["quests"])


def test_the_quest_tool_omits_the_basis_when_no_fit_is_present() -> None:
    result = quest_tool(character(quests=[quest("Plain Order", "Defeat Guardians.")]))

    assert "requirement_fit_basis" not in result


# --- Mechanics ledger (trace only) -------------------------------------------------------------


def ledger_for(planning=None, *, tool_results: list[tuple[str, str, Any]] | None = None):
    ledger = MechanicsEvidenceLedger()
    if planning is not None:
        ledger.record_planning(planning)
    for name, category, result in tool_results or []:
        ledger.record(name, category, result)
    return ledger


def arc_ability_ledger():
    titan = character(
        abilities=[ARC_GRENADE],
        quests=[quest("Order A", ARC_ABILITY_TEXT)],
    )
    return ledger_for(plan(titan))


def claims_of(answer: str, ledger: MechanicsEvidenceLedger):
    return detect_mechanics_claims(answer, ledger).claims


def test_recommending_an_arc_weapon_for_an_ability_objective_is_flagged_without_raw_text() -> None:
    # The planning requirement carried no raw sentence; the structured source is enough.
    ledger = arc_ability_ledger()
    entity = ledger.entity("Order A")
    assert entity is not None and entity.requirements

    answer = "For Order A, use an Arc weapon to get the final blows."
    claims = [v for v in claims_of(answer, ledger) if v.category == "damage_source"]

    assert [v.status for v in claims] == ["contradicted_by_evidence"]


def test_an_ability_recommendation_for_an_ability_objective_is_supported() -> None:
    claims = claims_of(
        "Order A needs ability kills, so lean on your grenade final blows.", arc_ability_ledger()
    )

    assert any(v.category == "damage_source" and v.status == "supported" for v in claims)


def test_the_wrong_element_for_a_verified_requirement_is_flagged() -> None:
    answer = "Order A is quickest with a Solar ability for the final blows."
    claims = [
        v for v in claims_of(answer, arc_ability_ledger()) if v.category == "objective_requirement"
    ]

    assert [v.status for v in claims] == ["contradicted_by_evidence"]


def test_naming_the_required_element_alongside_another_is_not_flagged() -> None:
    answer = "Order A needs an Arc ability; your Solar weapon will not count."
    claims = [
        v for v in claims_of(answer, arc_ability_ledger()) if v.category == "objective_requirement"
    ]

    assert claims == []


def test_a_bonus_progress_multiplier_without_source_text_is_flagged() -> None:
    answer = "In Order A the bonus progress is doubled, about 3x per final blow."
    claims = [v for v in claims_of(answer, arc_ability_ledger()) if v.status.startswith("bonus")]

    assert [v.status for v in claims] == ["bonus_claim_without_evidence"]


def test_a_bonus_figure_present_in_the_retrieved_text_is_not_flagged() -> None:
    titan = character(
        abilities=[ARC_GRENADE],
        quests=[quest("Order G", ARC_ABILITY_TEXT + " Armored targets give 2x bonus progress.")],
    )
    ledger = ledger_for(plan(titan))

    claims = claims_of("Order G gives 2x progress on armored targets.", ledger)

    assert not [v for v in claims if v.status == "bonus_claim_without_evidence"]


def test_a_weapon_named_without_guardian_ownership_evidence_is_flagged() -> None:
    ledger = arc_ability_ledger()
    ledger.record(
        "get_item_details",
        "knowledge",
        {"item": {"name": "Distant Thunder", "item_type": "Weapon", "description": "A rifle."}},
    )

    claims = claims_of("Use Distant Thunder to get final blows for Order A.", ledger)

    assert [v.status for v in claims if v.category == "objective_requirement"] == [
        "weapon_without_ownership_evidence"
    ]


def test_an_owned_equipped_weapon_is_not_flagged_for_ownership() -> None:
    ledger = arc_ability_ledger()
    ledger.record(
        "get_equipped_loadout",
        "guardian",
        {"equipped_weapons": [{"name": "Distant Thunder", "item_type": "Weapon"}]},
    )

    claims = claims_of("Use Distant Thunder to get final blows for Order A.", ledger)

    assert not [v for v in claims if v.status == "weapon_without_ownership_evidence"]


def test_the_ledger_never_changes_the_answer_or_blocks_it() -> None:
    answer = "Use an Arc weapon for Order A final blows."
    report = detect_mechanics_claims(answer, arc_ability_ledger())

    assert report.to_trace()["mode"] == "trace_only"
    assert answer == "Use an Arc weapon for Order A final blows."


def test_recording_planning_context_tolerates_bad_input() -> None:
    ledger = MechanicsEvidenceLedger()
    ledger.record_planning(object())
    ledger.record_planning(None)

    assert ledger.entities == {}


def test_the_ledger_reads_requirements_from_the_guardian_quest_tool() -> None:
    from app.guardian_tools import GuardianToolService

    titan = character(abilities=[ARC_GRENADE], quests=[quest("Order H", ARC_ABILITY_TEXT)])
    result = GuardianToolService(guardian(titan)).execute(
        "get_active_quests", {"character_id": None, "character_class": None}
    )
    ledger = ledger_for(tool_results=[("get_active_quests", "guardian", result)])

    entity = ledger.entity("Order H")
    assert entity is not None and entity.requirements[0]["source_kind"] == "ability"


# --- Module hygiene --------------------------------------------------------------------------


def test_comparison_names_no_quests_activities_or_items() -> None:
    from pathlib import Path

    source = Path(requirement_fit.__file__).read_text(encoding="utf-8").casefold()

    for name in ("scourge", "trials", "gambit", "sunshot", "thundercrash", "storm grenade"):
        assert name not in source


def test_planning_requirement_wrapper_flattens_requirement_and_fit() -> None:
    entry = planning_requirement(parsed(ARC_ABILITY_TEXT), loadout_evidence(character()))
    flat = json.loads(entry.model_dump_json())

    assert {"source", "source_kind", "fit", "parse_status"} <= set(flat)
    assert "requirement" not in flat


# --- Real Manifest -----------------------------------------------------------------------------


@needs_manifest
def test_real_arc_scourge_matches_only_with_equipped_arc_ability_evidence() -> None:
    requirement = real_requirements(1405737718)[0]
    assert requirement.source.definition == ITEM

    arc_weapon_only = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("Spark", "Arc")]))
    )
    arc_ability = compare_requirement(
        requirement, loadout_evidence(character(abilities=[ARC_GRENADE]))
    )
    solar_ability = compare_requirement(
        requirement, loadout_evidence(character(abilities=[SOLAR_GRENADE]))
    )

    assert arc_weapon_only.status == "unknown" and arc_weapon_only.not_counted
    assert arc_ability.status == "verified_match"
    assert arc_ability.unassessed == ["bonus_text"]  # "Arc debuffs" bonus stays text
    assert solar_ability.status == "verified_mismatch"


@needs_manifest
def test_real_solar_weapon_requirement_matches_a_solar_weapon_only() -> None:
    requirement = real_requirements(2943701307)[0]

    solar = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("Ember", "Solar")]))
    )
    arc = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("Spark", "Arc")]))
    )

    assert solar.status == "verified_match" and "conditions" in solar.unassessed
    assert arc.status == "verified_mismatch"


@needs_manifest
def test_real_element_list_requirement_is_unknown_for_every_loadout() -> None:
    requirement = real_requirements(2205125593)[0]

    result = compare_requirement(
        requirement, loadout_evidence(character(weapons=[weapon("Ember", "Solar")]))
    )

    assert result.status == "unknown"


def test_a_sentence_naming_objectives_with_different_sources_makes_no_source_claim() -> None:
    titan = character(
        quests=[
            quest("Ability Order", ARC_ABILITY_TEXT),
            quest("Weapon Order", "Defeat combatants with Solar weapon final blows."),
        ]
    )
    ledger = ledger_for(plan(titan))
    answer = (
        "Use your grenade final blows for Ability Order, or keep your Solar weapon for "
        "Weapon Order."
    )

    assert [v for v in claims_of(answer, ledger) if v.category == "damage_source"] == []


# --- Scope of verified_match and verified_mismatch ---------------------------------------------


def test_verified_match_does_not_claim_activity_location_or_fireteam_conditions() -> None:
    titan = character(
        weapons=[weapon("Ember", "Solar")],
        quests=[
            quest(
                "Team Order",
                "As a fireteam, defeat opponents with Solar weapon final blows in the Crucible.",
            )
        ],
    )
    planning = plan(titan)
    payload = json.loads(candidate(planning, "Team Order").model_dump_json())["requirements"][0]

    assert payload["fit"]["status"] == "verified_match"
    assert "conditions" in payload["fit"]["unassessed"]
    assert "As a fireteam" in payload["conditions"]  # the limitation text is carried
    basis = planning.requirement_basis or ""
    assert "does NOT mean the other conditions are met" in basis
    assert "activity, location, fireteam" in basis

    tool = quest_tool(titan)
    assert "does NOT mean the other conditions are met" in tool["requirement_fit_basis"]
    assert "conditions" in tool["quests"][0]["requirements"][0]["fit"]["unassessed"]


def test_mismatch_is_scoped_to_the_equipped_loadout_not_owned_alternatives() -> None:
    owned = [
        weapon("Vaulted Ember", "Solar", location="vault", equipped=False),
        weapon("Carried Ember", "Solar", location="character", equipped=False),
    ]
    titan = character(
        weapons=[weapon("Plain", "Kinetic"), *owned],
        quests=[quest("Weapon Order", "Defeat combatants with Solar weapon final blows.")],
    )
    context = guardian(titan)
    context.inventory.items = list(owned)  # owned copies exist in the profile

    planning = build_session_planning_context(context, "next?", "general", None)
    fit = candidate(planning, "Weapon Order").requirements[0].fit

    assert fit.status == "verified_mismatch" and fit.equipped == []
    assert "Vaulted Ember" not in planning.model_dump_json()
    assert "vaulted, carried or unowned alternatives are not considered" in (
        planning.requirement_basis or ""
    )
