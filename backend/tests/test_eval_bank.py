"""The eval system itself: scenario bank shape, coverage, and fixture validity (CI-safe)."""

from collections import Counter

import pytest

from app.ai import RecommendationService
from app.core_knowledge_provider import CORE_KNOWLEDGE_TOOL_DEFINITIONS
from app.destiny_knowledge import DESTINY_KNOWLEDGE_TOOL_DEFINITIONS
from app.guardian_tools import GuardianToolService
from app.models import GuardianContext
from app.session_planning import build_session_planning_context
from evals.golden import GOLDEN
from evals.model import AXES
from evals.profiles import PROFILE_DESCRIPTIONS, PROFILES, profile_facts
from evals.scenarios import BANK, REPRESENTATIVE, SUBSETS
from tests.test_core_knowledge_integration import ENGINE

GOLDEN_PROMPTS = [
    "What should I do next?",
    "Why should I finish Altars of Sorrow?",
    "Where is Altars of Sorrow?",
    "Look at what I own and recommend a Titan loadout",
    "Only recommend things I own",
    "What does the Rampage perk do?",
    "What's the weakest part of my current build?",
    "I only have 30 minutes. Pick one thing for me.",
    "Actually what if I wanted to play on my Hunter?",
    'What does "Altars" refer to?',
    "Is Altars of Sorrow matchmade?",
    "Can I equip two Exotic armor pieces?",
    "Can I equip an Exotic weapon and Exotic armor together?",
    "Use a HC for this build.",
    "What is a dungeon?",
    "What is the Nightfall this week?",
    "What is a Nightfall?",
]
DEFAULTS = {"character": "unspecified", "social": "unspecified", "account": "general"}


def _tool_names() -> set[str]:
    guardian = {
        value["name"] for value in GuardianToolService(PROFILES["titan_veteran"]()).definitions()
    }
    return (
        guardian
        | {value["name"] for value in DESTINY_KNOWLEDGE_TOOL_DEFINITIONS}
        | {value["name"] for value in CORE_KNOWLEDGE_TOOL_DEFINITIONS}
        | {"get_destiny_guide", "search_destiny_guides"}
    )


def test_golden_set_covers_every_required_prompt_in_order() -> None:
    assert [value.prompt for value in GOLDEN] == GOLDEN_PROMPTS
    assert all(value.golden for value in GOLDEN)


def test_bank_size_ids_and_subsets() -> None:
    assert 80 <= len(BANK) <= 150
    assert len({value.id for value in BANK}) == len(BANK)
    assert 20 <= len(REPRESENTATIVE) <= 30
    assert set(SUBSETS) == {
        "golden",
        "representative",
        "conversations",
        "acceptance",
        "mechanics",
        "playtest",
        "heldout",
        "heldout_retired",
        "full",
    }
    assert len(SUBSETS["acceptance"]) == 7
    assert len(SUBSETS["conversations"]) == 3
    assert all(len(value.turns) >= 3 for value in SUBSETS["conversations"])
    assert SUBSETS["full"] == BANK


def test_every_matrix_axis_value_is_covered() -> None:
    for axis, values in AXES.items():
        default = DEFAULTS.get(axis, "none")
        seen = Counter(value.tags.get(axis, default) for value in BANK)
        assert set(values) <= set(seen), (axis, set(values) - set(seen))
        assert all(key in values for key in seen), (axis, seen)


def test_scenarios_reference_real_profiles_tools_and_knowledge() -> None:
    tools = _tool_names()
    for value in BANK:
        assert value.profile in PROFILES, value.id
        assert set(value.expect.tools_any) <= tools, (value.id, value.expect.tools_any)
        if value.expect.ambiguity:
            assert ENGINE.resolve(value.expect.ambiguity).status == "ambiguous", value.id
        for entity in value.expect.grounded_entities:
            assert ENGINE.resolve(entity).status == "entity", (value.id, entity)
        for entity, _field in value.expect.unknown_fields:
            assert ENGINE.resolve(entity).status == "entity", (value.id, entity)


def test_owned_expectations_are_satisfiable_by_their_profile() -> None:
    for value in BANK:
        if not value.expect.only_owned or not value.expect.mentions_any:
            continue
        owned = profile_facts(PROFILES[value.profile]()).owned_items
        # mentions_any is matched as a word in the answer, so a name prefix suffices.
        assert any(item.startswith(name) for name in value.expect.mentions_any for item in owned), (
            value.id
        )


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profiles_are_valid_and_round_trip(name: str) -> None:
    context = PROFILES[name]()

    assert GuardianContext.model_validate_json(context.model_dump_json()) == context
    assert name in PROFILE_DESCRIPTIONS
    assert context == PROFILES[name](), "profiles must be deterministic"
    facts = profile_facts(context)
    for item, (restriction, _where) in facts.owned_items.items():
        assert restriction in {None, "Any", "Titan", "Hunter", "Warlock"}, item


def test_profiles_cover_required_account_states() -> None:
    facts = {name: profile_facts(factory()) for name, factory in PROFILES.items()}

    assert facts["missing_class"].classes == ("Titan", "Warlock")
    assert facts["new_player"].owned_items and not PROFILES["new_player"]().inventory.items
    assert len(PROFILES["large_vault"]().inventory.items) >= 25
    assert len(facts["near_complete"].near_complete) >= 4
    assert not facts["no_next_step"].near_complete
    assert not any(facts["no_next_step"].active_quests.values())
    titan_armor = {
        name
        for name, (restriction, where) in facts["wrong_class_vault"].owned_items.items()
        if where == "vault"
    }
    assert titan_armor and all(
        facts["wrong_class_vault"].owned_items[name][0] in {"Hunter", "Warlock"}
        for name in titan_armor
    )
    assert PROFILES["stale_manifest"]().data_availability.unavailable_components
    assert PROFILES["returning_player"]().characters[0].last_played.year < 2026


def test_same_prompt_yields_different_planning_context_per_profile() -> None:
    candidate_sets = {
        name: tuple(
            value.name
            for value in build_session_planning_context(
                factory(), "What should I do next?", "general", None
            ).candidates
        )
        for name, factory in PROFILES.items()
    }

    assert len(set(candidate_sets.values())) >= 7
    assert "The Edge of Fate" in candidate_sets["titan_veteran"]
    assert "The Edge of Fate" not in candidate_sets["no_next_step"]


def test_eval_runtime_reuses_the_production_service() -> None:
    from evals.runtime import RecordingRecommendationService

    assert issubclass(RecordingRecommendationService, RecommendationService)
    overridden = {
        name
        for name, value in vars(RecordingRecommendationService).items()
        if not name.startswith("__") and callable(value)
    }
    assert overridden == {"_create_response", "_execute_tool_call"}


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profile_loadouts_obey_the_grounded_exotic_equip_limits(name: str) -> None:
    rule = {fact.p: fact.o for fact in ENGINE._entities["rule.exotic_equip_limits"].facts}  # noqa: SLF001
    for character in PROFILES[name]().characters:
        equipped = [item for item in character.equipped_gear if item.tier == "Exotic"]
        weapons = [item for item in equipped if item.item_type == "Weapon"]
        armor = [item for item in equipped if item.item_type == "Armor"]
        assert len(weapons) <= rule["max_equipped_exotic_weapons"], (name, weapons)
        assert len(armor) <= rule["max_equipped_exotic_armor"], (name, armor)
        assert all(item.class_restriction == character.class_name for item in armor), name


def test_current_build_scenarios_expect_the_deterministically_resolved_character() -> None:
    from app.character_selection import resolve_current_character

    current_scenarios = [
        value for value in BANK if value.id.startswith(("build.current.", "build.using_now"))
    ]

    assert len(current_scenarios) >= 5
    for value in current_scenarios:
        selected = resolve_current_character(PROFILES[value.profile]())
        assert selected is not None and value.expect.character == selected.class_name, value.id
        assert not any(name in value.prompt for name in ("Titan", "Hunter", "Warlock")), value.id


def test_conversation_scenarios_expect_a_character_for_every_turn() -> None:
    from app.character_selection import resolve_current_character

    for value in (*SUBSETS["conversations"], *SUBSETS["acceptance"]):
        assert len(value.expect.turn_characters) == len(value.turns), value.id
        first = value.expect.turn_characters[0]
        current = resolve_current_character(PROFILES[value.profile]())
        explicit_first = any(name in value.turns[0] for name in ("Titan", "Hunter", "Warlock"))
        if not explicit_first:
            assert current is not None and first == current.class_name, value.id
        assert value.expect.turn_characters[-1] == value.expect.character, value.id


def test_playtest_scenarios_are_runnable_and_cover_the_reported_failures() -> None:
    tools = _tool_names()
    playtest = SUBSETS["playtest"]

    assert len(playtest) == 10
    assert len({value.id for value in playtest}) == len(playtest)
    assert not {value.id for value in playtest} & {value.id for value in BANK}
    for value in playtest:
        assert value.profile in PROFILES, value.id
        assert set(value.expect.tools_any) <= tools, value.id
        for axis, tag in value.tags.items():
            assert tag in AXES[axis], (value.id, axis, tag)
    # Goal-respect scenarios name gear the profile really has equipped, never a special case.
    goal = next(value for value in playtest if value.id == "playtest.single_improvement_goal")
    titan = next(c for c in PROFILES[goal.profile]().characters if c.class_name == "Titan")
    assert "Rose" in {item.name for item in titan.equipped_gear}


FRAGMENT_ELEMENTS = {"Ember": "solar", "Echo": "void", "Spark": "arc"}
KINETIC_SLOT_DAMAGE = {"Kinetic", "Stasis", "Strand"}
MOD_SLOTS = {
    "Helmet": "v2_head",
    "Gauntlets": "v2_arms",
    "Chest Armor": "v2_chest",
    "Leg Armor": "v2_legs",
    "Class Armor": "v2_class_item",
}


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profile_gear_is_a_possible_destiny_configuration(name: str) -> None:
    """Weapon slots, damage types, armor slots, mods, and subclass plugs as the game allows."""

    context = PROFILES[name]()
    owned = [*context.inventory.items, *(i for c in context.characters for i in c.equipped_gear)]
    for item in owned:
        if item.item_type == "Weapon":
            kinetic_slot = item.equipment_slot == "Kinetic Weapons"
            assert (item.damage_type in KINETIC_SLOT_DAMAGE) is kinetic_slot, (name, item.name)
            if item.tier == "Exotic":
                assert any(
                    p.category_identifier == "intrinsics" for p in item.socketed_plug_details
                )
        if item.item_type == "Armor":
            mods = [p.category_identifier or "" for p in item.socketed_plug_details]
            assert all(value.endswith(MOD_SLOTS[item.equipment_slot]) for value in mods), item.name
    for character in context.characters:
        weapons = [i.equipment_slot for i in character.equipped_gear if i.item_type == "Weapon"]
        armor = [i.equipment_slot for i in character.equipped_gear if i.item_type == "Armor"]
        assert sorted(weapons) == ["Energy Weapons", "Kinetic Weapons", "Power Weapons"], name
        assert sorted(armor) == sorted(MOD_SLOTS), (name, character.class_name)
        element = (character.subclass.damage_type or "").casefold()
        for plug in character.subclass.socketed_plug_details:
            assert f".{element}." in f".{plug.category_identifier}.", (name, plug.name)
            prefix = plug.name.split(" of ", 1)[0]
            if prefix in FRAGMENT_ELEMENTS:
                assert FRAGMENT_ELEMENTS[prefix] == element, (name, plug.name)
