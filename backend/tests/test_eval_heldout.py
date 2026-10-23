"""The held-out scenarios stay unseen, runnable, and separate from the main bank."""

import json
import pathlib

from evals.heldout import HELDOUT, KNOWN_ISSUES
from evals.heldout_retired import HELDOUT_RETIRED
from evals.model import AXES
from evals.profiles import EXOTIC_ARMOR, PROFILES, WEAPON_PERKS, WEAPONS, profile_facts
from evals.scenarios import BANK, SUBSETS
from tests.test_eval_bank import _tool_names

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.strip().lower()}
    if isinstance(value, dict):
        return set().union(*(_strings(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_strings(item) for item in value), set())
    return set()


def test_heldout_is_a_separate_subset_of_fifteen_runnable_scenarios() -> None:
    tools = _tool_names()
    assert SUBSETS["heldout"] == HELDOUT
    assert len(HELDOUT) == 15
    assert len({value.id for value in HELDOUT}) == len(HELDOUT)
    assert not {value.id for value in HELDOUT} & {value.id for value in BANK}
    assert all(value not in SUBSETS["full"] for value in HELDOUT)
    for value in HELDOUT:
        assert value.id.startswith("heldout."), value.id
        assert value.profile in PROFILES, value.id
        assert set(value.expect.tools_any) <= tools, value.id
        for axis, tag in value.tags.items():
            assert tag in AXES[axis], (value.id, axis, tag)


def test_heldout_covers_multi_turn_and_unavailable_data() -> None:
    assert sum(len(value.turns) >= 2 for value in HELDOUT) >= 5
    trials = next(value for value in HELDOUT if value.id.endswith("headshots_with_rose"))
    assert trials.expect.by_turn[0].answers_any
    assert {value.expect.style for value in HELDOUT} >= {"fact", "planning", "build"}
    assert len(KNOWN_ISSUES) == 5


def test_no_heldout_prompt_was_seen_in_the_bank_or_saved_drafts() -> None:
    seen = {turn.strip().lower() for value in BANK for turn in value.turns}
    for path in FIXTURES.glob("live_playtest_drafts*.json"):
        seen |= _strings(json.loads(path.read_text("utf-8")))
    for value in HELDOUT:
        for turn in value.turns:
            assert turn.strip().lower() not in seen, (value.id, turn)


def test_retired_scenarios_are_a_separate_regression_subset() -> None:
    retired = {value.id for value in HELDOUT_RETIRED}
    assert SUBSETS["heldout_retired"] == HELDOUT_RETIRED
    assert retired == {
        "heldout.frenzy_somewhere",
        "heldout.coil_still_worth_farming",
        "heldout.exotic_armor_stacking",
        "heldout.pvp_then_dungeon_build",
        "heldout.envious_assassin_elsewhere",
        "heldout.two_exotic_weapons",
        "heldout.no_stat_for_trials",
    }
    assert not retired & {value.id for value in HELDOUT}
    assert all(value not in SUBSETS["full"] for value in HELDOUT_RETIRED)
    assert all(value.note.startswith("retired held-out") for value in HELDOUT_RETIRED)
    runs = {value.id.removeprefix("heldout."): value.note for value in HELDOUT_RETIRED}
    assert "2026-10-22T04:52:27+00:00" in runs["frenzy_somewhere"]
    for name in ("envious_assassin_elsewhere", "two_exotic_weapons", "no_stat_for_trials"):
        assert "2026-10-22T06:01:24+00:00" in runs[name], name
    tools = _tool_names()
    for value in HELDOUT_RETIRED:
        assert value.profile in PROFILES, value.id
        assert set(value.expect.tools_any) <= tools, value.id


def test_replacements_do_not_reuse_retired_prompts_or_subjects() -> None:
    retired_prompts = {turn.strip().lower() for value in HELDOUT_RETIRED for turn in value.turns}
    banned = ("frenzy", "coil", "inmost", "synthoceps", "duality", "shattered throne")
    for value in HELDOUT:
        for turn in value.turns:
            assert turn.strip().lower() not in retired_prompts, (value.id, turn)
    replacements = [
        value
        for value in HELDOUT
        if value.id.removeprefix("heldout.")
        in {
            "xur_stock_now",
            "onslaught_build_then_waves",
            "whirlwind_blade_elsewhere",
            "class_premise_on_vaulted_armor",
            "headshots_with_rose",
        }
    ]
    assert len(replacements) == 5
    for value in replacements:
        assert not any(word in turn.lower() for turn in value.turns for word in banned), value.id


def test_the_second_batch_replacements_reuse_no_retired_subject() -> None:
    banned = (
        "envious",
        "edge transit",
        "ace of spades",
        "gjallarhorn",
        "sunshot",
        "star-eater",
        "trial",
        "streak",
        "k/d",
    )
    new = {"whirlwind_blade_elsewhere", "class_premise_on_vaulted_armor", "headshots_with_rose"}
    chosen = [value for value in HELDOUT if value.id.removeprefix("heldout.") in new]
    assert len(chosen) == 3
    for value in chosen:
        text = " ".join(value.turns).lower()
        assert not any(word in text for word in banned), value.id
    assert sum(len(value.turns) >= 2 for value in HELDOUT) >= 5


def test_replacement_expected_facts_match_the_profiles() -> None:
    titan = profile_facts(PROFILES["titan_veteran"]())
    assert titan.owned_items["Edge Transit"][1] == "equipped:Hunter"
    assert WEAPON_PERKS["Edge Transit"][0] == "Envious Assassin"
    envious = [name for name, perks in WEAPON_PERKS.items() if "Envious Assassin" in perks]
    assert envious == ["Edge Transit"]
    for name in ("Rose", "Calus Mini-Tool", "Apex Predator"):
        assert "Envious Assassin" not in WEAPON_PERKS[name]

    hunter = profile_facts(PROFILES["hunter_veteran"]())
    assert hunter.owned_items["Sunshot"][1] == "equipped:Hunter"
    assert hunter.owned_items["Star-Eater Scales"][1] == "equipped:Hunter"
    for name in ("Ace of Spades", "Gjallarhorn", "Le Monarque"):
        assert hunter.owned_items[name][1] == "vault", name
    for name in ("Ace of Spades", "Sunshot", "Gjallarhorn"):
        assert WEAPONS[name][2] == "Exotic"
    assert WEAPONS["Ace of Spades"][1] != WEAPONS["Sunshot"][1]
    assert WEAPONS["Gjallarhorn"][1] == "Power Weapons"

    warlock = PROFILES["warlock_veteran"]()
    activities = {
        item.name for character in warlock.characters for item in character.available_activities
    }
    assert any("Onslaught" in name for name in activities)
    equipped = profile_facts(warlock).owned_items
    for name in ("Sunbracers", "Sunshot", "Rose", "Falling Guillotine"):
        assert equipped[name][1] == "equipped:Warlock", name


def test_the_curated_guides_do_not_cover_the_no_guide_activity() -> None:
    guides = (pathlib.Path(__file__).parents[1] / "app" / "data" / "destiny_guides.json").read_text(
        "utf-8"
    )
    assert "onslaught" not in guides.lower()


def test_second_batch_expected_facts_match_the_profiles() -> None:
    warlock = PROFILES["warlock_veteran"]()
    facts = profile_facts(warlock)
    # Whirlwind Blade is only on Falling Guillotine, which the Warlock has equipped.
    holders = [name for name, perks in WEAPON_PERKS.items() if "Whirlwind Blade" in perks]
    assert holders == ["Falling Guillotine"]
    assert facts.owned_items["Falling Guillotine"][1] == "equipped:Warlock"
    hunter = next(value for value in warlock.characters if value.class_name == "Hunter")
    assert {item.name for item in hunter.equipped_gear if item.item_type == "Weapon"} == {
        "Rose",
        "Riskrunner",
        "Apex Predator",
    }
    assert "Hunter" in facts.classes and "Titan" in facts.classes
    # Contraverse Hold is a Warlock-only vaulted armor piece, in the gauntlets slot.
    assert EXOTIC_ARMOR["Contraverse Hold"] == ("Warlock", "Gauntlets")
    assert facts.owned_items["Contraverse Hold"] == ("Warlock", "vault")
    assert facts.owned_items["Phoenix Protocol"] == ("Warlock", "vault")
    # No fixture carries a weapon kill or precision count for the headshot question.
    dumped = json.dumps(PROFILES["titan_veteran"]().model_dump(mode="json")).lower()
    assert "headshot" not in dumped and "precision" not in dumped
